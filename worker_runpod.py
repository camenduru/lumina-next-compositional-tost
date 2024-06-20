import os, json, requests, runpod

import math
import random
import traceback

import fairscale.nn.model_parallel.initialize as fs_init
import gradio as gr
import numpy as np
from safetensors.torch import load_file
import torch
import torch.distributed as dist
from torchvision.transforms.functional import to_pil_image

import models
from transport import Sampler, create_transport
from diffusers.models import AutoencoderKL
from transformers import AutoModel, AutoTokenizer

discord_token = os.getenv('com_camenduru_discord_token')
web_uri = os.getenv('com_camenduru_web_uri')
web_token = os.getenv('com_camenduru_web_token')

with torch.inference_mode():
    path_type = "Linear" # ["Linear", "GVP", "VP"]
    prediction = "velocity" # ["velocity", "score", "noise"]
    loss_weight = None # [None, "velocity", "likelihood"]
    sample_eps = None
    train_eps = None
    atol = 1e-6
    rtol = 1e-3
    reverse = None
    likelihood = None
    rank = 0
    num_gpus = 1
    ckpt = "/content/Lumina-T2X/models"
    ema = True
    dtype = torch.bfloat16 #["bf16", "fp32"]

    os.environ["MASTER_PORT"] = str(8080)
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(num_gpus)

    dist.init_process_group("nccl")
    fs_init.initialize_model_parallel(1)
    torch.cuda.set_device(rank)
    train_args = torch.load(os.path.join(ckpt, "model_args.pth"))
    text_encoder = AutoModel.from_pretrained("4bit/gemma-2b", torch_dtype=dtype, device_map="cuda").eval()
    cap_feat_dim = text_encoder.config.hidden_size
    tokenizer = AutoTokenizer.from_pretrained("4bit/gemma-2b")
    tokenizer.padding_side = "right"

    vae = AutoencoderKL.from_pretrained((f"stabilityai/sd-vae-ft-{train_args.vae}" if train_args.vae != "sdxl" else "stabilityai/sdxl-vae"), torch_dtype=torch.float32).cuda()
    model = models.__dict__[train_args.model](
        qk_norm=train_args.qk_norm,
        cap_feat_dim=cap_feat_dim,
    )
    model.eval().to("cuda", dtype=dtype)
    ckpt = load_file(os.path.join(ckpt, f"consolidated{'_ema' if ema else ''}.{rank:02d}-of-{num_gpus:02d}.safetensors"), device="cpu",)
    model.load_state_dict(ckpt, strict=True)

# Adapted from pipelines.StableDiffusionXLPipeline.encode_prompt
def encode_prompt(prompt_batch, text_encoder, tokenizer, proportion_empty_prompts, is_train=True):
    captions = []
    for caption in prompt_batch:
        if random.random() < proportion_empty_prompts:
            captions.append("")
        elif isinstance(caption, str):
            captions.append(caption)
        elif isinstance(caption, (list, np.ndarray)):
            # take a random caption if there are multiple
            captions.append(random.choice(caption) if is_train else caption[0])

    with torch.no_grad():
        text_inputs = tokenizer(
            captions,
            padding=True,
            pad_to_multiple_of=8,
            max_length=256,
            truncation=True,
            return_tensors="pt",
        )

        text_input_ids = text_inputs.input_ids
        prompt_masks = text_inputs.attention_mask

        prompt_embeds = text_encoder(
            input_ids=text_input_ids.cuda(),
            attention_mask=prompt_masks.cuda(),
            output_hidden_states=True,
        ).hidden_states[-2]

    return prompt_embeds, prompt_masks

@torch.inference_mode()
def generate(input):
    values = input["input"]

    cap1 = values['cap1']
    cap2 = values['cap2']
    cap3 = values['cap3']
    cap4 = values['cap4']
    neg_cap = values['neg_cap']
    resolution = values['resolution'] # ["2048x1024 (4x1 Grids)","2560x1024 (4x1 Grids)","3072x1024 (4x1 Grids)","1024x1024 (2x2 Grids)","1536x1536 (2x2 Grids)","2048x2048 (2x2 Grids)","1024x2048 (1x4 Grids)","1024x2560 (1x4 Grids)","1024x3072 (1x4 Grids)",]
    num_sampling_steps = values['num_sampling_steps']
    cfg_scale = values['cfg_scale']
    solver = values['solver'] # ["euler", "midpoint", "rk4"]
    t_shift = values['t_shift']
    seed = values['seed']
    scaling_method = values['scaling_method'] # ["Time-aware", "None"]
    scaling_watershed = values['scaling_watershed']
    proportional_attn = values['proportional_attn']

    with torch.autocast("cuda", dtype):
        try:
            # begin sampler
            transport = create_transport(
                path_type,
                prediction,
                loss_weight,
                train_eps,
                sample_eps,
            )
            sampler = Sampler(transport)
            sample_fn = sampler.sample_ode(
                sampling_method=solver,
                num_steps=num_sampling_steps,
                atol=atol,
                rtol=rtol,
                reverse=reverse,
                time_shifting_factor=t_shift,
            )
            # end sampler

            do_extrapolation = "Extrapolation" in resolution
            split = resolution.split(" ")[1].replace("(", "")
            w_split, h_split = split.split("x")
            resolution = resolution.split(" ")[0]
            w, h = resolution.split("x")
            w, h = int(w), int(h)
            latent_w, latent_h = w // 8, h // 8
            if int(seed) != 0:
                torch.random.manual_seed(int(seed))
            z = torch.randn([1, 4, latent_h, latent_w], device="cuda").to(dtype)
            z = z.repeat(2, 1, 1, 1)

            cap_list = [cap1, cap2, cap3, cap4]
            global_cap = " ".join(cap_list)
            with torch.no_grad():
                if neg_cap != "":
                    cap_feats, cap_mask = encode_prompt(
                        cap_list + [neg_cap] + [global_cap], text_encoder, tokenizer, 0.0
                    )
                else:
                    cap_feats, cap_mask = encode_prompt(
                        cap_list + [""] + [global_cap], text_encoder, tokenizer, 0.0
                    )

            cap_mask = cap_mask.to(cap_feats.device)

            model_kwargs = dict(
                cap_feats=cap_feats[:-1],
                cap_mask=cap_mask[:-1],
                global_cap_feats=cap_feats[-1:],
                global_cap_mask=cap_mask[-1:],
                cfg_scale=cfg_scale,
                h_split_num=int(h_split),
                w_split_num=int(w_split),
            )
            if proportional_attn:
                model_kwargs["proportional_attn"] = True
                model_kwargs["base_seqlen"] = (train_args.image_size // 16) ** 2
            else:
                model_kwargs["proportional_attn"] = False
                model_kwargs["base_seqlen"] = None

            if do_extrapolation and scaling_method == "Time-aware":
                model_kwargs["scale_factor"] = math.sqrt(w * h / train_args.image_size**2)
                model_kwargs["scale_watershed"] = scaling_watershed
            else:
                model_kwargs["scale_factor"] = 1.0
                model_kwargs["scale_watershed"] = 1.0

            samples = sample_fn(z, model.forward_with_cfg, **model_kwargs)[-1]
            samples = samples[:1]

            factor = 0.18215 if train_args.vae != "sdxl" else 0.13025
            samples = vae.decode(samples / factor).sample
            samples = (samples + 1.0) / 2.0
            samples.clamp_(0.0, 1.0)

            img = to_pil_image(samples[0].float())

        except Exception:
            print(traceback.format_exc())

    result = img
    response = None
    try:
        source_id = values['source_id']
        del values['source_id']
        source_channel = values['source_channel']     
        del values['source_channel']
        job_id = values['job_id']
        del values['job_id']
        default_filename = os.path.basename(result)
        files = {default_filename: open(result, "rb").read()}
        payload = {"content": f"{json.dumps(values)} <@{source_id}>"}
        response = requests.post(
            f"https://discord.com/api/v9/channels/{source_channel}/messages",
            data=payload,
            headers={"authorization": f"Bot {discord_token}"},
            files=files
        )
        response.raise_for_status()
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    finally:
        if os.path.exists(result):
            os.remove(result)

    if response and response.status_code == 200:
        try:
            payload = {"jobId": job_id, "result": response.json()['attachments'][0]['url']}
            requests.post(f"{web_uri}/api/notify", data=json.dumps(payload), headers={'Content-Type': 'application/json', "authorization": f"{web_token}"})
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
        finally:
            return {"result": response.json()['attachments'][0]['url']}
    else:
        return {"result": "ERROR"}

runpod.serverless.start({"handler": generate})