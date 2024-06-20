import json
import math
import os
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

PORT = int(os.getenv('server_port'))
os.environ["MASTER_PORT"] = str(PORT+1)
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
def generate(command):
    values = json.loads(command)

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

    # master_port = values['master_port']
    # rank = values['rank']
    # request_queue = values['request_queue']
    # response_queue = values['response_queue']
    # mp_barrier = values['mp_barrier']
    # path_type = values['path_type']
    # prediction = values['prediction']
    # loss_weight = values['loss_weight']
    # train_eps = values['train_eps']
    # sample_eps = values['sample_eps']
    # atol = values['atol']
    # rtol = values['rtol']
    # reverse = values['reverse']

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
            img.save("/content/out.png")
            return img

        except Exception:
            print(traceback.format_exc())

import gradio as gr

with gr.Blocks(title=f"sdxl-turbo", css=".gradio-container {max-width: 544px !important}", analytics_enabled=False) as demo:
    with gr.Row():
      with gr.Column():
          textbox = gr.Textbox(show_label=False, value="""
            {   
                "cap1": "A colossal ancient robot stands amidst the ruins of a forgotten civilization. Its metallic body is covered in intricate carvings and symbols, showing signs of age and wear.",
                "cap2": "A colossal ancient robot stands amidst the ruins of a forgotten civilization. Its metallic body is covered in intricate carvings and symbols, showing signs of age and wear. ",
                "cap3": "A winding countryside path meanders through rolling green hills, lined with wildflowers and tall grasses swaying in the breeze.",
                "cap4": "A quaint countryside cottage bathed in the warm glow of the setting sun. The small house is surrounded by a lush garden filled with blooming flowers and tall, swaying grass.",
                "neg_cap": "low quality",
                "resolution": "2048x1024 (4x1 Grids)",
                "num_sampling_steps": 30,
                "cfg_scale": 4.0,
                "solver": "midpoint",
                "t_shift": 4,
                "seed": 0,
                "scaling_method": "Time-aware",
                "scaling_watershed": 0.3,
                "proportional_attn": true
            }
          """)
          button = gr.Button()
    with gr.Row(variant="default"):
        output_image = gr.Image(
            show_label=False,
            format=".png",
            type="pil",
            interactive=False,
            height=512,
            width=512,
            elem_id="output_image",
        )

    button.click(fn=generate, inputs=[textbox], outputs=[output_image], show_progress=True)

demo.queue().launch(inline=False, share=False, debug=True, server_name='0.0.0.0', server_port=PORT)