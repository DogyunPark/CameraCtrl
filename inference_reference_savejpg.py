import argparse
import json
import os
import cv2
#from moviepy.editor import VideoFileClip

import numpy as np
import torch
from tqdm import tqdm
from packaging import version as pver
from einops import rearrange
from safetensors import safe_open

from omegaconf import OmegaConf
from diffusers import (
    AutoencoderKL,
    DDIMScheduler
)
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers.pipelines.stable_diffusion.convert_from_ckpt import convert_ldm_vae_checkpoint, \
    convert_ldm_clip_checkpoint

from cameractrl.utils.util import save_videos_grid, save_videos_jpg
from cameractrl.models.unet import UNet3DConditionModelPoseCond
from cameractrl.models.pose_adaptor import CameraPoseEncoder
from cameractrl.pipelines.pipeline_animation import CameraCtrlPipeline
from cameractrl.utils.convert_from_ckpt import convert_ldm_unet_checkpoint
from cameractrl.data.dataset import Camera


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def custom_meshgrid(*args):
    # ref: https://pytorch.org/docs/stable/generated/torch.meshgrid.html?highlight=meshgrid#torch.meshgrid
    if pver.parse(torch.__version__) < pver.parse('1.10'):
        return torch.meshgrid(*args)
    else:
        return torch.meshgrid(*args, indexing='ij')


def get_relative_pose(cam_params):
    abs_w2cs = [cam_param.w2c_mat for cam_param in cam_params]
    abs_c2ws = [cam_param.c2w_mat for cam_param in cam_params]
    cam_to_origin = 0
    target_cam_c2w = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, -cam_to_origin],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])
    abs2rel = target_cam_c2w @ abs_w2cs[0]
    ret_poses = [target_cam_c2w, ] + [abs2rel @ abs_c2w for abs_c2w in abs_c2ws[1:]]
    ret_poses = np.array(ret_poses, dtype=np.float32)
    return ret_poses


def ray_condition(K, c2w, H, W, device):
    # c2w: B, V, 4, 4
    # K: B, V, 4

    B = K.shape[0]

    j, i = custom_meshgrid(
        torch.linspace(0, H - 1, H, device=device, dtype=c2w.dtype),
        torch.linspace(0, W - 1, W, device=device, dtype=c2w.dtype),
    )
    i = i.reshape([1, 1, H * W]).expand([B, 1, H * W]) + 0.5  # [B, HxW]
    j = j.reshape([1, 1, H * W]).expand([B, 1, H * W]) + 0.5  # [B, HxW]

    fx, fy, cx, cy = K.chunk(4, dim=-1)  # B,V, 1

    zs = torch.ones_like(i)  # [B, HxW]
    xs = (i - cx) / fx * zs
    ys = (j - cy) / fy * zs
    zs = zs.expand_as(ys)

    directions = torch.stack((xs, ys, zs), dim=-1)  # B, V, HW, 3
    directions = directions / directions.norm(dim=-1, keepdim=True)  # B, V, HW, 3

    rays_d = directions @ c2w[..., :3, :3].transpose(-1, -2)  # B, V, 3, HW
    rays_o = c2w[..., :3, 3]  # B, V, 3
    rays_o = rays_o[:, :, None].expand_as(rays_d)  # B, V, 3, HW
    # c2w @ dirctions
    rays_dxo = torch.cross(rays_o, rays_d)
    plucker = torch.cat([rays_dxo, rays_d], dim=-1)
    plucker = plucker.reshape(B, c2w.shape[1], H, W, 6)  # B, V, H, W, 6
    # plucker = plucker.permute(0, 1, 4, 2, 3)
    return plucker


def load_personalized_base_model(pipeline, personalized_base_model):
    print(f'Load civitai base model from {personalized_base_model}')
    if personalized_base_model.endswith(".safetensors"):
        dreambooth_state_dict = {}
        with safe_open(personalized_base_model, framework="pt", device="cpu") as f:
            for key in f.keys():
                dreambooth_state_dict[key] = f.get_tensor(key)
    elif personalized_base_model.endswith(".ckpt"):
        dreambooth_state_dict = torch.load(personalized_base_model, map_location="cpu")

    # 1. vae
    converted_vae_checkpoint = convert_ldm_vae_checkpoint(dreambooth_state_dict, pipeline.vae.config)
    pipeline.vae.load_state_dict(converted_vae_checkpoint)
    # 2. unet
    converted_unet_checkpoint = convert_ldm_unet_checkpoint(dreambooth_state_dict, pipeline.unet.config)
    _, unetu = pipeline.unet.load_state_dict(converted_unet_checkpoint, strict=False)
    assert len(unetu) == 0
    # 3. text_model
    pipeline.text_encoder = convert_ldm_clip_checkpoint(dreambooth_state_dict, text_encoder=pipeline.text_encoder)
    del dreambooth_state_dict
    return pipeline


def get_pipeline(ori_model_path, unet_subfolder, image_lora_rank, image_lora_ckpt, unet_additional_kwargs,
                 unet_mm_ckpt, pose_encoder_kwargs, attention_processor_kwargs,
                 noise_scheduler_kwargs, pose_adaptor_ckpt, personalized_base_model, gpu_id):
    vae = AutoencoderKL.from_pretrained(ori_model_path, subfolder="vae")
    tokenizer = CLIPTokenizer.from_pretrained(ori_model_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(ori_model_path, subfolder="text_encoder")
    unet = UNet3DConditionModelPoseCond.from_pretrained_2d(ori_model_path, subfolder=unet_subfolder,
                                                           unet_additional_kwargs=unet_additional_kwargs)
    pose_encoder = CameraPoseEncoder(**pose_encoder_kwargs)
    print(f"Setting the attention processors")
    unet.set_all_attn_processor(add_spatial_lora=image_lora_ckpt is not None,
                                add_motion_lora=False,
                                lora_kwargs={"lora_rank": image_lora_rank, "lora_scale": 1.0},
                                motion_lora_kwargs={"lora_rank": -1, "lora_scale": 1.0},
                                **attention_processor_kwargs)

    if image_lora_ckpt is not None:
        print(f"Loading the lora checkpoint from {image_lora_ckpt}")
        lora_checkpoints = torch.load(image_lora_ckpt, map_location=unet.device)
        if 'lora_state_dict' in lora_checkpoints.keys():
            lora_checkpoints = lora_checkpoints['lora_state_dict']
        _, lora_u = unet.load_state_dict(lora_checkpoints, strict=False)
        assert len(lora_u) == 0
        print(f'Loading done')

    if unet_mm_ckpt is not None:
        print(f"Loading the motion module checkpoint from {unet_mm_ckpt}")
        mm_checkpoints = torch.load(unet_mm_ckpt, map_location=unet.device)
        _, mm_u = unet.load_state_dict(mm_checkpoints, strict=False)
        assert len(mm_u) == 0
        print("Loading done")

    print(f"Loading pose adaptor")
    pose_adaptor_checkpoint = torch.load(pose_adaptor_ckpt, map_location='cpu')
    pose_encoder_state_dict = pose_adaptor_checkpoint['pose_encoder_state_dict']
    pose_encoder_m, pose_encoder_u = pose_encoder.load_state_dict(pose_encoder_state_dict)
    assert len(pose_encoder_u) == 0 and len(pose_encoder_m) == 0
    attention_processor_state_dict = pose_adaptor_checkpoint['attention_processor_state_dict']
    _, attn_proc_u = unet.load_state_dict(attention_processor_state_dict, strict=False)
    assert len(attn_proc_u) == 0
    print(f"Loading done")

    noise_scheduler = DDIMScheduler(**OmegaConf.to_container(noise_scheduler_kwargs))
    vae.to(gpu_id)
    text_encoder.to(gpu_id)
    unet.to(gpu_id)
    pose_encoder.to(gpu_id)
    pipe = CameraCtrlPipeline(
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        unet=unet,
        scheduler=noise_scheduler,
        pose_encoder=pose_encoder)
    if personalized_base_model is not None:
        load_personalized_base_model(pipeline=pipe, personalized_base_model=personalized_base_model)
    pipe.enable_vae_slicing()
    pipe = pipe.to(gpu_id)

    return pipe


def main(args):
    os.makedirs(args.out_root, exist_ok=True)
    video_pth = '{}/video'.format(args.out_root)
    image_pth = '{}/image'.format(args.out_root)
    os.makedirs(video_pth, exist_ok=True)
    os.makedirs(image_pth, exist_ok=True)
    rank = args.local_rank
    #setup_for_distributed(rank == 0)
    gpu_id = rank % torch.cuda.device_count()
    model_configs = OmegaConf.load(args.model_config)
    unet_additional_kwargs = model_configs[
        'unet_additional_kwargs'] if 'unet_additional_kwargs' in model_configs else None
    noise_scheduler_kwargs = model_configs['noise_scheduler_kwargs']
    pose_encoder_kwargs = model_configs['pose_encoder_kwargs']
    attention_processor_kwargs = model_configs['attention_processor_kwargs']

    eval_listdir = [x for x in os.listdir(args.eval_datadir)]
    filtered_eval_listdir = eval_listdir[:1000]

    sample_pairs = []
    for idx, listdir in enumerate(filtered_eval_listdir):
        filedir = '{}/{}'.format(args.eval_datadir, listdir)
        eval_file = [x for x in os.listdir(filedir)]

        #Target video base path
        base_path = filedir

        text_prompt_file = '{}/{}'.format(filedir, eval_file[-1])
        with open(text_prompt_file, 'r') as f:
            caption = f.readlines()[0]
        
        video_file = '{}/target_1_video.mp4'.format(filedir)
        cap = cv2.VideoCapture(video_file)
        if not cap.isOpened():
            print("Error: Could not open video file.")
            exit()
        
        frame_number = 0
        
        # Read and save each frame as a JPG file
        while True:
            ret, frame = cap.read()
            
            # If the frame was not grabbed, break the loop
            if not ret:
                print("End of video reached.")
                break
            
            # Construct the filename for each frame (e.g., frame_0001.jpg)
            #resized_frame = cv2.resize(frame, (args.image_width, args.image_height))
            resized_frame = frame
            frame_filename = os.path.join('/hub_data1/dogyun/reference_video/', f'{listdir}_target_1_video_frame_{frame_number:04d}.jpg')
            
            # Save the frame as a JPG file
            cv2.imwrite(frame_filename, resized_frame)
            
            # Print out the saved frame for feedback
            print(f"Saved: {frame_filename}")

            # Increment the frame counter
            frame_number += 1
        # Release the video capture object
        cap.release()

        video_file = '{}/target_2_video.mp4'.format(filedir)
        cap = cv2.VideoCapture(video_file)
        if not cap.isOpened():
            print("Error: Could not open video file.")
            exit()
        
        frame_number = 0
        
        # Read and save each frame as a JPG file
        while True:
            ret, frame = cap.read()
            
            # If the frame was not grabbed, break the loop
            if not ret:
                print("End of video reached.")
                break
            
            # Construct the filename for each frame (e.g., frame_0001.jpg)
            #resized_frame = cv2.resize(frame, (args.image_width, args.image_height))
            resized_frame = frame
            frame_filename = os.path.join('/hub_data1/dogyun/reference_video/', f'{listdir}_target_2_video_frame_{frame_number:04d}.jpg')
            
            # Save the frame as a JPG file
            cv2.imwrite(frame_filename, resized_frame)
            
            # Print out the saved frame for feedback
            print(f"Saved: {frame_filename}")

            # Increment the frame counter
            frame_number += 1
        # Release the video capture object
        cap.release()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_root", type=str)
    parser.add_argument("--eval_datadir", type=str)
    parser.add_argument("--image_height", type=int, default=256)
    parser.add_argument("--image_width", type=int, default=384)
    parser.add_argument("--video_length", type=int, default=16)
    parser.add_argument("--ori_model_path", type=str, help='path to the sd model folder')
    parser.add_argument("--unet_subfolder", type=str, help='subfolder name of unet ckpt')
    parser.add_argument("--motion_module_ckpt", type=str, help='path to the animatediff motion module ckpt')
    parser.add_argument("--image_lora_rank", type=int, default=2)
    parser.add_argument("--image_lora_ckpt", default=None)
    parser.add_argument("--personalized_base_model", default=None)
    parser.add_argument("--pose_adaptor_ckpt", default=None, help='path to the camera control model ckpt')
    parser.add_argument("--model_config", type=str)
    parser.add_argument("--num_inference_steps", type=int, default=25)
    parser.add_argument("--guidance_scale", type=float, default=14.0)
    parser.add_argument("--visualization_captions", required=True, help='prompts path, json or txt')
    parser.add_argument("--use_negative_prompt", action='store_true', help='whether to use negative prompts')
    parser.add_argument("--use_specific_seeds", action='store_true', help='whether to use specific seeds for each prompt')
    parser.add_argument("--trajectory_file", required=True, help='txt file')
    parser.add_argument("--original_pose_width", type=int, default=1280, help='the width of the video used to extract camera trajectory')
    parser.add_argument("--original_pose_height", type=int, default=720, help='the height of the video used to extract camera trajectory')
    parser.add_argument("--n_procs", type=int, default=8)

    # DDP args
    parser.add_argument("--world_size", default=1, type=int,
                        help="number of the distributed processes.")
    parser.add_argument('--local_rank', type=int, default=-1,
                        help='Replica rank on the current node. This field is required '
                             'by `torch.distributed.launch`.')
    args = parser.parse_args()
    main(args)
