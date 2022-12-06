import torch
from torchvision import utils
# git link: https://github.com/lucidrains/denoising-diffusion-pytorch.git
from denoising_diffusion_pytorch import Unet, GaussianDiffusion, Trainer
import os
import cv2

model = Unet(
    dim = 64,
    dim_mults = (1, 2, 4, 8)
).cuda()

diffusion = GaussianDiffusion(
    model,
    image_size = 128,
    timesteps = 1000,   # number of steps
    sampling_timesteps = 250,
    loss_type = 'l1'    # L1 or L2
).cuda()

training_folder = '/newdata/jiachen/project/diffusion/cat/'

trainer = Trainer(
    diffusion,
    training_folder,
    train_batch_size = 32,
    train_lr = 4e-4,
    train_num_steps = 500000,          # total training steps
    save_and_sample_every = 10000,
    fp16 = True,
    gradient_accumulate_every = 1,    # gradient accumulation steps
    ema_decay = 0.995,                # exponential moving average decay
    amp = True                        # turn on mixed precision
)

print('Start Training!')
trainer.train()
