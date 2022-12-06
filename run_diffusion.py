# 安装diffusion 
# conda create -n diffusion python=3.8
# pip install diffusers==0.3.0
import cv2
from diffusers import DDPMPipeline
import numpy as np

image_pipe = DDPMPipeline.from_pretrained("google/ddpm-celebahq-256")
image_pipe.to("cuda")
images = image_pipe().images  
image0 = np.array(images[0])[:,:,::-1]  # BGR2RGB
# 展示下生成的某张image长啥样~.
cv2.imwrite('./image0.jpg', image0)