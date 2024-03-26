import torch
import os

import cv2

import sys
from PIL import Image
import numpy as np

path = "/home/ubuntu/all_data/LOL_v2/Test/Normal/"
path2 = "/home/ubuntu/Project/latent-diffusion-inpainting-main/predict/lolv2_gt/"
filnames=os.listdir(path)
filnames.sort()
for filename in filnames:

    if os.path.splitext(filename)[1] == '.jpg':

#     # print(filename)
#         # if 'fake' in filename:
#         img = cv2.imread(path + filename)

#         # print(filename.replace(".jpg",".png"))

#         # newfilename = 'Flickr2K_'+filename[:-6]+'.png'
        # newfilename = filename.replace('low','normal')
        
#         # print(newfilename)
#         cv2.imwrite(path2 + newfilename,img)
#             # cv2.imwrite(path2 + filename,img)

        image = Image.open(path+filename).convert("RGB").resize((256,256))
        image.save(path2+filename)