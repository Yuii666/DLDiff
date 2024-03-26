import os
import torch
import cv2
import numpy as np
from PIL import Image
from omegaconf import OmegaConf
from main import instantiate_from_config
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from utils.cross import DPM

transform = transforms.ToPILImage()
def get_tensor(normalize=False, toTensor=True):
    transform_list = []
    if toTensor:
        transform_list += [transforms.ToTensor()]

    if normalize:
        transform_list += [transforms.Normalize((0.5, 0.5, 0.5),
                                                (0.5, 0.5, 0.5))]
    return transforms.Compose(transform_list)

def attn(attn_cro,filename):
    attn_cro_gray=attn_cro.detach().numpy()
    attn_cro_gray = cv2.cvtColor(attn_cro_gray, cv2.COLOR_BGR2GRAY)
    height=attn_cro_gray.shape[0]
    width=attn_cro_gray.shape[1]
    dst_cro = np.zeros((height,width,1),np.uint8)
    soblex=cv2.Sobel(attn_cro_gray,cv2.CV_64F,1,0,ksize=3) 
    sobley=cv2.Sobel(attn_cro_gray,cv2.CV_64F,0,1,ksize=3)
    dst=cv2.addWeighted(soblex,0.5,sobley,0.5,0)
    for i in range(0,dst.shape[0]-2):
        for j in range(0,dst.shape[1]-2):
            if dst[i,j] > 50:
                dst_cro[i,j] = 255
            else:
                dst_cro[i,j] = 0
    # filedir='/home/ubuntu/Project/latent-diffusion-inpainting-main/attn_img/'
    # os.makedirs(filedir, exist_ok=True)
    # cv2.imwrite(filedir+filename[:-4]+'attn.png', dst_cro)
    return dst_cro

def P2CV(input):
    img = cv2.cvtColor(np.asarray(input), cv2.COLOR_RGB2BGR)
    return img

def CV2P(input):
    img = Image.fromarray(cv2.cvtColor(input,cv2.COLOR_BGR2RGB))
    return img


def laplacian(image,normal_address):
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    image_lap = cv2.filter2D(image, cv2.CV_8UC3, kernel)
    # filedir='/home/ubuntu/Project/latent-diffusion-inpainting-main/lap_img/'
    # os.makedirs(filedir, exist_ok=True)
    # cv2.imwrite(filedir+normal_address[:-4]+'attn.png', image_lap)

    return image_lap

class Low2highTrain_ldm(Dataset):
    def __init__(self, size, data_root, config=None):
        self.size = size
        self.config = config or OmegaConf.create()
        self.data_root=data_root
        self.normal_list = []
        self.cross=DPM(3,64)
        
        for normal in os.listdir(data_root):
            self.normal_list.append(normal)

                    
    def __len__(self):
        return len(self.normal_list)


    def __getitem__(self, i):
        
        normal_address=self.normal_list[i]
        
        
        image = np.array(Image.open(self.data_root+"/"+normal_address).convert("RGB").resize((256,256)))
        image = image.astype(np.float32) / 255.0#
        image = torch.from_numpy(image)
        # print(self.data_root.replace('Normal','Low')+"/"+self.images[i])
        # low_image=np.array(Image.open(self.data_root.replace('Normal','Low')+"/"+normal_address.replace('normal','low')).convert("RGB").resize((256,256)))
        low_image=np.array(Image.open(self.data_root.replace('high','low')+"/"+normal_address).convert("RGB").resize((256,256)))
        # =====FourierFeat======
        # B = torch.randn(2,512)
        # fre_fea=input_mapping(low_image, B)
        # =====
        lap_image=CV2P(laplacian(P2CV(low_image),normal_address))
        # gray_image=lap_image.convert("L")   

        # mask = get_tensor()(gray_image)
        attn_img = get_tensor()(lap_image)
        # mask = get_tensor()(low_image)
        
        low_image = low_image.astype(np.float32) / 255.0#
        low_image = torch.from_numpy(low_image)
        
        attn_img = self.cross(attn_img).permute(1,2,0)*255
        attn_img = attn(attn_img,normal_address)
        attn_img = attn_img / 255.0
        attn_img = torch.from_numpy(np.array(attn_img))



        batch = {"image": np.squeeze(image,0), "attn_img": np.squeeze(attn_img,0),"low_image": np.squeeze(low_image,0)}
        for k in batch:
            batch[k] = batch[k] * 2.0 - 1.0

        return batch
