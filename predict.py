import  os, sys
sys.path.append(os.getcwd()+"/ldm")
from omegaconf import OmegaConf
from PIL import Image
import numpy as np
import torch
from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler


import os, sys
import cv2
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
import os
import torch
import torchvision.transforms as T
import torchvision.transforms as transforms
import math
from clean_fid_main.cleanfid import fid
import lpips
import time
 


transform_PIL = T.ToPILImage()


# config
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
yaml_path="ldm/config/low2light.yaml"
model_path="/DLDiff-main/checkpoints/DLDiff.ckpt"

##create model
def create_model(device):
    #load config and checkpoint
    config = OmegaConf.load(yaml_path)
    config.model['params']['ckpt_path']=model_path
    
    model = instantiate_from_config(config.model)
    sampler = DDIMSampler(model)
    model = model.to(device)

    return model,sampler

def get_tensor(normalize=False, toTensor=True):
    transform_list = []
    if toTensor:
        transform_list += [transforms.ToTensor()]

    if normalize:
        transform_list += [transforms.Normalize((0.5, 0.5, 0.5),
                                                (0.5, 0.5, 0.5))]
    return transforms.Compose(transform_list)


def process_data(image,filename):
     
    low_image=np.array(image.convert("RGB"))

    low_image = low_image.astype(np.float32) / 255.0#
    low_image = low_image[None].transpose(0,3,1,2)
    low_image = torch.from_numpy(low_image)
   
    # # ---

    attn_img = np.ones((low_image.shape[0], low_image.shape[1], 1), dtype=np.float32)
    attn_img = attn_img / 255.0#
    attn_img = attn_img[None].transpose(0,3,1,2)
    attn_img = torch.from_numpy(attn_img)
    

    batch = { "attn_tensor": attn_img, "low_image_tensor": low_image}
    for k in batch:
        batch[k] = batch[k] * 2.0 - 1.0

    return batch

def calculate_psnr(img1, img2, border=0):
    img1=np.array(img1)
    img2=np.array(img2)
    img1 = cv2.cvtColor(img1,cv2.COLOR_RGB2BGR)
    img2 = cv2.cvtColor(img2,cv2.COLOR_RGB2BGR)
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    h, w = img1.shape[:2]
    img1 = img1[border:h-border, border:w-border]
    img2 = img2[border:h-border, border:w-border]

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2)**2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(255.0 / math.sqrt(mse))


def calculate_ssim(img1, img2, border=0):
    '''calculate SSIM
    the same outputs as MATLAB's
    img1, img2: [0, 255]
    '''
    img1=np.array(img1)
    img2=np.array(img2)
    img1 = cv2.cvtColor(img1,cv2.COLOR_RGB2BGR)
    img2 = cv2.cvtColor(img2,cv2.COLOR_RGB2BGR)
    #img1 = img1.squeeze()
    #img2 = img2.squeeze()
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    h, w = img1.shape[:2]
    img1 = img1[border:h-border, border:w-border]
    img2 = img2[border:h-border, border:w-border]

    if img1.ndim == 2:
        return ssim(img1, img2)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            ssims = []
            for i in range(3):
                ssims.append(ssim(img1[:,:,i], img2[:,:,i]))
            return np.array(ssims).mean()
        elif img1.shape[2] == 1:
            return ssim(np.squeeze(img1), np.squeeze(img2))
    else:
        raise ValueError('Wrong input image dimensions.')


def ssim(img1, img2):
    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()




if __name__ == "__main__":
    model,sampler=create_model(device)

    img_path="/Data/LSRW_Eval/All/low/"
    gt_path="/Data/LSRW_Eval/All/high/"
    # img_path='/Data/LOL_v2/Test/Low/'
    # gt_path = '/Data/LOL_v2/Test/Normal/'
    gt_path2='/DLDiff-main/results/LSRW_gt/'
    save_path = '/DLDiff-main/results/LSRW/'
    filenames=os.listdir(img_path)
    filenames.sort()
    psnr_all = 0
    ssim_all=0
    fid_all= 0
    time_all=0
    loss_fn = lpips.LPIPS(net='vgg', version=0.1)
    time_list=[]
 

    for filename in filenames:
        path = img_path+filename
        # LSRW
        gt_image = Image.open(path.replace('low','high')).convert("RGB")
        # LOLv2
        # gt_image = Image.open(path.replace('Low','Normal').replace('low','normal')).convert("RGB")

        start_time = time.time()

        load_image=Image.open(path).convert("RGB")
        input_width=(load_image.size[0]//64)*64
        input_hight=(load_image.size[1]//64)*64
        load_image = load_image.resize((input_width,input_hight))

            

        batch=process_data(load_image,filename)
        mask_tensor=batch["attn_tensor"]
        masked_image_tensor=batch["low_image_tensor"]
        c = model.cond_stage_model.encode(masked_image_tensor.to(device))
        cc = torch.nn.functional.interpolate(mask_tensor.to(device),size=c.shape[-2:])
        c = torch.cat((c, cc), dim=1)
        shape = (c.shape[1]-1,)+c.shape[2:]
        print(filename)

        samples_ddim, _ = sampler.sample(S=5,conditioning=c,batch_size=c.shape[0],shape=shape, verbose=False)
        x_samples_ddim = model.decode_first_stage(samples_ddim.to(device))
        predicted_image_clamped = torch.clamp((x_samples_ddim+1.0)/2.0,min=0.0, max=1.0)
        output_PIL=transform_PIL(predicted_image_clamped[0]).resize((gt_image.size[0],gt_image.size[1]),Image.BICUBIC)

        end_time = time.time()
        time_=end_time-start_time
        time_list.append(time_)
        time_all+=time_

        psnr_ = calculate_psnr(gt_image,output_PIL)
        ssim_ = calculate_ssim(gt_image,output_PIL)

        psnr_all+=psnr_
        ssim_all+=ssim_
        output_PIL.save(os.path.join(save_path,filename.replace('low','normal')))
        gt_image.save(os.path.join(gt_path2,filename.replace('low','normal')))

    


    psnr_avg=psnr_all/len(filenames)
    ssim_avg=ssim_all/len(filenames)
    time_avg=(time_all-time_list[0])/len(filenames)
    print(f'平均Time(s)：{time_avg:.3f}')
    print(f'平均PSNR：{psnr_avg:.2f}，平均SSIM：{ssim_avg:.4f}')

    
    score = fid.compute_fid(save_path, gt_path2)
    print(f"平均FID：:{score:.3f}")

    loss_fn = lpips.LPIPS(net='vgg', version=0.1).to(device)
 
# the total list of images
    files = os.listdir(save_path)
    i = 0
    total_lpips_distance = 0
    average_lpips_distance = 0
    for file in files:
    
        try:
            # Load images
            img0 = lpips.im2tensor(lpips.load_image(os.path.join(save_path,file))).to(device)
            img1 = lpips.im2tensor(lpips.load_image(os.path.join(gt_path2,file))).to(device)
    
            # Compute distance
            current_lpips_distance = loss_fn.forward(img0, img1)
            total_lpips_distance = total_lpips_distance + current_lpips_distance
    
            # print('%s: %.3f'%(file, current_lpips_distance))
    
        except Exception as e:
            print(e)
    
    average_lpips_distance = float(total_lpips_distance) / len(files)
    print(f"平均LPIPS：:{average_lpips_distance:.3f}")
