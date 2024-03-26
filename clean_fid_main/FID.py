

# from cleanfid import fid
# score = fid.compute_fid('/home/ubuntu/Low-image/Diffusion-Low-Light-main/data/LL_dataset/BAID/test/resize_gt', '/home/ubuntu/Low-image/Diffusion-Low-Light-main-copy/results/test/zhanshiv17/LOLv1')
# print(f"s:{score:3f}")




from cleanfid import fid
score = fid.compute_fid('/home/ubuntu/Project/latent-diffusion-inpainting-main/predict/lolv2-2', '/home/ubuntu/Project/latent-diffusion-inpainting-main/predict/lolv2_gt/')
print(f"s:{score:3f}")