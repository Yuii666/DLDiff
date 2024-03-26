# DLDiff: Image Detail-guided Latent Diffusion Model for Low-Light Image Enhancement

## 🧊 Dataset
You can refer to the following links to download the [LOLv1](https://drive.google.com/file/d/157bjO1_cFuSd0HWDUuAmcHRJDVyWpOxB/view) as training data set.

### Data Loader
If you want to change it, feel free to modify the /ldm/ldm/data/PIL_data.py to change the data loading format.
During training, you need to replace the path of the training data set in the content of /DLDiff-main/ldm/config/low2light.yaml

## 🛠️ Environment
If you already have the ldm environment, please skip it

A suitable conda environment named ldm can be created and activated with:
```
conda env create -f environment.yaml
conda activate low2high
```

## 🌟 Pretrained Model
You can refer to the following links to download the sampling_model and the trainning_model available at [Baidu Netdisk](https://pan.baidu.com/s/1WOHXsovv1Dd5lbKEhU9VWg?pwd=rmk9)(rmk9) or [Google Drive](https://drive.google.com/drive/folders/1XxzJszsZiICG-Gr5kXi1Mc5QNKIamsHm?usp=drive_link).
Among them, the sampling_mode is used to predict results, and trainning_model is a pre-trained model used for model training.
The pretrained model should be saved in the ./checkpoints/

## 🖥️ Inference
### Prepare Testing Data:
you can download the [LSRW](https://pan.baidu.com/s/1XHWQAS0ZNrnCyZ-bq7MKvA)(code: wmrr) and the [LOLv2-real](https://drive.google.com/file/d/1dzuLCk9_gE2bFF222n3-7GVUlSVHpMYC/view)
### Testing
Run the follwing codes:
```
bash predict.sh
```
The testing results will be saved in the ./results folder.
The code includes modules for measuring PSNR, SSIM, FID, LPIPS, and time indicators. For FID measurement, ensure to download the ViT-B-32.pt model to the ./clip_model folder.

## 🧑‍💻 Train
### Training with a 3090 GPU
Run the follwing codes:
```
bash train.sh
```
You can modify the paths of the config and checkpoints in the train.sh script.
Example usage:
```
CUDA_VISIBLE_DEVICES=2 python main.py --base ldm/config/low2light.yaml --resume /DLDiff-main/checkpoints/train.ckpt --no_test False -t --gpus 0,
```

