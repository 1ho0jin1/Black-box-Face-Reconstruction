# Black-box-Face-Reconstruction
TOWARDS QUERY EFFICIENT AND GENERALIZABLE BLACK-BOX FACE RECONSTRUCTION ATTACK
To be presented in 2023 International Conference on Image Processing (ICIP)

## Requirements (not strict)
- PyTorch 1.9.1
- Torchvision 0.10.1
- CUDA 10.1/10.2
- NOTE: Any version is ok if you can use the StyleGAN2 from the <a href="https://github.com/rosinality/stylegan2-pytorch"> stylegan2-pytorch</a> repository.

## Setup
Download pretrained encoders and StyleGAN2 weights:
- <a href="https://drive.google.com/file/d/1eVq2hhjHiO494qkDcGhG5EdxYOilu--7/view?usp=share_link">VGGNet-19</a>
- <a href="https://drive.google.com/file/d/1pDOX9_bQAgSkJp8W-EVq4iKBg07gTQLE/view?usp=drivesdk">ResNet-50</a>
- <a href="https://drive.google.com/file/d/1BDDpjhUYCwQde6KzR2ztGkMqgE8Nq9E2/view?usp=share_link">SwinTransformer-S</a>
- <a href="https://drive.google.com/file/d/1W4ZmSxm3gROz205JoikqVeHRroM2_fXY/view?usp=share_link">StyleGAN2-FFHQ-256x256</a>

Download LFW and CFP-FP datasets:
- <a href="https://drive.google.com/file/d/1lckCEDPjOFAyJRjpdWnfseqI50_yEXAW/view?usp=share_link">LFW</a>
- <a href="https://drive.google.com/file/d/1s769SGpacLQ3qDx413RVtRbYQrJfu0M3/view?usp=share_link">CFP-FP</a>

The images for LFW and CFP-FP datasets are already cropped and aligned using two different schemes: <a href="https://github.com/timesler/facenet-pytorch" target="_blank">MTCNN by timesler</a> and <a href="https://github.com/JDAI-CV/FaceX-Zoo/issues/30"> FaceX-Zoo</a>.

After downloading, change the paths in ```dataset/dataset_conf.yaml``` and ```weight``` in ```encoder/encoder_conf.yaml``` accordingly.

## Usage
After the setup is done, simply run ```python reconstruct.py```.

## TODO
The paper for this work will be uploaded on ArXiv upon its acceptance to ICIP 2023.
