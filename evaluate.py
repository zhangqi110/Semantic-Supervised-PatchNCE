import os
import random
import torch
import numpy as np
from skimage import io
from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity
from tqdm import tqdm
from torchvision.transforms.functional import to_tensor 
from pytorch_fid.fid_score import calculate_activation_statistics, calculate_frechet_distance
from pytorch_fid.inception import InceptionV3

from util.perceptual import PerceptualHashValue

import warnings

warnings.filterwarnings('ignore')

# val_epoch = [ 'val_20', 'val_25', 'val_30', 'val_35', 'val_40']
val_epoch = [ 'val_5', 'val_10', 'val_15','val_20','val_25','val_30','val_35','val_40' ]
# val_epoch = ["val_20","val_25", 'val_30', 'val_35', 'val_40', "val_45", "val_50"]
# val_epoch = ["val_40","val_45", 'val_50', 'val_55', 'val_60', "val_65", "val_70", "val_75", "val_80"]
# val_epoch = ["val_30","val_35"]

# val_epoch = ['val_40' ]
for name in val_epoch:
    print(f"next ans: {name}")
    # MINST
    targ_dir = f'/mnt/bn/liuchen-lq-0228/zhangqi/AdaptiveSupervisedPatchNCE/results/ssp_test_wsrc_1/{name}/images/real_B'
    pred_dir = f'/mnt/bn/liuchen-lq-0228/zhangqi/AdaptiveSupervisedPatchNCE/results/ssp_test_wsrc_1/{name}/images/fake_B'
    # BCI
    # targ_dir = f'/mnt/bn/liuchen-lq-0228/zhangqi/AdaptiveSupervisedPatchNCE/results/BCI_baseline/{name}/images/real_B'
    # pred_dir = f'/mnt/bn/liuchen-lq-0228/zhangqi/AdaptiveSupervisedPatchNCE/results/BCI_baseline/{name}/images/fake_B'
    img_list = [f for f in os.listdir(pred_dir) if f.endswith(('png', 'jpg'))]
    img_format = '.' + img_list[0].split('.')[-1]
    img_list = [f.replace('.png', '').replace('.jpg', '') for f in img_list]
    random.seed(0)
    random.shuffle(img_list)

    # PHV statistics
    device = torch.device('cuda' if (torch.cuda.is_available()) else 'cpu')
    layers = ['layer_1', 'layer_2', 'layer_3', 'layer_4']
    PHV = PerceptualHashValue(
            T=0.01, network='resnet50', layers=layers, 
            resize=False, resize_mode='bilinear',
            instance_normalized=False).to(device)
    all_phv = []
    for i in tqdm(img_list):
        fake = io.imread(os.path.join(pred_dir, i + img_format))
        real = io.imread(os.path.join(targ_dir, i + img_format))

        fake = to_tensor(fake).to(device)
        real = to_tensor(real).to(device)

        phv_list = PHV(fake, real)
        all_phv.append(phv_list)
    all_phv = np.array(all_phv)
    all_phv = np.mean(all_phv, axis=0)
    res_str = ''
    for layer, value in zip(layers, all_phv):
        res_str += f'{layer}: {value:.4f} '
    print(res_str)
    print(np.round(all_phv, 4))

    # FID statistics
    device = torch.device('cuda' if (torch.cuda.is_available()) else 'cpu')
    num_avail_cpus = len(os.sched_getaffinity(0))
    num_workers = min(num_avail_cpus, 8)

    real_paths = [os.path.join(targ_dir, f + img_format) for f in img_list]
    fake_paths = [os.path.join(pred_dir, f + img_format) for f in img_list]
    print(f"Total number of images: {len(real_paths)}")

    dims = 2048
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
    model = InceptionV3([block_idx]).to(device)

    m1, s1 = calculate_activation_statistics(real_paths, model, batch_size=10, dims=dims,
                                        device=device, num_workers=num_workers)

    m2, s2 = calculate_activation_statistics(fake_paths, model, batch_size=10, dims=dims,
                                        device=device, num_workers=num_workers)

    fid_value = calculate_frechet_distance(m1, s1, m2, s2)

    print(f'FID: {fid_value:.2f}')

    # KID statistics
    command = f'python3 -W ignore util/kid_score.py --true {targ_dir} --fake {pred_dir}'
    os.system(command)

    # PSNR and SSIM statistics
    psnr = []
    ssim = []
    for i in tqdm(img_list):
        fake = io.imread(os.path.join(pred_dir, i + img_format))
        real = io.imread(os.path.join(targ_dir, i + img_format))
        PSNR = peak_signal_noise_ratio(fake, real)
        psnr.append(PSNR)
        SSIM = structural_similarity(fake, real, multichannel=True)
        ssim.append(SSIM)
    average_psnr = sum(psnr)/len(psnr)
    average_ssim = sum(ssim)/len(ssim)
    print(pred_dir)
    print("The average psnr is " + str(average_psnr))
    print("The average ssim is " + str(average_ssim))
    print(f"{average_psnr:.4f} {average_ssim:.4f}")
