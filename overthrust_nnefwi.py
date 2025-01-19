import os
os.chdir('./')
import time
#time.sleep(100*60)
import matplotlib.pyplot as plt 
import os
import torch
import numpy as np
import torch.nn as nn
from typing import List, Tuple, Optional
import matplotlib.pyplot as plt
import random 
from decimal import Decimal
import warnings
import torch.nn.functional as F
from functools import partial
warnings.filterwarnings('ignore')
os.environ['KMP_DUPLICATE_LIB_OK']='True'

from ultils.deepwave_fp import Physics_deepwave
from parameters_efwi import *
from ultils.utils import *
from kan_model.unet import U_Net
from kan_model.kan_unet import KANU_Net

vp_true,vs_true,rho_true = load_true_models(path_vp_true, path_vs_true, path_rho_true)
vp_true,vs_true,rho_true = vp_true.to(DEVICE), vs_true.to(DEVICE), rho_true.to(DEVICE)

vp_initial, vs_initial, rho_initial = load_init_models(path_vp_init,path_vs_init,path_rho_init)

if lack_low_fre == 'yes':
    src_new = seismic_filter(data=src_old.cpu(), \
                           filter_type='highpass',freqmin=cut_freq, \
                           freqmax=None,df=1/DT,corners=16)
    src_new = torch.tensor(src_new).to(torch.float32).to(DEVICE)
else:
    src_new = src_old
src_new = src_new.to(torch.float32).to(DEVICE)
print(src_new.dtype)
#print(vp_true.dtype)
print('wavelets shape:',src_new.shape)

physics = Physics(inpa['dh'], inpa['dt'],inpa['fdom'] ,size=deepwave_size,src=src_new,
                        src_loc=src_loc, rec_loc=rec_loc
                        )
taux_est = physics(vp_true,vs_true,rho_true)  
d_obs_vx = taux_est[0]
d_obs_vy = taux_est[1]


if noise_test == 'yes':
    d_obs_vx = add_gaussian_noise(d_obs_vx,noise_level = noise_level)
    d_obs_vy = add_gaussian_noise(d_obs_vy,noise_level = noise_level)
else:
    d_obs_vx = d_obs_vx
    d_obs_vy = d_obs_vy


vp_initial = vp_initial.to(DEVICE)
vs_initial = vs_initial.to(DEVICE)
rho_initial = rho_initial.to(DEVICE)
vp = vp_initial.requires_grad_(True)
vs = vs_initial.requires_grad_(True)
rho = rho_initial.requires_grad_(True)


criteria = torch.nn.L1Loss(reduction='sum')
model = KANU_Net(n_channels=1,n_classes=1,bilinear=True, device=DEVICE)#  U_Net(n_channels=1,n_classes=1)#  
model = model.to(DEVICE)
model.train()

lr = 1.0e-2
LR_MILESTONE = 550
optim_nnefwi = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.98, 0.999))
scheduler_optim_nnefwi = torch.optim.lr_scheduler.StepLR(optim_nnefwi, LR_MILESTONE, gamma=0.5)


### nnefwi loop

print("###################################################################")
print("##########                                               ##########")
print("##########             NNEFWI WITH ADAM IS RUNNING       ##########")
print("##########                                               ##########")
print("##########                                               ##########")
print("###################################################################")

for iter in range(total_iter):
    time_each_bath_start = time.time()
    loss_data, mp_sq, ms_sq,mrho_sq, model  = train_fun(
        Physics=Physics, 
        model=model,
        deepwave_size=deepwave_size,
        rho_initial=rho_initial,
        vx_initial=vp_initial,
        vy_initial=vs_initial,
        d_obs_vx=d_obs_vx,
        d_obs_vy=d_obs_vy,
        optim_nnefwi=optim_nnefwi, 
        criteria=criteria,
        mini_batches = mini_batches,
        src_loc=src_loc, 
        rec_loc=rec_loc, 
        src=src_new,
        inpa=inpa,
        vp_scale=vp_scale,
        vs_scale=vs_scale,
        rho_scale=rho_scale)

    all_loss_data.append(loss_data) 
    time_each_bath_end = time.time()
    time_each_iter.append(time_each_bath_end - time_each_bath_start)

    with torch.no_grad():
        all_loss_vx_model.append(
            criteria(mp_sq, vp_true).item()
        )
        all_loss_vy_model.append(
            criteria(ms_sq, vs_true).item()
        )
        all_loss_rho_model.append(
            criteria(mrho_sq, rho_true).item()
        )
        all_loss_model.append(
            criteria(mp_sq, vp_true).item() + criteria(ms_sq, vs_true).item()+ criteria(mrho_sq, rho_true).item()
        )
    snr_vp = ComputeSNR(mp_sq.detach().cpu().numpy(), \
                  vp_true.detach().cpu().numpy())
    SNR_vp = np.append(SNR_vp, snr_vp)
    snr_vs = ComputeSNR(ms_sq.detach().cpu().numpy(), \
                  vs_true.detach().cpu().numpy())
    SNR_vs = np.append(SNR_vs, snr_vs)
    snr_rho = ComputeSNR(mrho_sq.detach().cpu().numpy(), \
                  rho_true.detach().cpu().numpy())
    SNR_rho = np.append(SNR_rho, snr_rho)
 
    ssim_vp = ComputeSSIM(mp_sq.detach().cpu().numpy(), \
                  vp_true.detach().cpu().numpy())
    SSIM_vp = np.append(SSIM_vp, ssim_vp)
    ssim_vs = ComputeSSIM(ms_sq.detach().cpu().numpy(), \
                  vs_true.detach().cpu().numpy())
    SSIM_vs = np.append(SSIM_vs, ssim_vs)
    ssim_rho = ComputeSSIM(mrho_sq.detach().cpu().numpy(), \
                  rho_true.detach().cpu().numpy())
    SSIM_rho = np.append(SSIM_rho, ssim_rho)

    rerror_vp = ComputeRE(mp_sq.detach().cpu().numpy(), \
                  vp_true.detach().cpu().numpy())
    ERROR_vp = np.append(ERROR_vp, rerror_vp)
    rerror_vs = ComputeRE(ms_sq.detach().cpu().numpy(), \
                  vs_true.detach().cpu().numpy())
    ERROR_vs = np.append(ERROR_vs, rerror_vs)
    rerror_rho = ComputeRE(mrho_sq.detach().cpu().numpy(), \
                  rho_true.detach().cpu().numpy())
    ERROR_rho = np.append(ERROR_rho, rerror_rho)
    
    if (iter+1)%10 == 0:
        print(f"Iteration {iter + 1} = loss: {all_loss_data[-1]:.4f},model loss: {all_loss_model[-1]:.4f},time:{time_each_iter[-1]:.2f},snr_vp:{SNR_vp[-1]:.3f},snr_vs:{SNR_vs[-1]:.3f},snr_rho:{SNR_rho[-1]:.3f}")
        
    if (iter+1)%50 == 0:
        np.save(vp_nn_save_path + 'recx_iter_%s.npy' %(iter+1), mp_sq.cpu().detach().numpy(), 2)
        np.save(vs_nn_save_path + 'recx_iter_%s.npy' %(iter+1), ms_sq.cpu().detach().numpy(), 2)
        np.save(rho_nn_save_path + 'recx_iter_%s.npy' %(iter+1), mrho_sq.cpu().detach().numpy(), 2)
    scheduler_optim_nnefwi.step()


## save log_data
with torch.no_grad():
    np.savetxt(main_path_nn+'all_loss_data.txt', all_loss_data,delimiter=',')
    
    np.savetxt(main_path_nn+'all_loss_model.txt', all_loss_model, delimiter=',')
    np.savetxt(main_path_nn+'all_loss_vp_model.txt', all_loss_vx_model, delimiter=',')
    np.savetxt(main_path_nn+'all_loss_vs_model.txt', all_loss_vy_model, delimiter=',')
    np.savetxt(main_path_nn+'all_loss_rho_model.txt', all_loss_rho_model, delimiter=',')
    
    np.savetxt(main_path_nn+'vp_snr.txt', SNR_vp,delimiter=',')
    np.savetxt(main_path_nn+'vs_snr.txt', SNR_vs,delimiter=',')
    np.savetxt(main_path_nn+'rho_snr.txt', SNR_rho,delimiter=',')
    
    np.savetxt(main_path_nn+'vp_ssim.txt', SSIM_vp, delimiter=',')
    np.savetxt(main_path_nn+'vs_ssim.txt', SSIM_vs, delimiter=',')
    np.savetxt(main_path_nn+'rho_ssim.txt', SSIM_rho, delimiter=',')
    
    np.savetxt(main_path_nn+'error_vp.txt',ERROR_vp , delimiter=',')
    np.savetxt(main_path_nn+'error_vs.txt',ERROR_vs , delimiter=',')
    np.savetxt(main_path_nn+'error_rho.txt',ERROR_rho , delimiter=',')
    
    np.savetxt(main_path_nn+'time_unet.txt',time_each_iter , delimiter=',')