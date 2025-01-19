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
#os.environ['KMP_DUPLICATE_LIB_OK']='True'

from ultils.deepwave_fp import Physics_deepwave
from parameters_efwi import *
from ultils.utils import *
#from ultils.efwi_adam import efwi


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

optimer = torch.optim.Adam([{'params': [vp], 'lr': 1.5},
                                            {'params': [vs], 'lr': 1.5},
                                           {'params': [rho], 'lr': 0.5}])

### efwi loop

print("###################################################################")
print("##########                                               ##########")
print("##########             EFWI WITH ADAM IS RUNNING         ##########")
print("##########                                               ##########")
print("##########                                               ##########")
print("###################################################################")
t_start = time.time()

for iter in range(total_iter):
    loss_data_minibatch = []
    time_each_bath_start = time.time()
    for batch in range(mini_batches):
        loss_freqs = []
       
        optimer.zero_grad()
        
        src_loc_batch = src_loc[batch::mini_batches].to(DEVICE)
        rec_loc_batch = rec_loc[batch::mini_batches].to(DEVICE)
        src_batch = src_new[batch::mini_batches].to(DEVICE)
                
        physics = Physics(inpa['dh'], inpa['dt'],inpa['fdom'] ,size=deepwave_size,src=src_batch,
                        src_loc=src_loc_batch, rec_loc=rec_loc_batch
                        )
       
        vp = vp.to(DEVICE)
        vs = vs.to(DEVICE)
        rho = rho.to(DEVICE)
    
        taux_est = physics(vp,vs,rho) 
        taux_vx_est_filtered = taux_est[0].to(DEVICE)
        taux_vy_est_filtered = taux_est[1].to(DEVICE)
 
        taux_est_all = torch.cat((taux_vx_est_filtered,taux_vy_est_filtered),dim=1).to(DEVICE)
        
        d_obs_vx_filtered = d_obs_vx[:, batch::mini_batches].to(DEVICE)
        d_obs_vy_filtered = d_obs_vy[:, batch::mini_batches].to(DEVICE)
        d_obs_filtered_all = torch.cat((d_obs_vx_filtered,d_obs_vy_filtered),dim=1).to(DEVICE)

        loss_data = 1.0e8*criteria(taux_est_all, d_obs_filtered_all)#+tikhLoss# +(1-weight)*tikhLoss+weight*tvloss
        loss = loss_data   
        loss.backward()
        optimer.step()
        loss_freqs.append(loss_data.cpu().detach().numpy())
        loss_data_minibatch.append(np.mean(loss_freqs))
    all_loss_data.append(loss_data) 
    time_each_bath_end = time.time()
    time_each_iter.append(time_each_bath_end - time_each_bath_start)

    with torch.no_grad():
        all_loss_vx_model.append(
            criteria(vp.cpu(),vp_true.cpu()).detach().numpy().item()
        )
        
        all_loss_vy_model.append(
            criteria(vs.cpu(),vs_true.cpu()).detach().numpy().item()
        )
        all_loss_rho_model.append(
            criteria(rho.cpu(),rho_true.cpu()).detach().numpy().item()
        )
        all_loss_model.append(
            criteria(vp, vp_true).item()+ criteria(vs, vs_true).item()+ criteria(rho, rho_true).item()
        )
    
    snr_vp = ComputeSNR(vp.detach().cpu().numpy(), \
                  vp_true.detach().cpu().numpy())
    SNR_vp = np.append(SNR_vp, snr_vp)
    snr_vs = ComputeSNR(vs.detach().cpu().numpy(), \
                  vs_true.detach().cpu().numpy())
    SNR_vs = np.append(SNR_vs, snr_vs)
    snr_rho = ComputeSNR(rho.detach().cpu().numpy(), \
                  rho_true.detach().cpu().numpy())
    SNR_rho = np.append(SNR_rho, snr_rho)
 
    ssim_vp = ComputeSSIM(vp.detach().cpu().numpy(), \
                  vp_true.detach().cpu().numpy())
    SSIM_vp = np.append(SSIM_vp, ssim_vp)
    ssim_vs = ComputeSSIM(vs.detach().cpu().numpy(), \
                  vs_true.detach().cpu().numpy())
    SSIM_vs = np.append(SSIM_vs, ssim_vs)
    ssim_rho = ComputeSSIM(rho.detach().cpu().numpy(), \
                  rho_true.detach().cpu().numpy())
    SSIM_rho = np.append(SSIM_rho, ssim_rho)

    rerror_vp = ComputeRE(vp.detach().cpu().numpy(), \
                  vp_true.detach().cpu().numpy())
    ERROR_vp = np.append(ERROR_vp, rerror_vp)
    rerror_vs = ComputeRE(vs.detach().cpu().numpy(), \
                  vs_true.detach().cpu().numpy())
    ERROR_vs = np.append(ERROR_vs, rerror_vs)
    rerror_rho = ComputeRE(rho.detach().cpu().numpy(), \
                  rho_true.detach().cpu().numpy())
    ERROR_rho = np.append(ERROR_rho, rerror_rho)   
    
    if (iter+1)%1 == 0:
        # print(f"Iteration {iter + 1}, loss: {all_loss_data[-1]},model loss:{all_loss_model[-1]},tvnorm:{tvloss},tinorm:{tikhLoss}")
        print(f"Iteration {iter + 1} = loss: {all_loss_data[-1]:.4f},model loss: {all_loss_model[-1]:.4f},time:{time_each_iter[-1]:.2f},snr_vp:{SNR_vp[-1]:.3f},snr_vs:{SNR_vs[-1]:.3f},snr_rho:{SNR_rho[-1]:.3f}")
        
    if (iter+1)%50 == 0:
        np.save(vp_save_path + 'recx_iter_%s.npy' %(iter+1), vp.cpu().detach().numpy(), 2)
        np.save(vs_save_path + 'recx_iter_%s.npy' %(iter+1), vs.cpu().detach().numpy(), 2)
        np.save(rho_save_path + 'recx_iter_%s.npy' %(iter+1), rho.cpu().detach().numpy(), 2)
t_end = time.time()
elapsed_time = t_end - t_start
print('Running complete in {:.0f}m  {:.0f}s' .format(elapsed_time //60 , elapsed_time % 60))


#### save log_data

with torch.no_grad():
    #print(all_loss_data)

    all_loss_data_save = np.array([tensor.cpu().numpy() for tensor in all_loss_data])
    np.savetxt(main_path+'all_loss_data.txt', all_loss_data_save,delimiter=',')
    
    np.savetxt(main_path+'all_loss_model.txt', all_loss_model, delimiter=',')
    np.savetxt(main_path+'all_loss_vp_model.txt', all_loss_vx_model, delimiter=',')
    np.savetxt(main_path+'all_loss_vs_model.txt', all_loss_vy_model, delimiter=',')
    np.savetxt(main_path+'all_loss_rho_model.txt', all_loss_rho_model, delimiter=',')
    
    np.savetxt(main_path+'vp_snr.txt', SNR_vp,delimiter=',')
    np.savetxt(main_path+'vs_snr.txt', SNR_vs,delimiter=',')
    np.savetxt(main_path+'rho_snr.txt', SNR_rho,delimiter=',')
    
    np.savetxt(main_path+'vp_ssim.txt', SSIM_vp, delimiter=',')
    np.savetxt(main_path+'vs_ssim.txt', SSIM_vs, delimiter=',')
    np.savetxt(main_path+'rho_ssim.txt', SSIM_rho, delimiter=',')
    
    np.savetxt(main_path+'error_vp.txt',ERROR_vp , delimiter=',')
    np.savetxt(main_path+'error_vs.txt',ERROR_vs , delimiter=',')
    np.savetxt(main_path+'error_rho.txt',ERROR_rho , delimiter=',')

    np.savetxt(main_path+'time_efwi.txt',time_each_iter , delimiter=',')

