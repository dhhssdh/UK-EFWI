## parameter of source location, src, reciver locations and so on.
import numpy as np
from ultils.deepwave_fp import Physics_deepwave
import deepwave
import torch
from ultils.utils import *

gpu_count = torch.cuda.device_count()
print(f"The number of available GPUs is: {gpu_count}")
if torch.cuda.is_available():
    DEVICE = torch.device("cuda:0")  
    print(f"The selected GPU device is: {torch.cuda.get_device_name(DEVICE)}")
else:
    DEVICE = torch.device("cpu")
    print("No available GPUs detected, switched to using CPU")

## model 
path_vp_true = '../obs_and_model_data/vp_true.npy'
path_vs_true = '../obs_and_model_data/vs_true.npy'
path_rho_true = '../obs_and_model_data/rho_true.npy'

path_vp_init = './obs_and_model_data/vp_initial.npy'
path_vs_init = './obs_and_model_data/vs_initial.npy'
path_rho_init = './obs_and_model_data/rho_initial.npy'

#### obs data path
path_vx = './obs_and_model_data/obs_vx.npy'
path_vz = './obs_and_model_data/obs_vy.npy'
lack_low_fre = 'yes'
cut_freq = 5
noise_test = 'yes'
noise_level = 1.0

Physics = Physics_deepwave                                                                                                  
rp_properties = None                                    
model_shape = [128, 401]                                                                                                                         
DT = 0.008                                              
F_PEAK = 5 
T = 3                                             
DH = 20                                                  
N_SHOTS = 30                                             
N_SOURCE_PER_SHOT = 1                                    
inpa = {  
    'ns': N_SHOTS,        
    'sdo': 4,                                     
    'fdom': F_PEAK, 
    'dh': DH,   
    'dt': DT,  
    'acq_type': 1,                                     
    't': T, 
    'npml': 20,                                        
    'pmlR': 1e-5,                                      
    'pml_dir': 2,                                      
    'device': 1,                                       
    'seimogram_shape': '3d',                           
    'energy_balancing': False, 
    "chpr": 70,
}

t_in = str(inpa['t'])
dt_in = str(inpa["dt"])
NT = 800#int( Decimal(t_in) // Decimal(dt_in)  + 1)
print("NT:",NT)
inpa['rec_dis'] =  1 * inpa['dh']  # Define the receivers' distance

offsetx = inpa['dh'] * model_shape[1]
print("offsetx:",offsetx)
depth = inpa['dh'] * model_shape[0]
print("depth:",depth)
surface_loc_x = np.arange(13*inpa["dh"], offsetx-13*inpa["dh"], inpa['dh'], np.float32)

n_surface_rec = len(surface_loc_x)

surface_loc_z = 17 * inpa["dh"] * np.ones(n_surface_rec, np.float32)
surface_loc = np.vstack((surface_loc_x, surface_loc_z)).T
rec_loc_temp = surface_loc
src_loc_temp = np.vstack((
    np.linspace(13*inpa["dh"], offsetx-13*inpa["dh"], N_SHOTS, np.float32),
    2 * inpa["dh"] * np.ones(N_SHOTS, np.float32)
    )).T

src_loc_temp[:, 1] -= 2 * inpa['dh']
# Create the source
N_RECEIVERS = n_surface_rec 
print('N_RECEIVERS:',N_RECEIVERS)
   
# Shot 1 source located at cell [0, 1], shot 2 at [0, 2], shot 3 at [0, 3]
src_loc = torch.zeros(N_SHOTS, N_SOURCE_PER_SHOT, 2,
                        dtype=torch.int, device=DEVICE)
src_loc[:, 0, :] = torch.Tensor(np.flip(src_loc_temp) // DH)

src_loc[:,:,0] = 1
#print(src_loc)

# Receivers located at [0, 1], [0, 2], ... for every shot
rec_loc = torch.zeros(N_SHOTS, N_RECEIVERS, 2,
                        dtype=torch.long, device=DEVICE)
rec_loc[:, :, :] = (
    torch.Tensor(np.flip(rec_loc_temp)/DH)
    ) 
src_old = (
    deepwave.wavelets.ricker(F_PEAK, NT, DT, 1.5 / F_PEAK)
    .repeat(N_SHOTS, N_SOURCE_PER_SHOT, 1)
    .to(DEVICE)
    ) 

### log data
all_loss_data = []
all_loss_vx_model = []
all_loss_vy_model = []
all_loss_rho_model = []
all_loss_model =[]
criteria = torch.nn.L1Loss(reduction='sum')
SNR_vp = []
SSIM_vp = []
Loss_vp = []
ERROR_vp = []
SNR_vs = []
SSIM_vs = []
Loss_vs = []
ERROR_vs = []
SNR_rho = []
SSIM_rho = []
Loss_rho = []
ERROR_rho = []
time_each_iter = []

#save_path

vp_save_path = './reconstruction/efwi_good_init/fre_loss_6_and_noise5_tikh/vp/'
vs_save_path = './reconstruction/efwi_good_init/fre_loss_6_and_noise5_tikh/vs/'
rho_save_path = './reconstruction/efwi_good_init/fre_loss_6_and_noise5_tikh/rho/'

if not os.path.exists(vp_save_path):
    os.makedirs(vp_save_path)
if not os.path.exists(vs_save_path):
    os.makedirs(vs_save_path)
if not os.path.exists(rho_save_path):
    os.makedirs(rho_save_path)