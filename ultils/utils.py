"""
Common function
 

@author: fangshuyang (yangfs@hit.edu.cn)

"""

import torch
import deepwave
import numpy as np
import scipy
import scipy.io as spio
from torch import autograd
import torch.nn.functional as F
from torch.autograd import Variable
from torch.autograd import Function
import math
from math import exp
from IPython.core.debugger import set_trace
import scipy.stats
import warnings
from scipy.fftpack import hilbert
from scipy.signal import (cheb2ord, cheby2, convolve, get_window, iirfilter,
                          remez)
from scipy.signal import sosfilt
from scipy.signal import zpk2sos
from scipy.fft import fft, fftfreq
from scipy.signal import butter, lfilter
#from parameters_efwi import *


def seismic_filter(data,filter_type,freqmin,freqmax,df,corners,zerophase=False,axis=-1):
    """
    create the fileter for removing the frequency component of seismic data 
    """
    assert filter_type.lower() in ['bandpass', 'lowpass', 'highpass']

    if filter_type == 'bandpass':
        if freqmin and freqmax and df:
            filt_data = bandpass(data, freqmin, freqmax, df, corners, zerophase, axis)
        else:
            raise ValueError
    if filter_type == 'lowpass':
        if freqmax and df:
            filt_data = lowpass(data, freqmax, df, corners, zerophase, axis)
        else:
            raise ValueError
    if filter_type == 'highpass':
        if freqmin and df:
            filt_data = highpass(data, freqmin, df, corners, zerophase, axis)
        else:
            raise ValueError
    return filt_data



    
def bandpass(data, freqmin, freqmax, df, corners, zerophase, axis):
    """
    Butterworth-Bandpass Filter.
    Filter data from ``freqmin`` to ``freqmax`` using ``corners``
    corners.
    The filter uses :func:`scipy.signal.iirfilter` (for design)
    and :func:`scipy.signal.sosfilt` (for applying the filter).
    :type data: numpy.ndarray
    :param data: Data to filter.
    :param freqmin: Pass band low corner frequency.
    :param freqmax: Pass band high corner frequency.
    :param df: Sampling rate in Hz.
    :param corners: Filter corners / order.
    :param zerophase: If True, apply filter once forwards and once backwards.
        This results in twice the filter order but zero phase shift in
        the resulting filtered trace.
    :return: Filtered data.
    """
    fe = 0.5 * df
    low = freqmin / fe
    high = freqmax / fe
    # raise for some bad scenarios
    if high - 1.0 > -1e-6:
        msg = ("Selected high corner frequency ({}) of bandpass is at or "
               "above Nyquist ({}). Applying a high-pass instead.").format(
            freqmax, fe)
        warnings.warn(msg)
        return highpass(data, freq=freqmin, df=df, corners=corners,
                        zerophase=zerophase)
    if low > 1:
        msg = "Selected low corner frequency is above Nyquist."
        raise ValueError(msg)
    z, p, k = iirfilter(corners, [low, high], btype='band',
                        ftype='butter', output='zpk')
    sos = zpk2sos(z, p, k)
    if zerophase:
        firstpass = sosfilt(sos, data, axis)
        return sosfilt(sos, firstpass[::-1], axis)[::-1]
    else:
        return sosfilt(sos, data, axis)

    
def lowpass(data, freq, df, corners, zerophase, axis):
    """
    Butterworth-Lowpass Filter.
    Filter data removing data over certain frequency ``freq`` using ``corners``
    corners.
    The filter uses :func:`scipy.signal.iirfilter` (for design)
    and :func:`scipy.signal.sosfilt` (for applying the filter).
    :type data: numpy.ndarray
    :param data: Data to filter.
    :param freq: Filter corner frequency.
    :param df: Sampling rate in Hz.
    :param corners: Filter corners / order.
    :param zerophase: If True, apply filter once forwards and once backwards.
        This results in twice the number of corners but zero phase shift in
        the resulting filtered trace.
    :return: Filtered data.
    """
    fe = 0.5 * df
    f = freq / fe
    # raise for some bad scenarios
    if f > 1:
        f = 1.0
        msg = "Selected corner frequency is above Nyquist. " + \
              "Setting Nyquist as high corner."
        warnings.warn(msg)
    z, p, k = iirfilter(corners, f, btype='lowpass', ftype='butter',
                        output='zpk')
    sos = zpk2sos(z, p, k)
    if zerophase:
        firstpass = sosfilt(sos, data, axis)
        return sosfilt(sos, firstpass[::-1], axis)[::-1]
    else:
        return sosfilt(sos, data, axis)


def highpass(data, freq, df, corners, zerophase, axis):
    """
    Butterworth-Highpass Filter.
    Filter data removing data below certain frequency ``freq`` using
    ``corners`` corners.
    The filter uses :func:`scipy.signal.iirfilter` (for design)
    and :func:`scipy.signal.sosfilt` (for applying the filter).
    :type data: numpy.ndarray
    :param data: Data to filter.
    :param freq: Filter corner frequency.
    :param df: Sampling rate in Hz.
    :param corners: Filter corners / order.
    :param zerophase: If True, apply filter once forwards and once backwards.
        This results in twice the number of corners but zero phase shift in
        the resulting filtered trace.
    :return: Filtered data.
    """
    fe = 0.5 * df
    f = freq / fe
    # raise for some bad scenarios
    if f > 1:
        msg = "Selected corner frequency is above Nyquist."
        raise ValueError(msg)
    z, p, k = iirfilter(corners, f, btype='highpass', ftype='butter',
                        output='zpk')
    sos = zpk2sos(z, p, k)
    if zerophase:
        firstpass = sosfilt(sos, data, axis)
        return sosfilt(sos, firstpass[::-1], axis)[::-1]
    else:
        return sosfilt(sos, data, axis)

def load_true_models(path_vp_true,path_vs_true,path_rho_true):
    vp = torch.load(path_vp_true)
    vs = torch.load(path_vs_true)
    rho = torch.load(path_rho_true)
    return vp,vs,rho

def load_init_models(path_vp_init,path_vs_init,path_rho_init):
    vp = torch.load(path_vp_init)
    vs = torch.load(path_vs_init)
    rho = torch.load(path_rho_init)
    return vp,vs,rho

def load_obs(path_vx,path_vz):
    vx = torch.load(path_vx)
    vz = torch.load(path_vz)
    
    return vx,vz
    
def createlearnSNR(init_snr_guess,device):
    """
        create learned snr when amplitude is noisy and try to learn the noise
    """
    learn_snr_init = torch.tensor(init_snr_guess)
    learn_snr = learn_snr_init.clone()
    learn_snr = learn_snr.to(device)
    #set_trace()
    learn_snr.requires_grad = True
    
    return learn_snr, learn_snr_init
      

    
    
def gaussian(window_size, sigma):
    """
    gaussian filter
    """
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel):
    """
    create the window for computing the SSIM
    """
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window     = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window


def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1    = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2    = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2
    L  = 255
    C1 = (0.01*L) ** 2
    C2 = (0.03*L) ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)



def ComputeSSIM(img1, img2, window_size=11, size_average=True):
    """
    compute the SSIM between img1 and img2
    """
    img1 = Variable(torch.from_numpy(img1))
    img2 = Variable(torch.from_numpy(img2))
    
    if len(img1.size()) == 2:
        d = img1.size()
        img1 = img1.view(1,1,d[0],d[1])
        img2 = img2.view(1,1,d[0],d[1])
    elif len(img1.size()) == 3:
        d = img1.size()
        img1 = img1.view(d[2],1,d[0],d[1])
        img2 = img2.view(d[2],1,d[0],d[1]) 
    else:
        raise Exception('The shape of image is wrong!!!')
    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)


def ComputeSNR(rec,target):
    """
       Calculate the SNR between reconstructed image and true  image
    """
    if torch.is_tensor(rec):
        rec    = rec.cpu().data.numpy()
        target = target.cpu().data.numpy()
    
    if len(rec.shape) != len(target.shape):
        raise Exception('Please reshape the Rec and Target to correct Dimension!!')
    
    snr = 0.0
    if len(rec.shape) == 3:
        for i in range(rec.shape[0]):
            rec_ind     = rec[i,:,:].reshape(np.size(rec[i,:,:]))
            target_ind  = target[i,:,:].reshape(np.size(rec_ind))
            s      = 10*np.log10(sum(target_ind**2)/sum((rec_ind-target_ind)**2))
            snr    = snr + s
        snr = snr/rec.shape[0]
    elif len(rec.shape) == 2:
        rec       = rec.reshape(np.size(rec))
        target    = target.reshape(np.size(rec))
        snr       = 10*np.log10(sum(target**2)/sum((rec-target)**2))
    else:
        raise Exception('Please reshape the Rec to correct Dimension!!')
    return snr

def ComputeRSNR(rec,target):
    """
       Calculate the regressed-SNR(RSNR) between reconstructed image and true  image
    """
    if torch.is_tensor(rec):
        rec    = rec.cpu().data.numpy()
        target = target.cpu().data.numpy()
    
    if len(rec.shape) != len(target.shape):
        raise Exception('Please reshape the Rec and Target to correct Dimension!!')
    
    rec_ind     = rec.reshape(np.size(rec))
    target_ind  = target.reshape(np.size(rec))
    slope,intercept, _, _, _ = scipy.stats.linregress(rec_ind,target_ind)
    r           = slope*rec_ind + intercept
    rsnr        = 10*np.log10(sum(target_ind**2)/sum((r-target_ind)**2))
    
    if len(rec.shape) == 2:
        rec  = r.reshape(rec.shape[0],rec.shape[1])
    elif len(rec.shape) == 3:
        rec  = r.reshape(rec.shape[0],rec.shape[1],rec.shape[2])
    else:
        raise Exception('Wrong shape of reconstruction!!!')
    return rsnr, rec

def ComputeRE(rec,target):
    """
    Compute relative error between the rec and target
    """
    if torch.is_tensor(rec):
        rec    = rec.cpu().data.numpy()
        target = target.cpu().data.numpy()
    
    if len(rec.shape) != len(target.shape):
        raise Exception('Please reshape the Rec and Target to correct Dimension!!')
       
    rec    = rec.reshape(np.size(rec))
    target = target.reshape(np.size(rec))
    rerror = np.sqrt(sum((target-rec)**2)) / np.sqrt(sum(target**2))
    
    return rerror

    
def add_gaussian_noise(tensor,noise_level):  ## add sdt of noise
    noise = torch.randn_like(tensor)*tensor.std()*noise_level
    return tensor + noise
