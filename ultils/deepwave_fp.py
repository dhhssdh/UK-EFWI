#"forward modeling using Deepwave pakeage"
import torch
import deepwave
import torch.nn as nn

class Physics_deepwave(nn.Module):
    def __init__(self, dh, dt, F_PEAK,size,
                 src,src_loc, rec_loc,rp_properties=None):
        super(Physics_deepwave, self).__init__()
        self.dh = dh
        self.dt = dt
        self.src = src
        self.src_loc = src_loc
        self.rec_loc = rec_loc
        self.F_PEAK = F_PEAK
        self.size = size
        rp_properties = rp_properties
    
    def forward(self, vp,vs,rho):
        out = deepwave.elastic(
            *deepwave.common.vpvsrho_to_lambmubuoyancy(vp, vs,rho),
            self.dh, self.dt,
            source_amplitudes_y=self.src,
            source_amplitudes_x=self.src,
            source_locations_y=self.src_loc,
            source_locations_x=self.src_loc,
            receiver_locations_y=self.rec_loc,
            receiver_locations_x=self.rec_loc,
            pml_freq=self.F_PEAK
            )
        vx = out[15]
        vy = out[14]
        return vx.permute(0, 2, 1).unsqueeze(0),vy.permute(0, 2, 1).unsqueeze(0)