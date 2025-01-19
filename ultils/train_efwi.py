import torch
import numpy as np
def train_deepwave(Physics, 
                   model, 
                   deepwave_size,
                   rho_initial,
                   vx_initial,
                   vy_initial,
                   d_obs_vx,
                   d_obs_vy,
                   optim_nnefwi, 
                   criteria, 
                   mini_batches,
                   src_loc, 
                   rec_loc, 
                   src,
                   inpa,
                   vp_scale,
                   vs_scale,
                   rho_scale):
    loss_data_minibatch = []

    for batch in range(mini_batches):
        loss_freqs = []

        optim_nnefwi.zero_grad()
        earth_model_vp, earth_model_vs, earth_model_rho = model(vx_initial, vy_initial,
                                                                                         rho_initial)
        #if rec_source.shape == source_initial.shape:
        #    rec_source = 0.8*rec_source.requires_grad_(True)
        #else:
        #    print('Error for source output')
        src_loc_batch = src_loc[batch::mini_batches]
        rec_loc_batch = rec_loc[batch::mini_batches]
        src_batch = src[batch::mini_batches]

        # Initialize Physics object
        physics = Physics(inpa['dh'], inpa['dt'], inpa['fdom'], size=deepwave_size, src=src_batch,
                          src_loc=src_loc_batch, rec_loc=rec_loc_batch)

        # Earth model prediction from the transformer decoder
        device = earth_model_vp.device
        size_a = int(vx_initial.shape[0])
        size_b = int(vx_initial.shape[1])

        if earth_model_vp.view(size_a, size_b).shape == vx_initial.squeeze().shape and \
           earth_model_vs.view(size_a, size_b).shape == vy_initial.squeeze().shape:

            # Apply scaling and adjustments to the models
            vp = earth_model_vp.view(size_a, size_b) * vp_scale + vx_initial
            vs = earth_model_vs.view(size_a, size_b) * vs_scale + vy_initial
            rho = earth_model_rho.view(size_a, size_b) * rho_scale + rho_initial
            # Ensure models are trainable
            vp = vp.requires_grad_(True)
            vs = vs.requires_grad_(True)
            rho = rho.requires_grad_(True)
        else:
            print('The initial velocity model and the expected velocity model are of different sizes.')

        # Move models to device
        vp, vs, rho = vp.to(device), vs.to(device), rho.to(device)

        # Forward pass through the physics model
        mp, ms, mrho = vp, vs, rho
        taux_est = physics(mp, ms, mrho)
        taux_vx_est_filtered = taux_est[0]
        taux_vy_est_filtered = taux_est[1]

        # Combine estimated and observed data
        taux_est_all = torch.cat((taux_vx_est_filtered, taux_vy_est_filtered), dim=1)
        d_obs_vx_filtered = d_obs_vx[:, batch::mini_batches]
        d_obs_vy_filtered = d_obs_vy[:, batch::mini_batches]
        d_obs_filtered_all = torch.cat((d_obs_vx_filtered, d_obs_vy_filtered), dim=1)

        # Calculate loss
        loss_data = 1.0e8 * criteria(taux_est_all, d_obs_filtered_all)  # l2 norm
        loss = loss_data
        loss.backward()
        optim_nnefwi.step()

        # Store loss
        loss_freqs.append(loss_data)
        #print(loss_freqs.dtype)
        #loss_data_minibatch.append(np.mean(torch.stack(loss_freqs)).detach().cpu().numpy())
        #print(type(loss_freqs))
        
        loss_freqs = [tensor.detach().cpu().numpy() for tensor in loss_freqs]
        
        loss_data_minibatch.append(np.mean(loss_freqs))


    return np.mean(loss_data_minibatch), mp, ms, mrho, model