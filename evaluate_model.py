# Visualizations of the diffusion process

import os
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import torch
import auraloss
import rt60

from matplotlib.animation import FuncAnimation, PillowWriter
from tqdm import trange

from roomfuser.params import params
from roomfuser.dataset import RirDataset, FastRirDataset, RandomSinusoidDataset, RealRirDataset
from roomfuser.models import load_model
from roomfuser.inference import predict_batch

from roomfuser.utils import MinMaxScaler
from ir_preprocess import re_envelope

def calc_xcorr(sig1, sig2, fs):
    # ERs only
    start1 = 0
    start2 = 0
    time = int(0.05*fs)
    er1 = sig1[start1:start1+time]
    er2 = sig2[start2:start2+time]

    norm1 = np.linalg.norm(er1)
    er1_norm = er1 / norm1
    norm2 = np.linalg.norm(er2)
    er2_norm = er2 / norm2
    xcorr = np.correlate(er1_norm, er2_norm, mode = 'full')
    return np.max(xcorr)

def calc_clarity(sig, x, fs):
    time = int((x/1000)*fs)
    er = sig[0:time]
    lr = sig[time:]
    energy_er = np.sum(np.pow(er,2))
    energy_lr = np.sum(np.pow(lr,2))
    return 10*np.log10(energy_er/energy_lr)

def generate_random_rir():
    if params.dataset_name == "fast_rir":
        rir_dataset = FastRirDataset(
            params.fast_rir_dataset_path,
            n_rir=params.rir_len,
            trim_direct_path=params.trim_direct_path,
            scaler_path=params.fast_rir_scaler_path,
            frequency_response=params.frequency_response,
        )
    elif params.dataset_name == "fast_rir_de_env":
        from roomfuser.dataset import FastRirDeEnvDataset
        rir_dataset = FastRirDeEnvDataset(
            params.fast_rir_dataset_path,
            n_rir=params.rir_len,
            trim_direct_path=params.trim_direct_path,
            scaler_path=params.fast_rir_scaler_path,
            frequency_response=params.frequency_response,
        )
    elif params.dataset_name == "roomfuser":
        rir_dataset = RirDataset(
            params.roomfuser_dataset_path,
            n_rir=params.rir_len,
            trim_direct_path=params.trim_direct_path,
            scaler_path=params.roomfuser_scaler_path,
            frequency_response=params.frequency_response,
        )
    elif params.dataset_name == "sinusoid":
        rir_dataset = RandomSinusoidDataset(
            params.rir_len, params.n_samples_per_epoch
        )
    elif params.dataset_name == "real":
        scaler_path = params.roomfuser_scaler_path
        rir_dataset = RealRirDataset(
            params.roomfuser_dataset_path,
            label_path=params.roomfuser_label_path,
            n_rir=params.rir_len,
            trim_direct_path=params.trim_direct_path,
            scaler_path=scaler_path,
            frequency_response=params.frequency_response,
        )
    elif params.dataset_name == "artificial":
        scaler_path = params.roomfuser_scaler_path
        rir_dataset = ArtificialRirDataset(
            params.roomfuser_dataset_path,
            csv_file=params.roomfuser_label_path,
            n_rir=params.rir_len,
            trim_direct_path=params.trim_direct_path,
            scaler_path=scaler_path,
            frequency_response=params.frequency_response,
        )
    else:
        rir_dataset = RandomRirDataset(
            n_rir=params.rir_len,
            n_samples_per_epoch= params.n_samples_per_epoch
        )
    scaler_path = params.roomfuser_scaler_path

    if scaler_path != "":
        scaler = MinMaxScaler(scaler_path)
    else:
        scaler = None

    model = load_model(params)

    # Load checkpoint and use EMA weights for inference
    checkpoint = torch.load(params.model_path, map_location="cuda")
    
    # The EMA object stores its state differently, we need to extract the actual model weights
    if 'ema' in checkpoint and checkpoint['ema'] is not None:
        print("Loading EMA weights for inference...")
        # EMA stores weights under 'shadow' key
        if 'shadow' in checkpoint['ema']:
            # First load the raw model weights to get all parameters including batch norm running stats
            model.load_state_dict(checkpoint['model'])
            # Then overwrite trainable parameters with EMA weights
            model_state = model.state_dict()
            ema_shadow = checkpoint['ema']['shadow']
            for name, param in ema_shadow.items():
                if name in model_state:
                    model_state[name] = param
            model.load_state_dict(model_state)
        else:
            print("EMA structure doesn't match expected format, falling back to model weights")
            model.load_state_dict(checkpoint['model'])
    elif 'ema_state_dict' in checkpoint:
        print("Loading EMA weights for inference...")
        model.load_state_dict(checkpoint['ema_state_dict'])
    elif 'model' in checkpoint:
        print("Loading raw model weights for inference...")
        model.load_state_dict(checkpoint['model'])
    elif 'model_state_dict' in checkpoint:
        print("Loading raw model weights for inference...")
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        # Fallback for older checkpoints without structured format
        print("Loading legacy checkpoint format...")
        model.load_state_dict(checkpoint)

    model.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(model.device) 
    model.eval()

    noise_prior = model.noise_scheduler.noise_prior
    
    xcorr_avg = 0.0
    mse_loss = l1_loss = spect_loss = c50_loss = c80_loss = 0.0
    t60_loss_fast = t60_loss_overall = t60_loss_bands = 0.0
    mrstft = auraloss.freq.MultiResolutionSTFTLoss()
    mrstft.to(model.device)
    targ_diff = 0
    for j in trange(params.test_size):
        #i = j

        i = np.random.randint(len(rir_dataset))
        d = rir_dataset[i]

        target_audio = d["rir"]
        conditioner = d["conditioner"]
        target_labels = d["labels"]

        prior_mean = noise_prior.get_mean([target_labels], target_audio.unsqueeze(0))[0]
        # Generate audio
        audio_out, sr = predict_batch(model, conditioner=conditioner.unsqueeze(0),
                            batch_size=1, return_steps=True, labels=[target_labels],
                            scaler=scaler, frequency_response=params.frequency_response)
        
        # Re-envelope both target and predicted audio if using fast_rir_de_env dataset
        dim = audio_out[0][-1].shape[0]
        dim2 = target_audio.shape[0]
        temp = audio_out[0][-1].reshape(1, 1, dim)
        targ = target_audio.reshape(1, 1, dim)

        temp = temp.to(model.device)
        targ = targ.to(model.device)
        if dim > 512:
            with torch.no_grad():
                spect_loss += mrstft(temp, targ)
        
        audio = audio_out[0].cpu().numpy()
        audio = audio[-1]
        
        if scaler is not None:
            target_audio = scaler.descale(target_audio)
        if params.frequency_response:
            target_audio = torch.complex(target_audio[0], target_audio[1])
            target_audio = torch.fft.irfft(target_audio)

        # Re-envelope both target and predicted audio if using fast_rir_de_env dataset
        if params.dataset_name == "fast_rir_de_env" and "processing_params" in d:
            # Get the de-envelope processing parameters
            processing_params = d["processing_params"]
            slope = processing_params["slope"]
            intercept = processing_params["intercept"]
            start = processing_params["start"]
            sample_rate = processing_params["sample_rate"]
            target_audio2 = d["original_rir"]
            
            intercept2 = 1
            t60 = 5*(conditioner[9].cpu().numpy()+1)
            slope2 = (-60/t60)/sample_rate

            # Convert tensors to numpy for re_envelope function
            target_audio_np = target_audio.cpu().numpy() if isinstance(target_audio, torch.Tensor) else target_audio
            target_audio = target_audio2.cpu().numpy() if isinstance(target_audio2, torch.Tensor) else target_audio2
            audio_np = audio.cpu().numpy() if isinstance(audio, torch.Tensor) else audio

            # Re-envelope both signals
            # target_audio = re_envelope(target_audio_np, sample_rate, slope, intercept, start, len(target_audio_np))
            audio = re_envelope(audio_np, sample_rate, slope2, intercept2, start, len(audio_np))
            # targ_diff += np.sum(np.abs(target_audio-target_audio2))
            audio_full = audio
            target_audio_full = target_audio
            # Convert back to tensors
            #target_audio = torch.from_numpy(target_audio_reenv).float()
            #audio = torch.from_numpy(audio_reenv).float()

        else:
            target_audio = target_audio.cpu().numpy()
            target_audio_full = target_audio
            audio_full = audio

        if dim > 512:
            c50_targ = calc_clarity(target_audio_full, 50, sr)
            c50_est = calc_clarity(audio_full, 50, sr)

            c80_targ = calc_clarity(target_audio_full, 80, sr)
            c80_est = calc_clarity(audio_full, 80, sr)
        # Calculate and accumulate MSE Loss, L1 Loss, and T60 Losses
        with torch.no_grad():
            mse_loss += np.mean((target_audio - audio)**2)
            l1_loss += np.mean(np.abs((target_audio-audio)))
            if dim > 512:
                c50_loss += np.abs(c50_targ-c50_est)
                c80_loss += np.abs(c80_targ-c80_est)
            t60_loss_temp_fast = rt60.t60_parallel_fastRIR(0, target_audio_full, audio_full, sr)
            t60_loss_fast += t60_loss_temp_fast

            t60_loss_temp_overall = rt60.t60_parallel_overall(0, target_audio_full, audio_full, sr)
            t60_loss_overall += t60_loss_temp_overall

            t60_loss_temp_bands = rt60.t60_parallel_bands(0, target_audio_full, audio_full, sr)
            t60_loss_bands += t60_loss_temp_bands
            xcorr_avg += calc_xcorr(target_audio_full, audio_full, sr)

    # Print Losses
    t60_loss_fast /= params.test_size
    t60_loss_overall /= params.test_size
    t60_loss_bands /= params.test_size
    mse_loss /= params.test_size
    l1_loss /= params.test_size
    spect_loss /= params.test_size
    c50_loss /= params.test_size
    c80_loss /= params.test_size
    xcorr_avg /= params.test_size
    print(f"MSE Loss: {mse_loss:.4f}")
    print(f"T60 Loss FastRIR: {t60_loss_fast:.4f}")
    print(f"T60 Loss Overall: {t60_loss_overall:.4f}")
    print(f"T60 Loss Bands: ", t60_loss_bands)
    print(f"L1 Loss: {l1_loss:.4f}")
    print(f"Spect Loss: {spect_loss:.4f}")
    print(f"C50 Loss: {c50_loss:.4f}")
    print(f"C80 Loss: {c80_loss:.4f}")
    print(f"Xcorr Avg: {xcorr_avg:.4}")

if __name__ == "__main__":
    generate_random_rir()
