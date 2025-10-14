# Visualizations of the diffusion process

import os
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import torch
import rt60 as rt60
from ir_preprocess import re_envelope

from matplotlib.animation import FuncAnimation, PillowWriter
from tqdm import trange

from roomfuser.params import params
from roomfuser.dataset import RirDataset, FastRirDataset, RandomSinusoidDataset, FastRirDeEnvDataset, RandomRirDataset
from roomfuser.models import load_model
from roomfuser.inference import predict_batch

from roomfuser.utils import MinMaxScaler
import auraloss


def plot_diffusion(steps: np.array, target: np.array = None, envelopes=None, sr: int = 16000, rt60=None,
                   low_ord_input: np.array = None):
    """Plot the diffusion process.

    Args:
        steps (np.array): A numpy array of shape (n_steps, n_rir).

    """

    # Repeat the last step the same number of times as the first steps
    # This is to make the animation stop at the last step

    n_steps = steps.shape[0]

    last_step = steps[-1:]
    last_step = np.repeat(last_step, steps.shape[0], axis=0)
    steps = np.concatenate([steps, last_step], axis=0)

    fig, ax = plt.subplots()
    ax.set_xlim(0, steps.shape[1])
    ax.set_ylim(-1, 1)
    ax.set_xlabel("Tap number")
    ax.set_ylabel("Value")
    ax.set_title("Backward diffusion Process")

    line, = ax.plot([], [], lw=2)
    label = None
    if isinstance(rt60, torch.Tensor):
        rt60 = rt60.item()
    if rt60:
        label = "RT60={:.2f}".format(rt60)

    if target is not None:
        mse = np.mean((target - steps[-1])**2)
        labeled = f"Target (MSE={mse:.0E}, {label})"
        #labeled = "Target (MSE=%.0E) " %mse
        ax.plot(target, label=labeled, alpha=0.2, color="r")

    # Plot the envelope of the RIR based on the RT60
    if envelopes is not None:
        ax.plot(envelopes.numpy())

    if low_ord_input is not None:
        ax.plot(low_ord_input, label="Low order input", alpha=0.2, linestyle="--", color="g")

    if target is not None or label is not None:
        ax.legend(loc='upper right')

    def init():
        line.set_data([], [])
        return (line,)

    def animate(i):
        x = np.arange(steps.shape[1])
        y = steps[i]
        line.set_data(x, y)
        n_step = min(i, n_steps)
        ax.set_title(f"Backward diffusion Process (Step {n_step})")
        return (line,)

    anim = FuncAnimation(
        fig,
        animate,
        init_func=init,
        frames=steps.shape[0],
        interval=10,
        blit=True,
    )

    plt.close()
    return anim

def generate_rt60_histogram(rir_dataset):
    """
    Generate a histogram of RT60 values from the entire dataset.

    Args:
        rir_dataset: The dataset object containing RIRs and labels
    """
    print("Collecting RT60 statistics...")
    rt60_values = []

    # Loop through the entire dataset to collect RT60 values
    for i in trange(len(rir_dataset)):
        d = rir_dataset[i]
        labels = d["labels"]

        # Extract RT60 value based on dataset structure
        rt60 = None
        if "rt60" in labels:
            rt60 = labels["rt60"]
        elif "estimated_rt60" in labels:
            rt60 = labels["estimated_rt60"]
        elif "class" in labels and isinstance(labels["class"], (float, int)):
            rt60 = labels["class"]

        if rt60 is not None:
            if isinstance(rt60, torch.Tensor):
                rt60 = rt60.item()
            rt60_values.append(rt60)

    # Create histograms directory
    histograms_dir = "logs/histograms"
    os.makedirs(histograms_dir, exist_ok=True)

    # Plot and save histogram for RT60
    if rt60_values:
        plt.figure(figsize=(10, 6))
        plt.hist(rt60_values, bins=30, alpha=0.7, color='green')
        plt.xlabel('RT60 (seconds)')
        plt.ylabel('Count')
        plt.title('Distribution of RT60 Values')
        plt.grid(True, alpha=0.3)

        # Add statistics to the plot
        mean_rt60 = np.mean(rt60_values)
        median_rt60 = np.median(rt60_values)
        min_rt60 = np.min(rt60_values)
        max_rt60 = np.max(rt60_values)

        stats_text = f"Mean: {mean_rt60:.3f} s\nMedian: {median_rt60:.3f} s\n"
        stats_text += f"Min: {min_rt60:.3f} s\nMax: {max_rt60:.3f} s"

        plt.annotate(stats_text, xy=(0.95, 0.95), xycoords='axes fraction',
                     horizontalalignment='right', verticalalignment='top',
                     bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))

        plt.savefig(f"{histograms_dir}/rt60_histogram.png")
        plt.close()

        print(f"RT60 histogram saved to {histograms_dir}")
        print(f"RT60 statistics: Mean={mean_rt60:.3f}, Median={median_rt60:.3f}, "
              f"Min={min_rt60:.3f}, Max={max_rt60:.3f}")
    else:
        print("No RT60 data found in the dataset")

    return rt60_values

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
    else:
        rir_dataset = RandomRirDataset(
            n_rir=params.rir_len,
            n_samples_per_epoch= params.n_samples_per_epoch
        )
    
    #rt60_values = generate_rt60_histogram(rir_dataset)

    scaler_path = params.roomfuser_scaler_path
    #scaler = MinMaxScaler(scaler_path)
    scaler = None
    #scaler = rir_dataset.scaler

    model = load_model(params)
    
    model.load_state_dict(params.model_path)
    model.device = torch.device("cpu")
    #model.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    animations_dir = "logs/animations"
    os.makedirs(animations_dir, exist_ok=True)

    noise_prior = model.noise_scheduler.noise_prior
    for i in trange(params.n_viz_samples):
        #j = i
        j = np.random.randint(len(rir_dataset))
        d = rir_dataset[j]
        target_audio = d["rir"]
        conditioner = d["conditioner"]
        target_labels = d["labels"]
        
        # print("Labels: ", target_labels)
        prior_mean = noise_prior.get_mean([target_labels], target_audio.unsqueeze(0))[0]
        # Generate audio
        audio, sr = predict_batch(model, conditioner=conditioner.unsqueeze(0),
                              batch_size=1, return_steps=True, labels=[target_labels],
                              scaler=scaler, frequency_response=params.frequency_response)
        audio = audio[0].numpy()
        if scaler is not None:
            target_audio = scaler.descale(target_audio)
        if params.frequency_response:
            target_audio = torch.complex(target_audio[0], target_audio[1])
            target_audio = torch.fft.irfft(target_audio)
        
       
        #target_audio /= torch.max(torch.abs(target_audio))
        target_audio = target_audio.numpy()
        conditioner = conditioner.numpy()

        # Re-envelope both target and predicted audio if using fast_rir_de_env dataset
        if params.dataset_name == "fast_rir_de_env" and "processing_params" in d:
            # Get the de-envelope processing parameters
            processing_params = d["processing_params"]
            slope = processing_params["slope"]
            intercept = processing_params["intercept"] 
            start = processing_params["start"]
            sample_rate = processing_params["sample_rate"]
            target_audio = d["original_rir"] 
            target_audio = target_audio.cpu().numpy() if isinstance(target_audio, torch.Tensor) else target_audio

            intercept = 1
            print("t60_est: ", (60/(slope*sample_rate)))
            t60 = 5*(conditioner[9]+1)
            slope = (-60/t60)/sample_rate
            
            # Re-envelope both signals (both should already be numpy arrays)
            audio_reenv = np.zeros_like(audio)
            for step_idx in range(audio.shape[0]):
                audio_reenv[step_idx] = re_envelope(audio[step_idx], sample_rate, slope, intercept, start, len(audio[step_idx]))
            audio = audio_reenv

        # Plot diffusion process
        anim = plot_diffusion(audio, target_audio,
                              rt60=target_labels["rt60"],
                              low_ord_input=None)
        anim.save(f"{animations_dir}/diffusion_{i}.gif", writer=PillowWriter(fps=10))

        # Save audio
        sf.write(f"{animations_dir}/audio_{i}.wav", audio[-1], sr)
        sf.write(f"{animations_dir}/ref_audio_{i}.wav", target_audio, sr)

if __name__ == "__main__":
    generate_random_rir()
