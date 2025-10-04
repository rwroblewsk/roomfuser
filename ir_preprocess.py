#!/usr/bin/env python
"""Impulse response processing and T60 reverberation time analysis."""

import math
import os
from typing import List, Tuple, Union

import numpy as np
import scipy.signal as signal
import librosa as lb
from scipy import stats
import matplotlib.pyplot as plt

from acoustics.signal import bandpass
from acoustics.bands import _check_band_type, octave_low, octave_high

def _calculate_t60_from_sch(sch_db: np.ndarray, fs: int, init: float = -0.0, end: float = -60.0) -> Tuple[float, float, float]:
    """Calculate T60 from Schroeder decay curve using linear regression."""
    if math.isnan(sch_db[1]):
        return 0.5, 0.0, 0.0

    sch_init = sch_db[np.abs(sch_db - init).argmin()]
    sch_end = sch_db[np.abs(sch_db - end).argmin()]
    init_sample = np.where(sch_db == sch_init)[0][0]
    end_sample = np.where(sch_db == sch_end)[0][0]
    x = np.arange(init_sample, end_sample + 1) / fs
    y = sch_db[init_sample:end_sample + 1]
    slope, intercept = stats.linregress(x, y)[0:2]

    db_regress_init = (init - intercept) / slope
    db_regress_end = (end - intercept) / slope
    t60 = db_regress_end - db_regress_init

    return t60, slope, intercept


def create_ir_example(dur: float, t60: float, fs: int) -> Tuple[np.ndarray, int]:
    """Create synthetic impulse response with exponential decay.

    Args:
        dur: Duration in seconds
        t60: Reverberation time in seconds
        fs: Sample rate in Hz

    Returns:
        Tuple of (impulse response array, sample rate)
    """
    num_samples = int(dur * fs)
    t60_samples = int(t60 * fs)
    ir = np.random.uniform(-0.9, 0.9, num_samples)
    decay_rate = 6.907755 / t60
    ir = ir * np.exp(-decay_rate * np.arange(num_samples) / fs)
    return ir, fs


def calc_t60(sig: np.ndarray, fs: int, start: int) -> Tuple[float, float, float]:
    """Calculate T60 using basic Schroeder integration."""
    new_sig = np.abs(sig[start:])
    sch = np.cumsum(new_sig[::-1]**2)[::-1]
    sch_db = 10.0 * np.log10((sch/np.max(sch)) + 1e-10)

    t60, slope, intercept = _calculate_t60_from_sch(sch_db, fs)
    return t60, slope/fs, intercept


def calc_t60_bands(sig: np.ndarray, fs: int, start: int) -> Union[float, List[float]]:
    """Calculate T60 using octave band filtering (500Hz-1kHz bands)."""
    raw_signal = sig[start:]
    bands = np.array([62.5, 125, 250, 500, 1000, 2000])

    if np.max(np.abs(raw_signal)) == 0:
        return 0.5

    band_type = _check_band_type(bands)
    low = octave_low(bands[0], bands[-1])
    high = octave_high(bands[0], bands[-1])

    # Focus on 500Hz-1kHz bands
    bands = bands[3:5]
    low = low[3:5]
    high = high[3:5]

    t60 = np.zeros(bands.size)

    for band in range(bands.size):
        filtered_signal = bandpass(raw_signal, low[band], high[band], fs, order=8)
        abs_signal = np.abs(filtered_signal) / np.max(np.abs(filtered_signal))

        sch = np.cumsum(abs_signal[::-1]**2)[::-1]
        sch_db = 10.0 * np.log10((sch / np.max(sch)) + 1e-6)

        t60_val, slope, intercept = _calculate_t60_from_sch(sch_db, fs)
        if t60_val == 0.5:  # Error case
            return 0.5
        t60[band] = t60_val

    mean_t60 = np.mean(t60)
    if math.isnan(mean_t60):
        return 0.5
    return [mean_t60, slope/fs, intercept]


def calc_abs_log_slope(sig: np.ndarray, fs: int, start: int, N: int) -> Union[float, List[float]]:
    """Analyze decay slopes with Schroeder integration and stop detection.

    Args:
        sig: Input signal
        fs: Sample rate in Hz
        start: Start sample
        N: End sample

    Returns:
        List of [t60, slope/fs, intercept, stopN] or error fallback (0.5)
    """
    raw_signal = sig[start:N]

    init = -0.0
    end = -50.0
    factor = 1.0
   
    filtered_signal = raw_signal 
    abs_signal = np.abs(filtered_signal) / np.max(np.abs(filtered_signal))

    # Schroeder integration
    sch = np.cumsum(abs_signal[::-1]**2)[::-1]
    sch_db = 10.0 * np.log10((sch / np.max(sch)) + 0.00000001)
    if math.isnan(sch_db[1]):
        return .5

    # Linear regression
    sch_init = sch_db[np.abs(sch_db - init).argmin()]
    sch_end = sch_db[np.abs(sch_db - end).argmin()]
    init_sample = np.where(sch_db == sch_init)[0][0]
    end_sample = np.where(sch_db == sch_end)[0][0]
    x = np.arange(init_sample, end_sample + 1) / fs
    y = sch_db[init_sample:end_sample + 1]

    slope, intercept = stats.linregress(x, y)[0:2]
    line = np.arange(N-start)*slope/fs+intercept
    # print("end: ", end)
    # print("start: ", start)
    # plt.plot(np.arange(8000)*slope/fs+intercept)
    # plt.plot(sch_db-line)
    stopN = np.argmax(np.abs(sch_db[300:]-line[300:])>5)+300

    # Reverberation time (T30, T20, T10 or EDT)
    db_regress_init = (init - intercept) / slope
    db_regress_end = (end - intercept) / slope
    t60 = factor * (db_regress_end - db_regress_init)
    mean_t60 = t60
    if math.isnan(mean_t60):
        return .5
    return [t60, slope/fs, intercept, stopN]


def normalize(x: np.ndarray) -> np.ndarray:
    """Normalize signal amplitude to ±1.

    Args:
        x: Input signal

    Returns:
        Normalized signal
    """
    max_x = np.max(np.abs(x))
    norm_fact = 1/max_x
    return norm_fact*x


def de_envelope(sig: np.ndarray, fs: int, N: int) -> Tuple[np.ndarray, float, float, int]:
    """Remove exponential decay envelope from impulse response.

    Args:
        sig: Input signal
        fs: Sample rate in Hz
        N: Number of samples to process

    Returns:
        Tuple of (de-enveloped signal within gain, slope, intercept, start, stopN)
    """
    start = 0
    [t60, slope, intercept, stopN] = calc_abs_log_slope(sig, fs, start, N)
    est = (10 ** ((((np.arange(stopN-start)-start)*slope)+intercept)/20))

    de_env = sig[start:stopN] * 1/est
    de_env = np.append(sig[0:start], de_env)
    min_env = np.min(de_env)
    max_env = np.max(de_env)
    de_env_norm = normalize(de_env)
    
    return de_env, slope, intercept, start, stopN
    # return de_env_norm, slope, intercept, start, stopN

def de_envelope_extended(sig: np.ndarray, fs: int, target_length: int):
    N = len(sig)
    de_env, slope, intercept, start, stopN = de_envelope(sig, fs, N)

    gated_length = stopN - start
    loop_length = int(0.1 * gated_length)  # Last 10%
    # Ensure loop_length is even for perfect 50% overlap
    if loop_length % 2 == 1:
        loop_length += 1
    num_loops = 3

    # Create extended signal to match original ir_norm length
    extended_de_env = de_env.copy()
    current_length = len(extended_de_env)

    # Loop blocks of the last 10% section to fill remaining length
    if current_length < target_length:
        remaining_samples = target_length - current_length
        # Add in randomized blocks from loop section as many times as needed
        full_loops = remaining_samples // (loop_length//2)
        partial_loop = remaining_samples % (loop_length//2)

        hann = np.hanning(loop_length)
        overlap_length = loop_length // 2  # 50% overlap

        for i in range(full_loops):
            rand_block_num = np.random.randint(0, num_loops)
            block = de_env[stopN-(rand_block_num+1)*loop_length:stopN-rand_block_num*loop_length]

            # Store original block for debugging
            # original_block = block.copy()

            # Subsequent blocks: overlap-add with Hann windowing
            prev_tail = extended_de_env[-overlap_length:] * hann[:overlap_length]
            block_head = block[:overlap_length] * hann[overlap_length:]
            extended_de_env[-overlap_length:] = prev_tail + block_head
            extended_de_env = np.append(extended_de_env, block[overlap_length:])

            # Store original block for debugging
            original_block = prev_tail.copy()

        if partial_loop > 0:
            rand_block_num = np.random.randint(0, num_loops)
            block = de_env[stopN-(rand_block_num+1)*loop_length:stopN-rand_block_num*loop_length]
            partial_block = block[:partial_loop*2]

            if partial_loop < overlap_length:
                # Partial block is shorter than overlap length
                # For odd partial_loop, floor the overlap to ensure we don't exceed available samples
                partial_overlap = partial_loop
                
                if partial_overlap > 0:
                    # Create appropriate Hann window for the actual overlap length
                    partial_hann = np.hanning(2 * partial_overlap)  # Full window for proper fade
                    prev_tail = extended_de_env[-partial_overlap:] * partial_hann[:partial_overlap]
                    block_head = partial_block[:partial_overlap] * partial_hann[partial_overlap:]
                    extended_de_env[-partial_overlap:] = prev_tail + block_head
                    extended_de_env = np.append(extended_de_env, partial_block[partial_overlap:])
                else:
                    # No overlap possible (partial_loop = 1), just append
                    extended_de_env = np.append(extended_de_env, partial_block)

            else:
                # Normal overlap-add with full overlap length
                prev_tail = extended_de_env[-overlap_length:] * hann[:overlap_length]
                block_head = partial_block[:overlap_length] * hann[overlap_length:]
                extended_de_env[-overlap_length:] = prev_tail + block_head

                # Append remaining samples (no windowing for final end)
                extended_de_env = np.append(extended_de_env, partial_block[overlap_length:])

    else:
        extended_de_env = extended_de_env[:target_length]
    #print(f"Extended de-enveloped signal length: {len(extended_de_env)}")
    #print(f"Target length: {target_length}")
    return extended_de_env, slope, intercept, start

def re_envelope(sig: np.ndarray, fs: int, slope: float, intercept: float, start: int, N: int) -> np.ndarray:
    """Reapply exponential decay envelope to signal.

    Args:
        sig: Input signal
        fs: Sample rate in Hz
        slope: Decay slope parameter
        intercept: Decay intercept parameter
        start: sample to begin enveloping
        N: Number of samples to process

    Returns:
        Re-enveloped signal
    """
    est = (10 ** ((((np.arange(N-start)-start)*slope)+intercept)/20))
    re_env = sig[start:N] * est
    re_env = np.append(sig[0:start], re_env)

    return re_env


def main():
    """Example usage of the IR processing functions."""
    # Create synthetic IR for testing
    dur = 1.0  # seconds
    t60_true = 0.5
    ir_fs = 16000
    ir, sr = create_ir_example(dur, t60_true, ir_fs)

    # Normalize peak amplitude to 1
    ir_norm = normalize(ir)

    # Calculate de-envelope and re-envelope
    N = 8000
    de_env, slope, intercept, start, stopN = de_envelope(ir, ir_fs, N)
    re_env = re_envelope(de_env, ir_fs, slope, intercept, start, stopN-start+1)

    print(f"De-envelope done. stopN: {stopN}")

    # Calculate re-envelope gain for overlay plotting
    start = int(0.01*ir_fs)
    re_envelope_gain = np.zeros(len(ir))
    est = (10 ** ((((np.arange(stopN-start)-start)*slope)+intercept)/20))
    re_envelope_gain[start:stopN] = 20*np.log10(est)

    # Plot input signal with re-envelope gain overlay
    time_axis = np.arange(len(ir)) / ir_fs
    plt.figure(figsize=(10, 6))
    plt.plot(time_axis, 20*np.log10(np.abs(ir)), 'b-', label='Input signal', linewidth=1)
    plt.plot(time_axis[start:stopN], re_envelope_gain[start:stopN], 'r--', label='Re-envelope gain vs time', linewidth=2)

    # Add vertical lines at re-envelope boundaries
    plt.axvline(x=time_axis[start], color='k', linestyle='-', linewidth=1, label='Re-envelope start')
    plt.axvline(x=time_axis[stopN], color='k', linestyle='-', linewidth=1, label='Re-envelope end')

    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude (dB)')
    plt.title('Input Signal with Fitted Exponential Decay Overlay')
    plt.ylim(-120, 10)
    plt.legend()
    plt.grid(True)
    plt.show()

    # Plot de-enveloped signal
    de_time_axis = np.arange(len(de_env)) / ir_fs
    plt.figure(figsize=(10, 6))
    plt.plot(de_time_axis, 20*np.log10(np.abs(de_env) + 1e-12), 'b-', label='De-enveloped signal', linewidth=1)
    plt.xlabel('Time (s)')
    plt.ylim(-120, 10)
    plt.legend()
    plt.grid(True)
    # Add vertical lines at re-envelope boundaries
    plt.axvline(x=de_time_axis[start], color='k', linestyle='-', linewidth=1, label='Re-envelope start')
    plt.axvline(x=de_time_axis[stopN-1], color='k', linestyle='-', linewidth=1, label='Re-envelope end')
    plt.show()

    # Gate de-enveloped signal to envelope boundaries
    de_env[0:start] = 0
    de_env[stopN:] = 0

    # For testing: replace de-enveloped signal with constant for clearer overlap-add visualization
    # de_env[start:stopN] = 0.1  # Constant value instead of white noise

    extended_de_env, slope, intercept, start = de_envelope_extended(ir, ir_fs, 11200)
    
    # Plot the extended de-enveloped signal
    plt.figure(figsize=(12, 6))
    extended_de_time_axis = np.arange(len(extended_de_env)) / ir_fs
    plt.plot(extended_de_time_axis, 20*np.log10(np.abs(extended_de_env) + 1e-12), 'g-', label='Extended De-enveloped Signal', linewidth=1)

    # Add vertical lines at re-envelope boundaries
    plt.axvline(x=extended_de_time_axis[start], color='k', linestyle='-', linewidth=1, label='Re-envelope start')
    if stopN < len(extended_de_time_axis):
        plt.axvline(x=extended_de_time_axis[stopN], color='k', linestyle='-', linewidth=1, label='Re-envelope end')
    else:
        plt.axvline(x=extended_de_time_axis[min(stopN, len(extended_de_time_axis)-1)], color='k', linestyle='-', linewidth=1, label='Re-envelope end')

    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude (dB)')
    plt.title('Extended De-enveloped Signal (Before Re-enveloping)')
    plt.ylim(-120, 10)
    plt.legend()
    plt.grid(True)
    plt.show()

    # Extend de-enveloped signal by looping last 10% until original length
    # gated_length = stopN - start
    # loop_length = int(0.1 * gated_length)  # Last 10%
    # # Ensure loop_length is even for perfect 50% overlap
    # if loop_length % 2 == 1:
    #     loop_length += 1
    # num_loops = 3
    # # print("Loop section length:", loop_length)
    # # print("Number of blocks:", num_blocks)

    # # Create extended signal to match original ir_norm length
    # extended_de_env = de_env.copy()
    # current_length = len(extended_de_env)
    # target_length = 11200 # Should match length of IR to be generated
    # # target_length = len(ir_norm)

    # # Loop blocks of the last 10% section to fill remaining length
    # if current_length < target_length:
    #     remaining_samples = target_length - current_length
    #     # Add in randomized blocks from loop section as many times as needed
    #     full_loops = remaining_samples // loop_length
    #     partial_loop = remaining_samples % loop_length

    #     hann = np.hanning(loop_length)
    #     overlap_length = loop_length // 2  # 50% overlap

    #     # Debug plots: create figure for windowing visualization
    #     debug_fig, debug_axes = plt.subplots(full_loops + (1 if partial_loop > 0 else 0), 1,
    #                                        figsize=(12, 3 * (full_loops + (1 if partial_loop > 0 else 0))))
    #     if full_loops + (1 if partial_loop > 0 else 0) == 1:
    #         debug_axes = [debug_axes]

    #     for i in range(full_loops):
    #         rand_block_num = np.random.randint(0, num_loops)
    #         block = de_env[stopN-(rand_block_num+1)*loop_length:stopN-rand_block_num*loop_length]

    #         # Store original block for debugging
    #         # original_block = block.copy()

    #         # Subsequent blocks: overlap-add with Hann windowing
    #         prev_tail = extended_de_env[-overlap_length:] * hann[:overlap_length]
    #         block_head = block[:overlap_length] * hann[overlap_length:]
    #         extended_de_env[-overlap_length:] = prev_tail + block_head
    #         extended_de_env = np.append(extended_de_env, block[overlap_length:])

    #         # Store original block for debugging
    #         original_block = prev_tail.copy()

    #         # Debug plot for this block
    #         ax = debug_axes[i]
    #         time_block = np.arange(len(block)) / ir_fs
    #         ax.plot(time_block[:overlap_length], original_block, 'b-', label='Original block', alpha=0.7)
    #         ax.plot(time_block[:overlap_length], block_head, 'r-', label='Windowed overlap', linewidth=2)
    #         # ax.plot(time_block[overlap_length:], prev_tail + block_head, 'g-', label='Combined', linewidth=2)
    #         ax.plot(time_block[overlap_length:], block[overlap_length:], 'g-', label='Unwindowed remainder', linewidth=2)
    #         ax.axvline(x=time_block[overlap_length-1], color='k', linestyle='--', alpha=0.5, label='Overlap boundary')
    #         ax.set_title(f'Block {i+1}: Windowing Contributions')
    #         ax.set_ylabel('Amplitude (dB)')
    #         # ax.set_ylim(-80, 10)
    #         ax.legend()
    #         ax.grid(True)

    #     if partial_loop > 0:
    #         rand_block_num = np.random.randint(0, num_loops)
    #         block = de_env[stopN-(rand_block_num+1)*loop_length:stopN-rand_block_num*loop_length]
    #         partial_block = block[:partial_loop]
    #         original_partial = partial_block.copy()

    #         if partial_loop < overlap_length:
    #             # Partial block is shorter than overlap length, create appropriate Hann window
    #             partial_hann = np.hanning(partial_loop)
    #             prev_tail = extended_de_env[-partial_loop//2:] * partial_hann[partial_loop//2:]
    #             block_head = partial_block[:partial_loop//2] * partial_hann[:partial_loop//2]
    #             extended_de_env[-partial_loop:] = prev_tail + block_head
    #             extended_de_env = np.append(extended_de_env, partial_block[partial_loop//2:])

    #         else:
    #             # Normal overlap-add with full overlap length
    #             prev_tail = extended_de_env[-overlap_length:] * hann[:overlap_length]
    #             block_head = partial_block[:overlap_length] * hann[overlap_length:]
    #             extended_de_env[-overlap_length:] = prev_tail + block_head

    #             # Append remaining samples (no windowing for final end)
    #             extended_de_env = np.append(extended_de_env, partial_block[overlap_length:])

    #     debug_axes[-1].set_xlabel('Time (s)')
    #     plt.tight_layout()
    #     plt.suptitle('Debug: Windowing Effects on Each Added Block', y=0.98)
    #     plt.show()

    # else:
    #     extended_de_env = extended_de_env[:target_length]

    # # Plot the extended de-enveloped signal
    # plt.figure(figsize=(12, 6))
    # extended_de_time_axis = np.arange(len(extended_de_env)) / ir_fs
    # plt.plot(extended_de_time_axis, 20*np.log10(np.abs(extended_de_env) + 1e-12), 'g-', label='Extended De-enveloped Signal', linewidth=1)

    # # Add vertical lines at re-envelope boundaries
    # plt.axvline(x=extended_de_time_axis[start], color='k', linestyle='-', linewidth=1, label='Re-envelope start')
    # if stopN < len(extended_de_time_axis):
    #     plt.axvline(x=extended_de_time_axis[stopN], color='k', linestyle='-', linewidth=1, label='Re-envelope end')
    # else:
    #     plt.axvline(x=extended_de_time_axis[min(stopN, len(extended_de_time_axis)-1)], color='k', linestyle='-', linewidth=1, label='Re-envelope end')

    # plt.xlabel('Time (s)')
    # plt.ylabel('Amplitude (dB)')
    # plt.title('Extended De-enveloped Signal (Before Re-enveloping)')
    # plt.ylim(-120, 10)
    # plt.legend()
    # plt.grid(True)
    # plt.show()



    # Apply re-envelope gain to the extended signal
    extended_re_env = re_envelope(extended_de_env, ir_fs, slope, intercept, start, len(extended_de_env))

    save_audio = False  # Set to True to save audio files for comparison
    if (save_audio):
        # Save audio files for comparison
        import scipy.io.wavfile as wavfile

        # Normalize both signals to prevent clipping
        ir_norm_audio = ir / np.max(np.abs(ir_norm)) * 0.95
        extended_norm_audio = extended_re_env / np.max(np.abs(extended_re_env)) * 0.95

        # Save original IR
        wavfile.write('original_ir.wav', ir_fs, ir_norm_audio.astype(np.float32))
        print(f"Saved original IR to 'original_ir.wav' ({len(ir_norm_audio)} samples, {len(ir_norm_audio)/ir_fs:.2f}s)")

        # Save extended IR
        wavfile.write('extended_ir.wav', ir_fs, extended_norm_audio.astype(np.float32))
        print(f"Saved extended IR to 'extended_ir.wav' ({len(extended_norm_audio)} samples, {len(extended_norm_audio)/ir_fs:.2f}s)")

        print(f"Extension ratio: {len(extended_norm_audio)/len(ir_norm_audio):.2f}x longer")

    # Plot the final extended and re-enveloped result below original IR
    # plt.figure(figsize=(10, 12))
    plt.figure(figsize=(12, 8))

    # Top subplot: Original IR
    plt.subplot(3, 1, 1)
    time_axis = np.arange(len(ir_norm)) / ir_fs
    plt.plot(time_axis, 20*np.log10(np.abs(ir_norm) + 1e-12), 'b-', label='Original IR', linewidth=1)
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude (dB)')
    plt.title('Original Impulse Response')
    plt.ylim(-120, 10)
    plt.legend()
    plt.grid(True)

    # Middle subplot: Gated de-enveloped signal
    plt.subplot(3, 1, 2)
    gated_time_axis = np.arange(len(de_env)) / ir_fs
    plt.plot(gated_time_axis, 20*np.log10(np.abs(de_env) + 1e-12), 'g-', label='Gated De-enveloped', linewidth=1)

    # Add vertical lines at re-envelope boundaries
    plt.axvline(x=gated_time_axis[start], color='k', linestyle='-', linewidth=1, label='Re-envelope start')
    if stopN < len(gated_time_axis):
        plt.axvline(x=gated_time_axis[stopN], color='k', linestyle='-', linewidth=1, label='Re-envelope end')
    else:
        plt.axvline(x=gated_time_axis[-1], color='k', linestyle='-', linewidth=1, label='Re-envelope end')

    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude (dB)')
    plt.title('Gated De-enveloped Signal')
    plt.xlim(0, (len(ir_norm)-1)/ir_fs)
    plt.ylim(-120, 10)
    plt.legend()
    plt.grid(True)

    # Bottom subplot: Extended and re-enveloped signal
    plt.subplot(3, 1, 3)
    extended_time_axis = np.arange(len(extended_re_env)) / ir_fs
    plt.plot(extended_time_axis, 20*np.log10(np.abs(extended_re_env) + 1e-12), 'm-', label='Extended & Re-enveloped', linewidth=1)

    # Add vertical lines at re-envelope boundaries
    plt.axvline(x=extended_time_axis[start], color='k', linestyle='-', linewidth=1, label='Re-envelope start')
    if stopN < len(extended_time_axis):
        plt.axvline(x=extended_time_axis[stopN], color='k', linestyle='-', linewidth=1, label='Re-envelope end')
    else:
        plt.axvline(x=extended_time_axis[min(stopN, len(extended_time_axis)-1)], color='k', linestyle='-', linewidth=1, label='Re-envelope end')

    # Add vertical lines for each half-frame of appended data
    # if current_length < target_length:  # Only if extension occurred
    #     original_end_sample = len(de_env)
    #     for i in range(full_loops):
    #         half_frame_position = original_end_sample + i * overlap_length
    #         if half_frame_position < len(extended_time_axis):
    #             plt.axvline(x=extended_time_axis[half_frame_position], color='r', linestyle=':', alpha=0.7, linewidth=1,
    #                        label='Half-frame start' if i == 0 else '')

    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude (dB)')
    plt.title('Extended De-enveloped Signal with Re-applied Envelope')
    plt.xlim(0, (len(ir_norm)-1)/ir_fs)
    plt.ylim(-120, 10)
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()

