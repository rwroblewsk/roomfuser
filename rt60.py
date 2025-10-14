"""
Code adapted directly from FAST-RIR: https://github.com/anton-jeran/FAST-RIR
"""

import numpy as np
import math
from scipy.io import wavfile
from scipy import stats

from acoustics.utils import _is_1d
from acoustics.signal import bandpass
from acoustics.bands import (_check_band_type, octave_low, octave_high, third_low, third_high)

import soundfile as sf
from multiprocessing import Pool

def t60_impulse_fastRIR(raw_signal,fs,isReal):  # pylint: disable=too-many-locals
    """
    Reverberation time from a WAV impulse response.
    :param file_name: name of the WAV file containing the impulse response.
    :param bands: Octave or third bands as NumPy array.
    :param rt: Reverberation time estimator. It accepts `'t30'`, `'t20'`, `'t10'` and `'edt'`.
    :returns: Reverberation time :math:`T_{60}`
    """
    if not isReal:
        raw_signal = raw_signal[0:4096]
    bands =np.array([62.5 ,125, 250, 500, 1000, 2000])

    if np.max(raw_signal)==0 and np.min(raw_signal)==0:
        print('came 1')
        return .5

    # fs, raw_signal = wavfile.read(file_name)
    band_type = _check_band_type(bands)

    # if band_type == 'octave':
    low = octave_low(bands[0], bands[-1])
    high = octave_high(bands[0], bands[-1])
    # elif band_type == 'third':
    #     low = third_low(bands[0], bands[-1])
    #     high = third_high(bands[0], bands[-1])


    init = -0.0
    end = -60.0
    factor = 1.0
    bands =bands[3:5]
    low = low[3:5]
    high = high[3:5]

    t60 = np.zeros(bands.size)
    # bands = np.array([0])

    for band in range(bands.size):
        # Filtering signal
        # filtered_signal = raw_signal
        filtered_signal = bandpass(raw_signal, low[band], high[band], fs, order=8)
        abs_signal = np.abs(filtered_signal) / np.max(np.abs(filtered_signal))

        # Schroeder integration
        sch = np.cumsum(abs_signal[::-1]**2)[::-1]
        sch_db = 10.0 * np.log10((sch / np.max(sch)) + 1e-10)
        if math.isnan(sch_db[1]):
            print('came 2')
            return .5

        # Linear regression
        sch_init = sch_db[np.abs(sch_db - init).argmin()]
        sch_end = sch_db[np.abs(sch_db - end).argmin()]
        init_sample = np.where(sch_db == sch_init)[0][0]
        end_sample = np.where(sch_db == sch_end)[0][0]
        x = np.arange(init_sample, end_sample + 1) / fs
        y = sch_db[init_sample:end_sample + 1]
        slope, intercept = stats.linregress(x, y)[0:2]

        # Reverberation time (T30, T20, T10 or EDT)
        db_regress_init = (init - intercept) / slope
        db_regress_end = (end - intercept) / slope
        t60[band] = factor * (db_regress_end - db_regress_init)
    
    mean_t60 =(t60[1]+t60[0])/2
    if math.isnan(mean_t60):
        print('came 3')
        return .5
    return mean_t60

def t60_impulse_overall(raw_signal,fs,isReal):  # pylint: disable=too-many-locals
    """
    Reverberation time from a WAV impulse response.
    :param file_name: name of the WAV file containing the impulse response.
    :param bands: Octave or third bands as NumPy array.
    :param rt: Reverberation time estimator. It accepts `'t30'`, `'t20'`, `'t10'` and `'edt'`.
    :returns: Reverberation time :math:`T_{60}`
    """

    if not isReal:
        raw_signal = raw_signal[0:4096]
    if np.max(raw_signal)==0 and np.min(raw_signal)==0:
        print('came 1')
        return .5

    init = -0.0
    end = -60.0
    factor = 1.0

    filtered_signal = raw_signal
    abs_signal = np.abs(filtered_signal) / np.max(np.abs(filtered_signal))

    # Schroeder integration
    sch = np.cumsum(abs_signal[::-1]**2)[::-1]
    sch_db = 10.0 * np.log10((sch / np.max(sch)) + 1e-10)
    if math.isnan(sch_db[1]):
        print('came 2')
        return .5

    # Linear regression
    sch_init = sch_db[np.abs(sch_db - init).argmin()]
    sch_end = sch_db[np.abs(sch_db - end).argmin()]
    init_sample = np.where(sch_db == sch_init)[0][0]
    end_sample = np.where(sch_db == sch_end)[0][0]
    x = np.arange(init_sample, end_sample + 1) / fs
    y = sch_db[init_sample:end_sample + 1]
    slope, intercept = stats.linregress(x, y)[0:2]

    # Reverberation time (T30, T20, T10 or EDT)
    db_regress_init = (init - intercept) / slope
    db_regress_end = (end - intercept) / slope
    mean_t60 = factor * (db_regress_end - db_regress_init)

    if math.isnan(mean_t60):
        print('came 3')
        return .5
    return mean_t60

def t60_impulse_bands(raw_signal,fs, isReal):  # pylint: disable=too-many-locals
    """
    Reverberation time from a WAV impulse response.
    :param file_name: name of the WAV file containing the impulse response.
    :param bands: Octave or third bands as NumPy array.
    :param rt: Reverberation time estimator. It accepts `'t30'`, `'t20'`, `'t10'` and `'edt'`.
    :returns: Reverberation time :math:`T_{60}`
    """
    if not isReal:
        raw_signal = raw_signal[0:4096]
    bands =np.array([62.5 ,125, 250, 500, 1000, 2000])

    if np.max(raw_signal)==0 and np.min(raw_signal)==0:
        print('came 1')
        return .5

    band_type = _check_band_type(bands)

    # if band_type == 'octave':
    low = octave_low(bands[0], bands[-1])
    high = octave_high(bands[0], bands[-1])
    # elif band_type == 'third':
    #     low = third_low(bands[0], bands[-1])
    #     high = third_high(bands[0], bands[-1])


    init = -0.0
    end = -60.0
    factor = 1.0

    t60 = np.zeros(bands.size)

    for band in range(bands.size):
        # Filtering signal
        filtered_signal = bandpass(raw_signal, low[band], high[band], fs, order=8)
        abs_signal = np.abs(filtered_signal) / np.max(np.abs(filtered_signal))

        # Schroeder integration
        sch = np.cumsum(abs_signal[::-1]**2)[::-1]
        sch_db = 10.0 * np.log10((sch / np.max(sch)) + 1e-10)
        if math.isnan(sch_db[1]):
            print('came 2')
            return .5

        # Linear regression
        sch_init = sch_db[np.abs(sch_db - init).argmin()]
        sch_end = sch_db[np.abs(sch_db - end).argmin()]
        init_sample = np.where(sch_db == sch_init)[0][0]
        end_sample = np.where(sch_db == sch_end)[0][0]
        x = np.arange(init_sample, end_sample + 1) / fs
        y = sch_db[init_sample:end_sample + 1]
        slope, intercept = stats.linregress(x, y)[0:2]

        # Reverberation time (T30, T20, T10 or EDT)
        db_regress_init = (init - intercept) / slope
        db_regress_end = (end - intercept) / slope
        t60[band] = factor * (db_regress_end - db_regress_init)

        if math.isnan(t60[band]):
            print('came 3')
            return .5
    return t60

def t60_error(filename1,filename2):
    real_wave,fs = sf.read(filename1)
    # print(fs)
    fake_wave,fs = sf.read(filename2)
    # print(fs)

    channel = int(real_wave.size/len(real_wave))
    pool = Pool(processes=8)

    results =[]
    T60_error = 0
    for n in range(channel):
        results.append(t60_parallel(0,real_wave,fake_wave,fs))
        T60_error += t60_parallel(0,real_wave,fake_wave,fs)

    T60_error = T60_error/channel

    return str(T60_error)

def t60_impulse(raw_signal,fs,t60, isReal):  # pylint: disable=too-many-locals
    """
    Reverberation time from a WAV impulse response.
    :param file_name: name of the WAV file containing the impulse response.
    :param bands: Octave or third bands as NumPy array.
    :param rt: Reverberation time estimator. It accepts `'t30'`, `'t20'`, `'t10'` and `'edt'`.
    :returns: Reverberation time :math:`T_{60}`
    """
    if not isReal:
        raw_signal = raw_signal[0:4096]
    bands =np.array([62.5 ,125, 250, 500,1000, 2000])

    if np.max(raw_signal)==0 and np.min(raw_signal)==0:
        print('came 1')
        return .5
    
    band_type = _check_band_type(bands)

    # if band_type == 'octave':
    low = octave_low(bands[0], bands[-1])
    high = octave_high(bands[0], bands[-1])
    # elif band_type == 'third':
    #     low = third_low(bands[0], bands[-1])
    #     high = third_high(bands[0], bands[-1])

    
    init = -0.0
    end = -60.0
    factor = 1.0
    bands =bands[3:5]
    low = low[3:5]
    high = high[3:5]

    t60 = np.zeros(bands.size)

    for band in range(bands.size):
        # Filtering signal
        filtered_signal = bandpass(raw_signal, low[band], high[band], fs, order=8)
        abs_signal = np.abs(filtered_signal) / np.max(np.abs(filtered_signal))

        # Schroeder integration
        sch = np.cumsum(abs_signal[::-1]**2)[::-1]
        sch_db = 10.0 * np.log10(sch / np.max(sch))
        if math.isnan(sch_db[1]):
            print('came 2')
            return .5

        # Linear regression
        sch_init = sch_db[np.abs(sch_db - init).argmin()]
        sch_end = sch_db[np.abs(sch_db - end).argmin()]
        init_sample = np.where(sch_db == sch_init)[0][0]
        end_sample = np.where(sch_db == sch_end)[0][0]
        x = np.arange(init_sample, end_sample + 1) / fs
        y = sch_db[init_sample:end_sample + 1]
        slope, intercept = stats.linregress(x, y)[0:2]

        # Reverberation time (T30, T20, T10 or EDT)
        db_regress_init = (init - intercept) / slope
        db_regress_end = (end - intercept) / slope
        t60[band] = factor * (db_regress_end - db_regress_init)
    print("band1: ", t60[0], " band2: ", t60[1])
    mean_t60 =(t60[1]+t60[0])/2
    # print("meant60 is ", mean_t60)
    if math.isnan(mean_t60):
        print('came 3')
        return .5
    return mean_t60

def t60_error(filename1,filename2):
    real_wave,fs = sf.read(filename1)
    fake_wave,fs = sf.read(filename2)

    channel = int(real_wave.size/len(real_wave))
    pool = Pool(processes=8)
    
    results =[]
    for n in range(channel):
        results.append(pool.apply_async(t60_parallel, args=(n,real_wave,fake_wave,fs,)))
    
    T60_error =0
    for result in results:
        T60_error =  T60_error + result.get()

    T60_error = T60_error/channel
    
    pool.close()
    pool.join()

    return str(T60_error)

def t60_parallel(n,real_wave,fake_wave,fs,t60):
    real_wave_single = real_wave
    fake_wave_single   = fake_wave
    Real_T60_val = t60_impulse(real_wave_single,fs, t60,True)
    Fake_T60_val = t60_impulse(fake_wave_single,fs,t60,False)
    T60_diff = abs(Real_T60_val-Fake_T60_val)

    return T60_diff


def t60_parallel_fastRIR(n,real_wave,fake_wave,fs):
    real_wave_single = real_wave
    fake_wave_single   = fake_wave
    Real_T60_val = t60_impulse_fastRIR(real_wave_single,fs,True)
    Fake_T60_val = t60_impulse_fastRIR(fake_wave_single,fs, False)
    T60_diff = abs(Real_T60_val-Fake_T60_val)

    return T60_diff

def t60_parallel_overall(n,real_wave,fake_wave,fs):
    real_wave_single = real_wave
    fake_wave_single   = fake_wave
    Real_T60_val = t60_impulse_overall(real_wave_single,fs, True)
    Fake_T60_val = t60_impulse_overall(fake_wave_single,fs, False)
    T60_diff = abs(Real_T60_val-Fake_T60_val)

    return T60_diff

def t60_parallel_bands(n,real_wave,fake_wave,fs):
    real_wave_single = real_wave
    fake_wave_single   = fake_wave
    Real_T60_val = t60_impulse_bands(real_wave_single,fs, True)
    Fake_T60_val = t60_impulse_bands(fake_wave_single,fs, False)
    T60_diff = abs(Real_T60_val-Fake_T60_val)

    return T60_diff

if __name__ == '__main__':
    t60_impulse('/home/anton/Desktop/gamma101/data/evaluation_all/SF1/Hotel_SkalskyDvur_ConferenceRoom2-MicID01-SpkID01_20170906_S-09-RIR-IR_sweep_15s_45Hzto22kHz_FS16kHz.v00.wav')
  
