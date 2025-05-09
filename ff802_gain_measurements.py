# -*- coding: utf-8 -*-
"""
Gain settings and clip voltage of the Fireface 802 knobs
--------------------------------------------------------
The RME Fireface 802 channels 9-12 have an inbuilt gain knob 
with only 3 pre-defined gain values: +6, +30 and +60. The gain
values of the intermediate positions are not clear, and are definitely
not linear as experience has shown. 

This report will set out to finally measure and verify the gains of
the pre-defined positions and the intermediate positions too. 

IMPORTANT
---------
Remember that the Fireface 802 (FF802 from now) has different clip voltages
on channels 9-12 depending on the 'Lo/Hi' gain setting chosen digitally and 
whether you use the instrument or XLR inputs!

The clip voltage @'Lo' gain setting (default) with the instrument is 27 dBu
and with the XLR is 16 dBu.


Notes
-----
I only later realised after looking into the Fireface 802 docs that the 
instrument line can't handle signals below -33 dBu (~17 mVrms).
And I've been using 

Created on Mon May  5 14:55:47 2025

@author: theja
"""
import glob
import matplotlib.pyplot as plt
import natsort
import numpy as np
import pandas as pd
import os 
import soundfile as sf
import scipy.ndimage as ndi
import scipy.signal as signal 
import tqdm

dB = lambda X: 20*np.log10(X)
rms = lambda X: np.sqrt(np.mean(X**2))
dBu_ref = 0.775 #  0 dBu  is this value in Vrms 


def vpp2rms(vpp_value):
    return vpp_value/(2*np.sqrt(2))

def extract_tones(input_audio, samplerate, min_durn=0.5,
                  min_env_level=1e-2, avg_windowsize=0.05):
    '''
    Calculates the absolute Hilbert envelope and then does 
    segmentation. Works best when you have a simple tone with a 
    constant level.
    
    This really works well only on very clean data...
    
    Parameters
    ----------
    input_audio : np.array
        Audio to be processed
    samplerate : float>0
        Sample rate in Hz
    min_durn : float>0, optional 
        MInimum duration a segment must be to be
        considered valid. Defaults to 0.5 s.
    min_env_level : float>0, optional 
        Minimum level of the absolute Hilbert envelope. 
        Defaults to 0.01. 
    avg_windowsize : float>0, optional 
        The mean averaging window size in seconds. 
        Defaults to  0.05 s
    
    Returns
    -------
    valid_segments: list with slices
        Indices of the valid segmented parts
    valid_audio_clips : list with np.arrays
        List with the segmented audio parts. 
    
    '''
    
    envelope = signal.hilbert(input_audio)
    env_abs = abs(envelope).flatten()
    # smooth out the envelope
    avg_window = np.ones(int(fs*avg_windowsize))/int(fs*avg_windowsize)
    avg_window = avg_window.flatten()
    conv_envelope = signal.convolve(env_abs, avg_window)
    
    segments_label, num = ndi.label(conv_envelope>min_env_level)
    segments = ndi.find_objects(segments_label)

    valid_segments = []
    valid_audio_clips = []
    for each in segments:
        each_seg = each[0]
        durn = (each_seg.stop - each_seg.start)/fs
        if durn >= min_durn:
            valid_segments.append(each_seg)
            valid_audio_clips.append(valid_audio_bp[each_seg.start:each_seg.stop])
    return valid_segments, valid_audio_clips

def get_vpp_from_filename(filename):
    return float(filename.split('_')[4][:-4])
def get_channelnum_from_filename(filename):
    return int(filename.split('_')[3][2:])
def get_gainpositions_from_filename(filename):
    parsed_output = filename.split('_')[5][3:]
    startend_positions = parsed_output.split('-')
    if len(startend_positions)>1:
        return np.arange(int(startend_positions[0]),
                     int(startend_positions[-1])+1)
    else:
        return np.array([int(startend_positions[0])])

#%%
# manufacturer specified clip Vrms
instrumentline_clip_level = 27 # dBu
clip_vrms_instrumentjack = dBu_ref*10**(instrumentline_clip_level/20) 
#%%


audio_files = natsort.natsorted(glob.glob('fireface_802_gainknob/2025-05-08/*.wav'))
v_pp_files = [float(each.split('_')[4][:-4]) for each in audio_files] 
chnum_files = [each.split('_')[3][2:] for each in audio_files] 

fs = sf.info(audio_files[-1]).samplerate
b,a = signal.butter(2, np.array([15e3,25e3])/(fs*0.5), 'bandpass')

#%%
import os 

all_measurements = []
for each in tqdm.tqdm(audio_files):
    each_rec, fs = sf.read(each)
    if sf.info(each).channels==2:
        rms_channels = [rms(each_rec[:,i]) for i in range(2)]
        valid_channel = rms_channels==np.max(rms_channels)
        valid_audio = each_rec[:,valid_channel]
    else:
        valid_audio = each_rec.copy()
    valid_audio_bp = signal.lfilter(b,a,valid_audio)
    # segment out each of the gain position recordings
    gain_positions = get_gainpositions_from_filename(each)
    input_vpp = get_vpp_from_filename(each)
    channelnum = get_channelnum_from_filename(each)
    seg_inds, audio_clips = extract_tones(valid_audio_bp, fs, min_durn=0.25,
                                          min_env_level=1.25e-4)
    df_columns = ['recording', 'channel_num', 'input_Vpp','gain_position',
                  'rms', 'dBrms','measured_vrms']
    audiofilename = os.path.split(each)[-1][:-4]

    if not os.path.exists('plots'):
        os.mkdir('plots')
    fig, axs = plt.subplots(len(audio_clips))
    for ax_num, current_ax in enumerate(axs):
        t = np.linspace(0, audio_clips[ax_num].size/fs,
                        audio_clips[ax_num].size)
        current_ax.plot(t, audio_clips[ax_num])
    
    plt.savefig(os.path.join('plots',f'{audiofilename}.png'))

    
    measurements = pd.DataFrame(data=[],
                                index=range(len(gain_positions)), 
                                columns=df_columns)
    if len(gain_positions) != len(audio_clips):
        raise ValueError(f'Poor segmentation. Expected {len(gain_positions)} segments, but got {len(audio_clips)} ')
    measurements['recording'] = os.path.split(each)[-1]
    measurements['input_Vpp'] = input_vpp*1e-3
    for i,clip in enumerate(audio_clips):
        rms_value = rms(clip)
        measurements.loc[i,'rms'] = rms_value
        measurements.loc[i,'gain_position'] = gain_positions[i]
        measurements.loc[i, 'dBrms'] = 20*np.log10(rms_value)
        measurements.loc[i,'channel_num'] = channelnum
    all_measurements.append(measurements)
        
        
#%%
all_measures_df = pd.concat(all_measurements).reset_index(drop=True)
# compute the actual Vrms measured using knowledge of the 
# clip voltage. 
max_rms = np.sqrt(2)/2
all_measures_df['measured_rel_maxrms'] = all_measures_df['rms']/max_rms
all_measures_df['input_vrms'] = vpp2rms(all_measures_df['input_Vpp'])
all_measures_df['approx_gain'] = (clip_vrms_instrumentjack/all_measures_df['input_vrms'])*all_measures_df['measured_rel_maxrms']
all_measures_df['approx_dBgain'] = 20*np.log10(np.array(all_measures_df['approx_gain'],
                                                        dtype=np.float64))

cols = ['measured_rel_maxrms',
        'input_vrms',
        'approx_gain', 
        'approx_dBgain']
all_measures_df[cols] = pd.to_numeric(all_measures_df[cols].stack(), errors='coerce').unstack()

all_measures_df = all_measures_df[all_measures_df['input_vrms']>=0.17]

all_measures_df.to_csv('all_rms_measurements.csv')


#%%

import scipy
from scipy import optimize

bygainpos = all_measures_df.groupby('gain_position')
position_num = 0
pos8_bychannel = bygainpos.get_group(position_num).groupby('channel_num')

pos8_channel9 = pos8_bychannel.get_group(9).loc[:,['measured_rel_maxrms',
                                                   'gain_position', 'input_vrms',
                                                   'input_Vpp']]

# make the matrix to solve
A = np.tile([1,-1], pos8_channel9.shape[0]).reshape(-1,2)
right_side = dB(pos8_channel9['measured_rel_maxrms']) - dB(pos8_channel9['input_vrms'])
B = np.array(right_side).reshape(-1,1)

vinrms = np.array(pos8_channel9['input_vrms'])
obs_normrms = pos8_channel9['measured_rel_maxrms']

def calculate_deviation(x0):
    dbgain, vclip_rms = x0
    obs_dbnormrms = dB(obs_normrms)
    exp_dbnormrms = dB(vinrms) + dbgain - dB(vclip_rms)
    #total_residual = np.sum(10**((obs_dbnormrms - exp_dbnormrms)/20))
    total_residual = np.median(obs_dbnormrms - exp_dbnormrms)
    return 10**(total_residual/20)

gain_bounds = (3,65)
clipvrms_bounds = (10, 20)
exp_bounds = [gain_bounds, clipvrms_bounds]

expected = 60 - dB(clip_vrms_instrumentjack)


initial_guess = np.array([55, clip_vrms_instrumentjack])
estimated = optimize.minimize(calculate_deviation, 
                                   initial_guess, method='Nelder-Mead',
                                   bounds=exp_bounds)

print(estimated.x)




