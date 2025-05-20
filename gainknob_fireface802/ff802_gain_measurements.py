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
instrument line can handle at most -33 dBu (~17 mVrms) for the +60 dB gain, 
but also in general this seems to be the minimum limit from which point the 
signal is above the baseline noise. 

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

def vrms2pp(vrms):
    return vrms*2*np.sqrt(2)

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
    conv_envelope : np.array
        The smoothed Hilbert envelope of the audio 
    
    '''
    
    envelope = signal.hilbert(input_audio)
    env_abs = abs(envelope).flatten()
    # smooth out the envelope
    avg_window = np.ones(int(fs*avg_windowsize))/int(fs*avg_windowsize)
    avg_window = avg_window.flatten()
    conv_envelope = signal.convolve(env_abs, avg_window, 'same')
    
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
    
    return valid_segments, valid_audio_clips, conv_envelope

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

# The minimum signal voltage that the instrument lines can handles are
# -33 dBu 
# https://rme-audio.de/downloads/fface_802_e.pdf
minlevel_vrms = dBu_ref*10**(-33/20)
min_vpp_level = np.around(vrms2pp(minlevel_vrms),3)
print(f'Min Vpp the 802 instrument lines can handle are: {min_vpp_level} Vpp')

# let's set out threshold 2 dB above this
min_vpp_threshold = min_vpp_level*10**(2/20)



#%%


#audio_files = natsort.natsorted(glob.glob('fireface_802_gainknob/2025-05-08/*.wav'))
audio_files = natsort.natsorted(glob.glob('fireface_802_gainknob-data/2025-05-15/*.wav'))
v_pp_files = [float(each.split('_')[4][:-4]) for each in audio_files] 
chnum_files = [each.split('_')[3][2:] for each in audio_files] 

fs = sf.info(audio_files[-1]).samplerate
b,a = signal.butter(2, np.array([15e3,25e3])/(fs*0.5), 'bandpass')

#%%
import os 

all_measurements = []
for ii, each in tqdm.tqdm(enumerate(audio_files)):
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
    seg_inds, audio_clips, envelope = extract_tones(valid_audio_bp, fs, min_durn=0.25,
                                          min_env_level=0.5e-3)
    df_columns = ['recording', 'channel_num', 'input_Vpp','gain_position',
                  'rms', 'dBrms','measured_vrms']
    audiofilename = os.path.split(each)[-1][:-4]

    if not os.path.exists('plots'):
        os.mkdir('plots')
    fig, axs = plt.subplots(len(audio_clips)+1)
    for ax_num, current_ax in enumerate(axs):
        if ax_num <len(axs)-1:
            t = np.linspace(0, audio_clips[ax_num].size/fs,
                            audio_clips[ax_num].size)
            current_ax.plot(t, audio_clips[ax_num])
            
            current_ax.sharey(axs[-1])
            dbrms = np.around(dB(rms(audio_clips[ax_num])), 2)
            maxrms = np.sqrt(2)/2
            maxnorm_dbrms = dB(rms(audio_clips[ax_num])/maxrms)
            maxnorm_dbrms = np.around(maxnorm_dbrms, 2)
            current_ax.set_title(f'dbrms re max: {maxnorm_dbrms}, dbrms:{dbrms}')
        else:
            t = np.linspace(0, valid_audio_bp.size/fs, valid_audio_bp.size)
            current_ax.plot(t, abs(valid_audio_bp))
            current_ax.plot(t, envelope*2, 'r')
            
            
            
        
    plt.savefig(os.path.join('plots',f'file_{ii}_{audiofilename}.png'))
    plt.close()

    
    measurements = pd.DataFrame(data=[],
                                index=range(len(gain_positions)), 
                                columns=df_columns)
    if len(gain_positions) != len(audio_clips):
        raise ValueError(f'Poor segmentation: {each}. Expected {len(gain_positions)} segments, but got {len(audio_clips)} ')
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
# Clean up the measurements


all_measures_df = pd.concat(all_measurements).reset_index(drop=True)



# remove some faulty measurements
bad_files = ['ff802_ch09_100mVpp_pos0-4_gain.wav',
             'ff802_ch09_100mVpp_pos4-8_gain.wav',
             'ff802_ch11_1000mVpp_pos0-4_gain.wav']

bad_rows = all_measures_df['recording'].isin(bad_files)

all_measures_df = all_measures_df[np.invert(bad_rows)]

# remove the clipped measurements
all_measures_df = all_measures_df[all_measures_df['dBrms'] <= -3].reset_index(drop=True)

#%%

# compute the actual Vrms measured using knowledge of the 
# clip voltage. 
max_rms = np.sqrt(2)/2
all_measures_df['measured_rel_maxrms'] = all_measures_df['rms']/max_rms
all_measures_df['input_vrms'] = vpp2rms(all_measures_df['input_Vpp'])

cols = ['measured_rel_maxrms',
        'input_vrms',]
all_measures_df[cols] = pd.to_numeric(all_measures_df[cols].stack(), errors='coerce').unstack()

all_measures_df = all_measures_df[all_measures_df['input_Vpp']>=min_vpp_threshold]



all_measures_df.to_csv('all_rms_measurements.csv')

#%%
# Now calculate the relative gain w.r.t position 0. 
by_channel_vpp = all_measures_df.groupby(['channel_num', 'input_Vpp'])

fig, axs = plt.subplots(4,1, figsize=(5,12))

all_relgains = []

for rec_name, subdf in by_channel_vpp:
    ref_dBrms = np.float64(subdf.loc[subdf['gain_position']==0,'dBrms'])
    subdf['relgain_repos0'] = subdf['dBrms'] - ref_dBrms
    ax_index = int(subdf['channel_num'].unique()-9)
    print(rec_name)
    print(subdf['relgain_repos0'])
    plt.sca(axs[ax_index])
    plt.plot(subdf['gain_position'], subdf['relgain_repos0'], '-*')
    plt.text(2.5, 3, f'channel {ax_index+9} Fireface 802, inst. jack')
    all_relgains.append(subdf)
for each in range(4):
    plt.sca(axs[each])
    major_yticks = np.arange(0, 54, 6)
    minor_yticks = np.arange(0,55)
    axs[each].set_yticks(major_yticks)
    axs[each].set_yticks(minor_yticks, minor=True)
    axs[each].grid(which='both')
    plt.hlines([0,24], 0,8)
    
    # draw grid
    for loc in range(0, 54):
        axs[each].axhline(loc, alpha=0.2,
                          color='#b0b0b0', linestyle='-', linewidth=0.8)
        plt.grid()
    plt.tight_layout()

axs[1].set_ylabel('rel gain dB, re position 0')
axs[-1].set_xlabel('Gain knob position')
plt.sca(axs[0]);plt.title('channel-wise gain rel position 0')
plt.savefig('relative_gain_repos0.png')
#%%
# What's the gain of the 0th position however? 

onlypos0 = all_measures_df[all_measures_df['gain_position']==0]
onlypos0_bychannel = onlypos0.groupby('channel_num')


def calculate_deviation(x0):
    dbgain, vclip_rms = x0
    obs_dbnormrms = dB(obs_normrms)
    exp_dbnormrms = dB(vinrms) + dbgain - dB(vclip_rms)
    # total_residual = np.sum(10**(abs(obs_dbnormrms - exp_dbnormrms)/20))
    total_residual = np.sum(np.abs(obs_dbnormrms - exp_dbnormrms))
    return total_residual


gain_bounds = (4,8)
clipvrms_bounds = (15, 20)
exp_bounds = [gain_bounds, clipvrms_bounds]

import scipy
from scipy import optimize

initial_guess = np.array([9, clip_vrms_instrumentjack])

# compare results from both optimisation methods
all_dfs = []
for rec_name, subdf in onlypos0_bychannel:
    df = pd.DataFrame(index=range(2), columns=['estgain_dB', 
                                               'estclip_Vrms',
                                               'lstsq_method',
                                               'position', 'channel',
                                               'avg_Vcliprms',
                                               'avg_estgain_dB'])
    df['position'] = 0
    df['channel'] = rec_name
    for i,method in enumerate(['dogbox', 'trf']):
        obs_normrms = np.float64(subdf['rms']/(np.sqrt(2)/2))
        vinrms = np.float64(subdf['input_vrms'])
        estimated = optimize.least_squares(calculate_deviation, 
                                           initial_guess, 
                                           bounds=exp_bounds,
                                           method=method)
        df.loc[i,'lstsq_method'] = method
        df.loc[i,'estgain_dB'] = estimated.x[0]
        df.loc[i,'estclip_Vrms'] = estimated.x[1]
    df['avg_Vcliprms'] = np.mean(df['estclip_Vrms'])
    df['avg_estgain_dB'] = np.mean(df['estgain_dB'])
    all_dfs.append(df)
        
estimated_gains = pd.concat(all_dfs).reset_index(drop=True)
estimated_gains.to_csv('estimated_gain_pos0_vcliprms_fireface802.csv')

#%%
# Now let's average out all the gain readings from the other positions. 

df_allrelgains = pd.concat(all_relgains)
allrelgains_bych = df_allrelgains.groupby(['channel_num','gain_position'])
relgains = []
for (chnum, pos), subdf in allrelgains_bych:
    df = pd.DataFrame(index=range(1), columns=['channel_num', 'gain_position',
                                          'est_meangain_dB'])
    mean_gain = dB(np.mean(10**(subdf['relgain_repos0']/20)))
    print(pos, mean_gain)
    df['est_meangain_dB'] = mean_gain
    df['channel_num'] = chnum
    df['gain_position'] = pos
    relgains.append(df)
    
mean_relgains = pd.concat(relgains).reset_index(drop=True)

fig, ax = plt.subplots(1,1)

for chnum, subdf in mean_relgains.groupby('channel_num'):
    gainpos0 = estimated_gains.groupby('channel').get_group(chnum)['avg_estgain_dB'].unique()
    plt.plot(subdf.loc[:,'gain_position'],
             subdf.loc[:,'est_meangain_dB']+gainpos0,'-*', 
             label=f'Channel {chnum}')
ins = ax.inset_axes([0.6,0.1,0.5,0.5])
img = plt.imread('ff802_gain_knob_labelled.png')
ins.imshow(img);ins.set_xticks([]);ins.set_yticks([])

major_yticks = np.arange(0, 54, 3)
plt.sca(ax)
plt.legend()
minor_yticks = np.arange(0,55)
ax.set_yticks(major_yticks, major_yticks, fontsize=12)
ax.set_yticks(minor_yticks, minor=True)
ax.set_xticks(range(8))
ax.grid(which='both')
plt.hlines([6,30], 0,8, 'k')
plt.xlabel('Gain position', fontsize=14)
plt.ylabel('Estimated gain, dB', fontsize=14)
plt.title('Fireface 802- instrument jack channels 9-12 Sr. no.: 23746770 ')
plt.savefig('fireface_ch9-12_gain_estimates-srno-23746770.png')

mean_relgains.to_csv('fireface802-srno_23746770_gainpositions.csv')

