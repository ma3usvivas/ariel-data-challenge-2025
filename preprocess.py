import os
import pandas as pd
import numpy as np
import scipy

ROOT = os.path.join(os.getcwd(),'ariel-data-challenge-2025')
VERSION = 'v1'
MODE = 'train'
adc_info = pd.read_csv(os.path.join(ROOT, f'{MODE}_adc_info.csv')).set_index('planet_id')
SIGMA_TRANSITIONS = 60
PLANET_NAMES = list(adc_info.index)
STARS = np.array(adc_info["star"]).astype("i4")
STAR_KEYS = np.unique(STARS)
output_dir = os.path.join(os.getcwd(),'tmp','data_light_raw')

def get_planet_data(planet_name, bin_size=1): 
    path_planet = "%s/%dAIRS-_signal%s.npy" % (output_dir, planet_name,VERSION)
    if os.path.isfile(path_planet):
        data = np.load(path_planet)
    else:
        data = load_calibrated_data(planet_name, f_signal_pct = 50, a_signal_pct = 50)
        np.save(path_planet, data)
    
    if bin_size > 1:
        bin_mean, bin_var = binning(data, bin_size)
        return bin_mean
    else:
        return data

def dgauss(sig):
    xs = np.arange(-3.*sig, 3.*sig+1)
    den = 2.*sig*sig
    ys = np.exp(-np.square(xs)/den)
    dys = -2*xs/den*ys
    return dys

def d2gauss(sig):
    xs = np.arange(-3.*sig, 3.*sig+1)
    den = 2.*sig*sig
    ys = np.exp(-np.square(xs)/den)
    d2ys = np.square(2/den)*ys*(xs-sig)*(xs+sig)
    return d2ys

def clip_outliers(data, sigma, n = 15):
    """ sliding in-place removal of outliers"""

    n = min(n, data.size)
    n -= n%2
    off = n//2

    cumsum = np.cumsum(data)
    cumsum2 = np.cumsum(np.square(data))
      
    mean_win = np.empty(data.shape, "f8")
    std_win = np.empty(data.shape, "f8")        

    mean_win[off:-off] = (cumsum[n:]-cumsum[:-n])/n
    std_win[off:-off] = np.sqrt((cumsum2[n:]-cumsum2[:-n])/n - np.square(mean_win[off:-off]))

    mean_win[:off] = mean_win[off]
    mean_win[-off:] = mean_win[-off-1]    
    std_win[:off] = std_win[off]
    std_win[-off:] = std_win[-off-1]   
    
    data = np.clip(data, mean_win-std_win*sigma, mean_win+std_win*sigma)
    
    return data

def safe_savgol_filter(data, window_size, order=2):
    """ Data smoothing via savgol_filter """
    if data.size < window_size:
        return data
    return scipy.signal.savgol_filter(data, window_size, order)

def find_transit_edges(S, sigma):
    """ Find the centers of the transitions """

    Sc = np.convolve(S, dgauss(sigma), mode="valid")
    off = int((S.size-Sc.size)/2)
    mid = Sc.size//2
    
    transit_start = np.argmin(Sc[3:mid-3])+off+3
    transit_end = np.argmax(Sc[mid+3:-3])+off+mid+3

    return transit_start, transit_end

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def find_transit_slopes(S, transit_start, transit_end, sigma):
    """find the width of the transitions"""
    
    Sc2 = np.convolve(S, d2gauss(sigma), mode="valid")
    off = int((S.size-Sc2.size)/2)

    t1 = transit_start - off
    t2 = transit_end - off
    
    sz = 2*sigma
    t1a = np.argmin(Sc2[t1-sz:t1+1])+t1-sz+off
    t1b = np.argmax(Sc2[t1:t1+sz+1])+t1+off
    t2a = np.argmax(Sc2[t2-sz:t2+1])+t2-sz+off
    t2b = np.argmin(Sc2[t2:t2+sz+1])+t2+off

    return t1a, t1b, t2a, t2b

def star_statistics():
    """ SNR statistics by star and weights per planet """
    
    all_means = np.zeros((len(PLANET_NAMES), 357), "f8")
    all_vars  = np.zeros((len(PLANET_NAMES), 357), "f8")
    default_transition_width = 150
    
    for k,p in enumerate(PLANET_NAMES):    
        data = get_planet_data(p)
        S = np.mean(data, axis=0)
        S /= S.mean()
        start, end = find_transit_edges(S, SIGMA_TRANSITIONS)
        outs1 = data[:, :start-default_transition_width]
        outs2 = data[:, end+default_transition_width+1:]
        all_means[k, :] = (outs1.mean(axis=1) + outs2.mean(axis=1)) / 2.
        all_vars [k, :] = (outs1.var (axis=1) + outs2.var (axis=1)) / 2.
    
    ideal_weights = all_means / all_vars # at same noise level (divide by std), weight by snr
    snr = all_means / np.sqrt(all_vars)
        
    star_snrs = {}
    for star in STAR_KEYS:
        star_snrs[star] = snr[STARS==star].mean(axis=0)
        
    return ideal_weights, star_snrs

PLANETS_WEIGHTS, STARS_SNRS = star_statistics()