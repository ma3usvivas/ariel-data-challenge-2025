import numpy as np
from functools import partial
import pandas as pd
import itertools
import multiprocessing as mp
import os
import glob 
from astropy.stats import sigma_clip

from tqdm import tqdm

#ROOT = "/kaggle/input/ariel-data-challenge-2025/"
ROOT = os.path.join(os.getcwd(),'ariel-data-challenge-2025')
MODE = 'train'
VERSION = "v1"
ORIGIN = 'os'
CHUNKS_SIZE = 4
DO_MASK = True
DO_THE_NL_CORR = True
DO_DARK = True
DO_FLAT = True
MIN_WL = 39
MAX_WL = 321

if ORIGIN == 'kaggle':
    path_folder = '/kaggle/input/ariel-data-challenge-2025/' # path to the folder containing the data
    path_out = '/kaggle/tmp/data_light_raw/' # path to the folder to store the light data
    output_dir = '/kaggle/tmp/data_light_raw/' # path for the output directory

if ORIGIN == 'os':
    path_folder = os.path.join(os.getcwd(),'ariel-data-challenge-2025') # path to the folder containing the data
    path_out = os.path.join(os.getcwd(),'tmp','data_light_raw') # path to the folder to store the light data
    output_dir = os.path.join(os.getcwd(),'tmp','data_light_raw')  # path for the output directory


sensor_sizes_dict = {
    "AIRS-CH0": [[11250, 32, 356], [32, 356]],
    "FGS1": [[135000, 32, 32], [32, 32]],
}  # input, mask



def get_gain_offset():
    """
    Get the gain and offset for a given planet and sensor

    Unlike last year's challenge, all planets use the same adc_info.
    We can just hard code it.
    """
    gain = 0.4369
    offset = -1000.0
    return gain, offset


def read_data(planet_id, sensor, mode, signal_num=0):
    """
    Read the data for a given planet and sensor
    """
    # get all noise correction frames and signal
    if ORIGIN=='kaggle':
        signal = pd.read_parquet(
        ROOT + "/train/" + str(planet_id) + "/" + sensor + f"_signal_{signal_num}.parquet",
        engine="pyarrow",
        )
        dark_frame = pd.read_parquet(
        f"{ROOT}/{mode}/{planet_id}/{sensor}_calibration_{signal_num}/dark.parquet",
        engine="pyarrow",
        )
        dead_frame = pd.read_parquet(
        f"{ROOT}/{mode}/{planet_id}/{sensor}_calibration_{signal_num}/dead.parquet",
        engine="pyarrow",
        )
        linear_corr = pd.read_parquet(
        f"{ROOT}/{mode}/{planet_id}/{sensor}_calibration_{signal_num}/linear_corr.parquet",
        engine="pyarrow",
        )
        flat_frame = pd.read_parquet(
        f"{ROOT}/{mode}/{planet_id}/{sensor}_calibration_{signal_num}/flat.parquet",
        engine="pyarrow",
        )
    if ORIGIN=='os':
        train_folder = os.path.join(ROOT,'train')
        planet_folder = os.path.join(train_folder,str(planet_id))
        signal_folder = os.path.join(planet_folder,f"{sensor}_calibration_{signal_num}")
        
        signal = pd.read_parquet(
            os.path.join(planet_folder,f"{sensor}_signal_{signal_num}.parquet"),
            engine='pyarrow',
        ).values.astype(np.float64).reshape(sensor_sizes_dict[sensor][0])
        
        dark_frame = pd.read_parquet(
            os.path.join(signal_folder,"dark.parquet"),
            engine='pyarrow',
        ).values.astype(np.float64).reshape(sensor_sizes_dict[sensor][1])
        
        dead_frame = pd.read_parquet(
            os.path.join(signal_folder,"dead.parquet"),
            engine='pyarrow',
        ).values.astype(np.float64).reshape(sensor_sizes_dict[sensor][1])

        linear_corr = pd.read_parquet(
            os.path.join(signal_folder,"linear_corr.parquet"),
            engine='pyarrow',
        ).values.astype(np.float64).reshape([6] + sensor_sizes_dict[sensor][1])

        flat_frame = pd.read_parquet(
            os.path.join(signal_folder,"flat.parquet"),
            engine='pyarrow',
        ).values.astype(np.float64).reshape(sensor_sizes_dict[sensor][1])

    # read_frame = pd.read_parquet(
    #     f"{ROOT}/{mode}/{planet_id}/{sensor}_calibration/read.parquet",
    #     engine="pyarrow",
    # )

    return (
        signal,
        dark_frame,
        dead_frame,
        linear_corr,
        flat_frame,
        # read_frame,
    )


def ADC_convert(signal, gain, offset):
    """
    Step 1: Analog-to-Digital Conversion (ADC) correction

    The Analog-to-Digital Conversion (adc) is performed by the detector to convert the
    pixel voltage into an integer number. We revert this operation by using the gain
    and offset for the calibration files 'train_adc_info.csv'.
    """

    signal /= gain
    signal += offset
    return signal


def mask_hot_dead(signal, dead, dark):
    """
    Step 2: Mask hot/dead pixel

    The dead pixels map is a map of the pixels that do not respond to light and, thus,
    can't be accounted for any calculation. In all these frames the dead pixels are
    masked using python masked arrays. The bad pixels are thus masked but left
    uncorrected. Some methods can be used to correct bad-pixels but this task,
    if needed, is left to the participants.
    """

     
    hot = sigma_clip(dark, sigma=5, maxiters=5).mask
    if hot is None:
        hot = np.zeros_like(dark, dtype=bool)
    hot = np.tile(hot, (signal.shape[0], 1, 1)).astype(bool)
    dead = np.tile(dead, (signal.shape[0], 1, 1)).astype(bool)

    # Set values to np.nan where dead or hot pixels are found
    signal[dead] = np.nan
    signal[hot] = np.nan
    return signal


def apply_linear_corr(c, signal):
    """
    Step 3: linearity Correction

    The non-linearity of the pixels' response can be explained as capacitive leakage
    on the readout electronics of each pixel during the integration time. The number
    of electrons in the well is proportional to the number of photons that hit the
    pixel, with a quantum efficiency coefficient. However, the response of the pixel
    is not linear with the number of electrons in the well. This effect can be
    described by a polynomial function of the number of electrons actually in the well.
    The data is provided with calibration files linear_corr.parquet that are the
    coefficients of the inverse polynomial function and can be used to correct this
    non-linearity effect.
    Using horner's method to evaluate the polynomial
    """
    assert c.shape[0] == 6  # Ensure the polynomial is of degree 5

    return (
        (((c[5] * signal + c[4]) * signal + c[3]) * signal + c[2]) * signal + c[1]
    ) * signal + c[0]

def clean_dark(signal, dark, dt):
    """
    Step 4: dark current subtraction

    The data provided include calibration for dark current estimation, which can be
    used to pre-process the observations. Dark current represents a constant signal
    that accumulates in each pixel during the integration time, independent of the
    incoming light. To obtain the corrected image, the following conventional approach
    is applied: The data provided include calibration files such as dark frames or
    dead pixels' maps. They can be used to pre-process the observations. The dark frame
    is a map of the detector response to a very short exposure time, to correct for the
    dark current of the detector.

    image - (dark * dt)

    The corrected image is conventionally obtained via the following: where the dark
    current map is first corrected for the dead pixel.
    """

    dark = np.tile(dark, (signal.shape[0], 1, 1))
    signal -= dark * dt[:, np.newaxis, np.newaxis]
    return signal


def get_cds(signal):
    """
    Step 5: Get Correlated Double Sampling (CDS)

    The science frames are alternating between the start of the exposure and the end of
    the exposure. The lecture scheme is a ramp with a double sampling, called
    Correlated Double Sampling (CDS), the detector is read twice, once at the start
    of the exposure and once at the end of the exposure. The final CDS is the
    difference (End of exposure) - (Start of exposure).
    """

    return np.subtract(signal[1::2, :, :], signal[::2, :, :])

def bin_obs(cds_signal,binning):
    """
    Step 5 (Optional): Time Binning

    This step is performed mianly to save space. 
    Time series observations are binned together at specified frequency.
    """

    cds_transposed = cds_signal.transpose(0,1,3,2)
    cds_binned = np.zeros((cds_transposed.shape[0], cds_transposed.shape[1]//binning, cds_transposed.shape[2], cds_transposed.shape[3]))
    for i in range(cds_transposed.shape[1]//binning):
        cds_binned[:,i,:,:] = np.sum(cds_transposed[:,i*binning:(i+1)*binning,:,:], axis=1)
    return cds_binned

def correct_flat_field(flat, signal):
    """
    Step 6: Flat Field Correction

    The flat field is a map of the detector response to uniform illumination, to
    correct for the pixel-to-pixel variations of the detector, for example the
    different quantum efficiencies of each pixel.
    """

    return signal / flat

def process_planet(planet_id, signal_num, do_mask=True, do_nl_corr=True, do_dark=True, do_flat=True):
    """
    Process a single planet's data
    """
    if ORIGIN=='kaggle':
        axis_info = pd.read_parquet(ROOT + "axis_info.parquet")
    if ORIGIN=='os':
        axis_info = pd.read_parquet(os.path.join(ROOT,'axis_info.parquet'))
    dt_airs = axis_info["AIRS-CH0-integration_time"].dropna().values

    
    # load all data for this planet and sensor
    f_signal, f_dark_frame, f_dead_frame, f_linear_corr, f_flat_frame = read_data(
        planet_id, 'FGS1', mode=MODE, signal_num=signal_num
    )
    a_signal, a_dark_frame, a_dead_frame, a_linear_corr, a_flat_frame = read_data(
        planet_id, 'AIRS-CH0', mode=MODE, signal_num=signal_num
    )
    gain, offset = get_gain_offset()

    # Step 1: ADC correction
    f_signal = ADC_convert(f_signal, gain, offset)
    a_signal = ADC_convert(a_signal, gain, offset)
    # Step 2: Mask hot/dead pixel
    if do_mask:
        f_signal = mask_hot_dead(f_signal, f_dead_frame, f_dark_frame)
        a_signal = mask_hot_dead(a_signal, a_dead_frame, a_dark_frame)
    # Step 3: linearity Correction
    if do_nl_corr:
        f_signal = apply_linear_corr(f_linear_corr, f_signal)
        a_signal = apply_linear_corr(a_linear_corr, a_signal)
    # Step 4: dark current subtraction
    if do_dark=='':
        dt = np.ones(len(f_signal)) * 0.1
        dt[1::2] += 4.5
        f_signal = clean_dark(f_signal, f_dark_frame, dt)
        dt = dt_airs
        dt[1::2] += 0.1
        a_signal = clean_dark(a_signal, a_dark_frame, dt)
        

    # Step 5: Get Correlated Double Sampling (CDS)
    f_signal = get_cds(f_signal)
    a_signal = get_cds(a_signal)

    # Step 6: Flat Field Correction
    if do_flat:
        f_signal = correct_flat_field(f_flat_frame, f_signal)
        a_signal = correct_flat_field(a_flat_frame, a_signal)
 
    f_signal = np.nanmean(f_signal, axis=(1,2))
    f_signal = np.reshape(f_signal, (-1,12)).mean(axis=1)
    
    a_signal = np.nanmean(a_signal, axis=1)
    a_signal = [a_signal[:,i] for i in range(MIN_WL,MAX_WL)]
    signals = []
    signals.append(f_signal)
    signals = signals + a_signal
    signals = np.array(signals)
    # save the processed signal
    np.save(os.path.join(path_out,
        str(planet_id) + f"_{signal_num}_signal_{VERSION}.npy"),
        signals.astype(np.float64),
    )



#For multiple signals, use the wrapped function outside main
def wrapped(args):
        planet_id, signal_num = args
        return process_planet(
            planet_id,
            signal_num,
            do_mask=DO_MASK,
            do_nl_corr=DO_THE_NL_CORR,
            do_dark=DO_DARK,
            do_flat=DO_FLAT
    )

if __name__ == "__main__":

    

    

    if not os.path.exists(path_out):
        os.makedirs(path_out)
        print(f"Directory {path_out} created.")
    else:
        print(f"Directory {path_out} already exists.")

    if ORIGIN == 'kaggle':
        star_info = pd.read_csv(ROOT + f"/{MODE}_star_info.csv", index_col="planet_id")
    if ORIGIN == 'os':
        star_info = pd.read_csv(os.path.join(ROOT, f"{MODE}_star_info.csv"))
    planet_ids = star_info['planet_id'].tolist()

    planet_signals = {}

    for planet_id in os.listdir(os.path.join(ROOT,'train')):
        
        planet_folder = os.path.join(ROOT,'train', planet_id)
        
        if os.path.isdir(planet_folder):
            num_objs = len(os.listdir(planet_folder))
            if num_objs == 4:
                planet_signals[planet_id] = [0]
            elif num_objs == 8:
                planet_signals[planet_id] = [0, 1]
            else:
                print(f"Warning: planet {planet_id} with unwanted number of signals: {num_objs/4}")
    
    
    planet_signal_pairs = [(planet_id, signal_num) for planet_id, signal_nums in planet_signals.items() for signal_num in signal_nums]

    

    # Use up to 4 threads!
    with mp.Pool(processes=CHUNKS_SIZE) as pool:
        list(tqdm(pool.imap(wrapped, planet_signal_pairs), total=len(planet_signal_pairs)))

    # join processed signals in a single file

    print("Processing part 1 complete complete!")