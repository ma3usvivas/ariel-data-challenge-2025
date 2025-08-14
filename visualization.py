import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython.display import HTML
import seaborn as sns

def playAnimation(data, sensor):

    
    total_flux = np.nansum(data, axis=(1, 2))
    # Key frame indices (quarter-points + first/last)
    base_frames = [len(data)*i//100 for i in range(100)]+[len(data)-1]

    # Find brightest and darkest frame indices
    brightest_idx = np.argmax(total_flux)
    darkest_idx = np.argmin(total_flux)

    # Add to the frame list, avoiding duplicates
    extra_frames = []
    for idx in [brightest_idx, darkest_idx]:
        if idx not in base_frames and (idx != -1 and idx != len(data)-1):
            extra_frames.append(idx)
    frames = base_frames + extra_frames
    frames = sorted(frames)

    # Set up clean styling
    plt.style.use('default')
    if sensor == 'FGS1':
        fig, ax = plt.subplots(figsize=(10, 6))
        vmin, vmax = np.nanpercentile(data[0], [1, 99])
        ax.set_title("Frame 0")
        im = ax.imshow(data[frames[0]], cmap='hot', aspect='equal', vmin=vmin, vmax=vmax)
        ax.set_xticks([])
        ax.set_yticks([])
        title_text = ax.text(0.5, 1.05, "", transform=ax.transAxes, ha="center", va="bottom", fontsize=11)
    elif sensor == 'AIRS':    
        fig, ax = plt.subplots(figsize=(10, 3))
        vmin, vmax = np.nanpercentile(data[0], [1, 99])
        ax.set_title("Frame 0")
        im = ax.imshow(data[frames[0]], cmap='hot', aspect='equal', vmin=vmin, vmax=vmax)
        ax.set_xticks([])
        ax.set_yticks([])
        title_text = ax.text(0.5, 1.05, "", transform=ax.transAxes, ha="center", va="bottom", fontsize=11)
        

        

    # Atualiza cada frame da animação
    def update(i):
        idx = frames[i]
        frame = data[idx]
        im.set_array(data[idx])

        vmin, vmax = np.nanpercentile(frame, [2, 98])
        im.set_clim(vmin, vmax)
        
        time_min = idx * 0.1 / 60
        if idx == brightest_idx:
            title = f'Brightest\nT={time_min:.1f} min'
        elif idx == darkest_idx:
            title = f'Darkest\nT={time_min:.1f} min'
        else:
            title = f'T={time_min:.1f} min'
        title_text.set_text(title)
        ax.set_xticks([])
        ax.set_yticks([])
        return [im, title_text]
    
    ani = FuncAnimation(fig, update, frames=len(frames), blit=True, repeat = False)
    plt.close(fig)
    return ani

def seeTotalIntensityFGS1(data):
    """
    Plota o valor máximo de cada observação ao longo da primeira dimensão do array.
    
    Parâmetros:
    - array3d: numpy array com shape (67500, 32, 32)
    """
    total_flux = np.nansum(data, axis=(1,2))
    fig, ax_light = plt.subplots(figsize = (18,10))
    time_hours = np.arange(len(total_flux)) * 0.1 / 3600
    sample = slice(None, None, max(1, len(total_flux)//2000))

    ax_light.plot(time_hours[sample], total_flux[sample], 
                 color='lightsteelblue', alpha=0.4, linewidth=0.5, label='Raw flux')

    window = 500
    moving_avg = pd.Series(total_flux).rolling(window, center=True).mean()
    ax_light.plot(time_hours[sample], moving_avg.iloc[sample], 
                 color='darkblue', linewidth=3, label=f'{window}-frame average')

    colors = ['red', 'orange', 'green', 'purple', 'brown', 'lime', 'black']
    base_frames = [len(data)*i//10 for i in range(10)]+[len(data)-1]

    # Find brightest and darkest frame indices
    brightest_idx = np.argmax(total_flux)
    darkest_idx = np.argmin(total_flux)

    # Add to the frame list, avoiding duplicates
    extra_frames = []
    for idx in [brightest_idx, darkest_idx]:
        if idx not in base_frames and (idx != -1 and idx != len(data)-1):
            extra_frames.append(idx)
    frames = base_frames + extra_frames
    frames = sorted(frames)
    for i, idx in enumerate(frames):
        time_point = idx * 0.1 / 3600
        ax_light.axvline(time_point, color=colors[i % len(colors)], alpha=0.8, linewidth=1.5, linestyle='--')

    ax_light.set_xlabel('Time (hours)', fontsize=12)
    ax_light.set_ylabel('Total Flux (counts)', fontsize=12)
    ax_light.set_title(f'Transit Light Curve - Planet {1873185}', fontsize=14, pad=15)
    ax_light.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax_light.legend(loc='upper right', framealpha=0.9)

    smooth_flux = moving_avg.dropna()
    flux_min, flux_max = smooth_flux.min(), smooth_flux.max()
    flux_range = flux_max - flux_min
    margin = flux_range * 0.1
    ax_light.set_ylim(flux_min - margin, flux_max + margin)

    plt.tight_layout(pad=2.0)
    plt.show()

def seeMeanIntensityFGS1(data):
    """
    Plota o valor máximo de cada observação ao longo da primeira dimensão do array.
    
    Parâmetros:
    - array3d: numpy array com shape (67500, 32, 32)
    """
    total_flux = np.nanmean (data, axis=(1,2))
    fig, ax_light = plt.subplots(figsize = (18,10))
    time_hours = np.arange(len(total_flux)) * 0.1 / 3600
    sample = slice(None, None, max(1, len(total_flux)//2000))

    ax_light.plot(time_hours[sample], total_flux[sample], 
                 color='lightsteelblue', alpha=0.4, linewidth=0.5, label='Raw flux')

    window = 500
    moving_avg = pd.Series(total_flux).rolling(window, center=True).mean()
    ax_light.plot(time_hours[sample], moving_avg.iloc[sample], 
                 color='darkblue', linewidth=3, label=f'{window}-frame average')

    colors = ['red', 'orange', 'green', 'purple', 'brown', 'lime', 'black']
    base_frames = [len(data)*i//10 for i in range(10)]+[len(data)-1]

    # Find brightest and darkest frame indices
    brightest_idx = np.argmax(total_flux)
    darkest_idx = np.argmin(total_flux)

    # Add to the frame list, avoiding duplicates
    extra_frames = []
    for idx in [brightest_idx, darkest_idx]:
        if idx not in base_frames and (idx != -1 and idx != len(data)-1):
            extra_frames.append(idx)
    frames = base_frames + extra_frames
    frames = sorted(frames)
    for i, idx in enumerate(frames):
        time_point = idx * 0.1 / 3600
        ax_light.axvline(time_point, color=colors[i % len(colors)], alpha=0.8, linewidth=1.5, linestyle='--')

    ax_light.set_xlabel('Time (hours)', fontsize=12)
    ax_light.set_ylabel('Total Flux (counts)', fontsize=12)
    ax_light.set_title(f'Transit Light Curve - Planet {1873185}', fontsize=14, pad=15)
    ax_light.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax_light.legend(loc='upper right', framealpha=0.9)

    smooth_flux = moving_avg.dropna()
    flux_min, flux_max = smooth_flux.min(), smooth_flux.max()
    flux_range = flux_max - flux_min
    margin = flux_range * 0.1
    ax_light.set_ylim(flux_min - margin, flux_max + margin)

    plt.tight_layout(pad=2.0)
    plt.show()

def seeAIRS(data, index):
    plt.figure(figsize=(10, 3))
    sns.heatmap(data[index])
    plt.ylabel('spatial dimension')
    plt.xlabel('wavelength dimension')
    plt.show()

def seeMeanIntensityAIRS(data):
    total_flux = np.nanmean(np.nanmean(data, axis=2), axis=1)
    cum_signal = np.cumsum(total_flux)
    window=42
    smooth_signal = (cum_signal[window:] - cum_signal[:-window]) / window
    _, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
    ax1.plot(total_flux, label='raw net signal')
    ax1.legend()
    ax2.plot(smooth_signal, color='c', label='smoothened net signal')
    ax2.legend()
    ax2.set_xlabel('time')
    for time_step in [17532, 21852, 44412, 48012]:
        ax2.axvline(time_step * 11250 // 135000, color='gray')
    plt.suptitle('AIRS-CH0 time series', y=0.96)
    plt.show()
    
