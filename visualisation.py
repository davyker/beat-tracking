import matplotlib.pyplot as plt
import numpy as np
import librosa
import os


def plot_tempo_autocorrelation(lags, corr_bounded, smoothed_corr, best_idx, interpolated_lag, 
                                tempo, min_lag, min_bpm, max_bpm, sr, hop_length,
                                figsize=(12, 8), save_path=None, show=True):
    """
    Plot autocorrelation analysis for tempo estimation.
    
    Parameters:
    -----------
    lags : array
        Lag values
    corr_bounded : array
        Raw autocorrelation values
    smoothed_corr : array
        Smoothed autocorrelation values
    best_idx : int
        Index of the best peak
    interpolated_lag : float
        Interpolated lag value
    tempo : float
        Estimated tempo in BPM
    min_lag : int
        Minimum lag value
    min_bpm : float
        Minimum BPM for plotting range
    max_bpm : float
        Maximum BPM for plotting range
    sr : int
        Sample rate
    hop_length : int
        Hop length for frame conversion
    figsize : tuple
        Figure size (width, height)
    save_path : str
        Path to save figure (optional)
    show : bool
        Whether to display the plot
    
    Returns:
    --------
    fig : matplotlib.figure.Figure
        The created figure
    """
    fig = plt.figure(figsize=figsize)
    
    # Plot raw and smoothed autocorrelation in lag space
    plt.subplot(2, 1, 1)
    plt.plot(lags, corr_bounded, label='Raw Autocorrelation')
    plt.plot(lags, smoothed_corr, 'r-', linewidth=2, label='Smoothed')
    plt.axvline(x=min_lag + best_idx, color='g', linestyle='--', label=f'Peak at lag {min_lag + best_idx}')
    plt.axvline(x=interpolated_lag, color='m', linestyle='-.', label=f'Interpolated lag {interpolated_lag:.1f}')
    plt.xlabel('Lag (frames)')
    plt.ylabel('Correlation')
    plt.legend()
    plt.title('Autocorrelation for Tempo Estimation')
    
    # Plot in BPM space
    plt.subplot(2, 1, 2)
    bpms = 60 * sr / (lags * hop_length)
    plt.plot(bpms, corr_bounded, label='Raw')
    plt.plot(bpms, smoothed_corr, 'r-', linewidth=2, label='Smoothed')
    plt.axvline(x=tempo, color='g', linestyle='--', label=f'Tempo: {tempo:.1f} BPM')
    plt.xlabel('Tempo (BPM)')
    plt.ylabel('Correlation')
    plt.legend()
    plt.gca().set_xlim(min_bpm, max_bpm)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    if show:
        plt.show()
    
    return fig


def plot_beat_tracking_results(onset_env, beat_times, ground_truth_beats, 
                               lags, corr, implied_tempo, gt_tempo,
                               f_measure, precision, recall,
                               sr, hop_length,
                               figsize=(12, 8), save_path=None, show=True):
    """
    Plot beat tracking results with onset envelope and tempo estimation.
    
    Parameters:
    -----------
    onset_env : array
        Onset strength envelope
    beat_times : array
        Detected beat times in seconds
    ground_truth_beats : array or None
        Ground truth beat annotations
    lags : array
        Lag values for autocorrelation
    corr : array
        Autocorrelation values
    implied_tempo : float
        Detected tempo in BPM
    gt_tempo : float or None
        Ground truth tempo in BPM
    f_measure : float
        F-measure score
    precision : float
        Precision score
    recall : float
        Recall score
    sr : int
        Sample rate
    hop_length : int
        Hop length for frame conversion
    figsize : tuple
        Figure size (width, height)
    save_path : str
        Path to save figure (optional)
    show : bool
        Whether to display the plot
    
    Returns:
    --------
    fig : matplotlib.figure.Figure
        The created figure
    """
    fig = plt.figure(figsize=figsize)
    
    # Plot onset envelope with beats
    plt.subplot(2, 1, 1)
    times = librosa.times_like(onset_env, sr=sr, hop_length=hop_length)
    plt.plot(times, onset_env, label='Onset strength')
    plt.vlines(beat_times, 0, onset_env.max(), color='r', linestyle='--', label='Detected Beats')
    
    if ground_truth_beats is not None:
        plt.vlines(ground_truth_beats[:,0], 0, onset_env.max(), color='g', 
                  linestyle='-', linewidth=1, alpha=0.7, label='Ground Truth Beats')
    
    plt.legend()
    title_str = "Onset envelope with beats"
    if f_measure is not None:
        title_str += f" (F-meas.: {f_measure:.2f}, Prec.: {precision:.2f}, Rec.: {recall:.2f})"
    plt.title(title_str)
    plt.xlabel('Time (s)')
    plt.ylabel('Onset Strength')
    
    # Plot tempo autocorrelation
    plt.subplot(2, 1, 2)
    plt.plot(60 * sr / (lags * hop_length), corr)
    plt.axvline(x=implied_tempo, color='r', linestyle='--', 
               label=f'Detected Tempo: {implied_tempo:.1f} BPM')
    
    if gt_tempo is not None:
        plt.axvline(x=gt_tempo, color='g', linestyle='-.', 
                   label=f'Ground Truth Tempo: {gt_tempo:.1f} BPM')
    
    plt.xlabel('Tempo (BPM)')
    plt.ylabel('Autocorrelation')
    plt.title('Tempo estimation via autocorrelation')
    plt.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    if show:
        plt.show()
    
    return fig


def plot_genre_histograms(genre_dir, tempos, f_measures, precisions, recalls, hop_length,
                         hist_bins=50, figsize=(12, 8), save_path=None, show=True):
    """
    Plot histograms of metrics for a single genre.
    
    Parameters:
    -----------
    genre_dir : str
        Name of the genre
    tempos : list
        List of tempo values
    f_measures : list
        List of F-measure scores
    precisions : list
        List of precision scores
    recalls : list
        List of recall scores
    hop_length : int
        Hop length used in processing
    hist_bins : int
        Number of histogram bins
    figsize : tuple
        Figure size (width, height)
    save_path : str
        Path to save figure (optional)
    show : bool
        Whether to display the plot
    
    Returns:
    --------
    fig : matplotlib.figure.Figure
        The created figure
    """
    fig, axs = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle(f'{genre_dir} Genre Metrics with Hop Length {hop_length}')
    
    # Tempo histogram
    axs[0, 0].hist(tempos, bins=hist_bins, alpha=0.7)
    axs[0, 0].axvline(x=np.mean(tempos), color='r', linestyle='--', 
                     label=f'Mean: {np.mean(tempos):.2f} BPM')
    axs[0, 0].set_xlabel('Tempo (BPM)')
    axs[0, 0].set_ylabel('Count')
    axs[0, 0].set_title('Distribution of Estimated Tempos')
    axs[0, 0].legend()
    
    # F-measure histogram
    if f_measures:
        axs[0, 1].hist(f_measures, bins=hist_bins, alpha=0.7, color='blue')
        axs[0, 1].axvline(x=np.mean(f_measures), color='r', linestyle='--', 
                         label=f'Mean: {np.mean(f_measures):.4f}')
        axs[0, 1].set_xlabel('F-measure')
        axs[0, 1].set_ylabel('Count')
        axs[0, 1].set_title('Distribution of F-measures')
        axs[0, 1].legend()
    
    # Precision histogram
    if precisions:
        axs[1, 0].hist(precisions, bins=hist_bins, alpha=0.7, color='green')
        axs[1, 0].axvline(x=np.mean(precisions), color='r', linestyle='--', 
                         label=f'Mean: {np.mean(precisions):.4f}')
        axs[1, 0].set_xlabel('Precision')
        axs[1, 0].set_ylabel('Count')
        axs[1, 0].set_title('Distribution of Precisions')
        axs[1, 0].legend()
    
    # Recall histogram
    if recalls:
        axs[1, 1].hist(recalls, bins=hist_bins, alpha=0.7, color='red')
        axs[1, 1].axvline(x=np.mean(recalls), color='r', linestyle='--', 
                         label=f'Mean: {np.mean(recalls):.4f}')
        axs[1, 1].set_xlabel('Recall')
        axs[1, 1].set_ylabel('Count')
        axs[1, 1].set_title('Distribution of Recalls')
        axs[1, 1].legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    if show:
        plt.show()
    
    return fig


def plot_genre_tempo_comparison(genres_data, hop_length,
                               figsize=(12, 6), save_path=None, show=True):
    """
    Plot tempo comparison across genres with error bars.
    
    Parameters:
    -----------
    genres_data : dict
        Dictionary with genre names as keys and data dicts as values
        Each data dict should have 'tempo_mean' and 'tempo_std' keys
    hop_length : int
        Hop length used in processing
    figsize : tuple
        Figure size (width, height)
    save_path : str
        Path to save figure (optional)
    show : bool
        Whether to display the plot
    
    Returns:
    --------
    fig : matplotlib.figure.Figure
        The created figure
    """
    fig = plt.figure(figsize=figsize)
    
    # Sort genres by F-measure if available, otherwise alphabetically
    if 'f_measure_mean' in list(genres_data.values())[0]:
        sorted_genres = sorted(genres_data.keys(), 
                             key=lambda g: genres_data[g]['f_measure_mean'])
    else:
        sorted_genres = sorted(genres_data.keys())
    
    means = [genres_data[g]['tempo_mean'] for g in sorted_genres]
    stds = [genres_data[g]['tempo_std'] for g in sorted_genres]
    
    plt.bar(range(len(sorted_genres)), means, yerr=stds, capsize=10)
    plt.xticks(range(len(sorted_genres)), sorted_genres, rotation=45, ha='right')
    plt.xlabel('Genre')
    plt.ylabel('Tempo (BPM)')
    plt.title('Mean Tempo by Genre with Standard Deviation')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    if show:
        plt.show()
    
    return fig


def plot_evaluation_metrics_comparison(genres_data, hop_length,
                                      figsize=(12, 6), save_path=None, show=True):
    """
    Plot evaluation metrics comparison across genres using grouped bar chart.
    
    Parameters:
    -----------
    genres_data : dict
        Dictionary with genre names as keys and data dicts as values
        Each data dict should have metric keys: 'f_measure_mean', 'precision_mean', 
        'recall_mean', 'downbeat_correct_prop'
    hop_length : int
        Hop length used in processing
    figsize : tuple
        Figure size (width, height)
    save_path : str
        Path to save figure (optional)
    show : bool
        Whether to display the plot
    
    Returns:
    --------
    fig : matplotlib.figure.Figure
        The created figure
    """
    fig = plt.figure(figsize=figsize)
    
    # Sort genres by F-measure
    sorted_genres = sorted(genres_data.keys(), 
                         key=lambda g: genres_data[g].get('f_measure_mean', 0))
    
    f_measures = [genres_data[g].get('f_measure_mean', 0) for g in sorted_genres]
    precisions = [genres_data[g].get('precision_mean', 0) for g in sorted_genres]
    recalls = [genres_data[g].get('recall_mean', 0) for g in sorted_genres]
    downbeat_corrects = [genres_data[g].get('downbeat_correct_prop', 0) for g in sorted_genres]
    
    bar_width = 0.2
    r1 = np.arange(len(sorted_genres))
    r2 = [x + bar_width for x in r1]
    r3 = [x + bar_width for x in r2]
    r4 = [x + bar_width for x in r3]
    
    plt.bar(r1, f_measures, width=bar_width, label='F-measure', color='blue')
    plt.bar(r2, precisions, width=bar_width, label='Precision', color='green')
    plt.bar(r3, recalls, width=bar_width, label='Recall', color='red')
    plt.bar(r4, downbeat_corrects, width=bar_width, label='Downbeat Correct', color='purple')
    
    plt.xlabel('Genre')
    plt.ylabel('Score')
    plt.title('Beat Tracking Evaluation Metrics by Genre')
    plt.xticks([r + bar_width * 1.5 for r in range(len(sorted_genres))], 
              sorted_genres, rotation=45, ha='right')
    plt.legend()
    plt.ylim(0, 1.1)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    if show:
        plt.show()
    
    return fig


def save_and_show(fig, save_path=None, show=True):
    """
    Helper function to save and/or show a figure.
    
    Parameters:
    -----------
    fig : matplotlib.figure.Figure
        Figure to save/show
    save_path : str
        Path to save figure (optional)
    show : bool
        Whether to display the plot
    """
    if save_path:
        fig.savefig(save_path)
        print(f"Figure saved to {save_path}")
    if show:
        plt.show()
    else:
        plt.close(fig)


def format_genre_label(genre_name):
    """
    Format genre names for consistent display.
    
    Parameters:
    -----------
    genre_name : str
        Raw genre name
    
    Returns:
    --------
    str : Formatted genre name
    """
    # Split CamelCase and hyphenated names
    formatted = genre_name.replace('-', ' ')
    # Add space before capital letters (for CamelCase)
    import re
    formatted = re.sub(r'(?<!^)(?=[A-Z])', ' ', formatted)
    return formatted