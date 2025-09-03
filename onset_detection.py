import librosa
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
from visualisation import plot_tempo_autocorrelation, plot_beat_tracking_results
# import mir_eval 

# GLOBAL VARIABLES

VERBOSE = False

# AUDIO
SR = 22050
HOP_LENGTH = 64
FRAMES_PER_SEC = SR / HOP_LENGTH
FMAX = 8000
PERCUSSIVE_ONLY = False

# TEMPO
USE_GENRES = True
if USE_GENRES:
    MIN_BPM = {'ChaChaCha': 90, 'Jive':160, 'Quickstep':200, 'Rumba-American':115, 'Rumba-International':95, 'Rumba-Misc':90, 'Samba':95, 'Tango':120, 'VienneseWaltz':170, 'Waltz':80}
    MAX_BPM = {'ChaChaCha': 135, 'Jive':180, 'Quickstep':210, 'Rumba-American':145, 'Rumba-International':105, 'Rumba-Misc':105, 'Samba':105, 'Tango':130, 'VienneseWaltz':180, 'Waltz':90}
    MIN_LAG = {genre: SR * 60 // (MAX_BPM[genre] * HOP_LENGTH) for genre in MIN_BPM}
    MAX_LAG = {genre: SR * 60 // (MIN_BPM[genre] * HOP_LENGTH) for genre in MAX_BPM}
else:
    MIN_BPM = 70
    MAX_BPM = 220
    MIN_LAG = SR * 60 // (MAX_BPM * HOP_LENGTH)
    MAX_LAG = SR * 60 // (MIN_BPM * HOP_LENGTH)

CANDIDATE_TEMPO_MULTIPLIERS = np.arange(0.98, 1.02, 0.004)

# CANDIDATE BEAT GENERATION
NUM_CANDIDATES = 25
SEARCH_START = 0
SEARCH_END = 15

# BEAT TRACKING
ALLOWED_ERROR = 0.13
MOMENTUM = 0.95
LEEWAY = 0.07 # (seconds) Also used in evaluation

# PLOTTING
PLOT = True
def PLOT_CONDITION(f_measure, precision, recall):
    return False
    # return precision < 0.9 or recall < 0.9

# ------------------------------

def vprint(*args, **kwargs):
    if VERBOSE:
        print(*args, **kwargs)

def onset_strength(y, aggregate='weighted'):
    """Onset strength function using spectral flux"""
    y = librosa.util.normalize(y)
    S = np.abs(librosa.stft(y, hop_length=HOP_LENGTH))
    
    if FMAX is not None and FMAX < SR/2:
        freq_bins = librosa.fft_frequencies(sr=SR, n_fft=2*(S.shape[0]-1))
        max_bin = np.searchsorted(freq_bins, FMAX)
        S = S[:max_bin, :]
    
    # Compute spectral flux and apply half-wave rectification
    diff = np.maximum(0, np.diff(S, axis=1, prepend=S[:, [0]])) # Prepend first column to maintain shape - gives diff of 0 at the start
    
    # Apply aggregation method
    if aggregate == 'weighted':
        n_bins = S.shape[0]
        freqs = librosa.fft_frequencies(sr=SR, n_fft=2*(n_bins-1))
        
        weights = np.ones(n_bins)
        
        # Masks for important frequency ranges
        low_mid_mask = (freqs >= 200) & (freqs <= 800)
        high_mid_mask = (freqs >= 1000) & (freqs <= 3000)
        
        weights[low_mid_mask] = 1.5  # Emphasize bass drum/percussion
        weights[high_mid_mask] = 1.2  # Emphasize transients/snares/hi-hats
        
        # Normalize
        weights = weights / np.sum(weights)
        
        # Apply weighted aggregation
        onset_env = np.sum(diff * weights[:, np.newaxis], axis=0)
        
    elif callable(aggregate):
        onset_env = aggregate(diff, axis=0)
    else:
        onset_env = np.mean(diff, axis=0)
    
    onset_env = np.ascontiguousarray(onset_env, dtype=np.float32)
    
    return onset_env

def detect_onsets(audio_file):

    y, _ = librosa.load(audio_file, sr=SR)
    
    if PERCUSSIVE_ONLY:
        y_harmonic, y_percussive = librosa.effects.hpss(y)
        x = y_percussive
    else:
        x = y
    
    onset_env = onset_strength(x)
    
    return onset_env

def tempo_estimate_with_parabolic_interpolation(onset_env, genre=None):
    """Tempo estimation using smoothed autocorrelation"""

    corr = librosa.autocorrelate(onset_env, max_size=len(onset_env))
    
    # Restrict to feasible range
    if genre is not None and USE_GENRES:
        min_lag = MIN_LAG[genre]
        max_lag = MAX_LAG[genre]
        min_bpm = MIN_BPM[genre]
        max_bpm = MAX_BPM[genre]
    else:
        min_lag = MIN_LAG
        max_lag = MAX_LAG
        min_bpm = MIN_BPM
        max_bpm = MAX_BPM

    corr_bounded = corr[min_lag:max_lag]
    lags = np.arange(min_lag, max_lag)
    
    # Apply smoothing to the autocorrelation
    window_size = 5
    smoothed_corr = np.convolve(corr_bounded, np.hamming(window_size)/np.sum(np.hamming(window_size)), mode='same')
    
    # Find peaks in the smoothed autocorrelation
    from scipy.signal import find_peaks
    peaks, _ = find_peaks(smoothed_corr)
    
    # If no peaks found, use the max  point
    if len(peaks) == 0:
        best_idx = np.argmax(smoothed_corr)
    else:
        # Find the strongest peak
        peak_values = smoothed_corr[peaks]
        best_peak_idx = np.argmax(peak_values)
        best_idx = peaks[best_peak_idx]
    
    # Parabolic interpolation for improved peak finding
    if 0 < best_idx < len(smoothed_corr) - 1:
        y0, y1, y2 = smoothed_corr[best_idx-1:best_idx+2]
        peak_offset = 0.5 * (y0 - y2) / (y0 - 2*y1 + y2)
        interpolated_idx = best_idx + peak_offset
    else:
        interpolated_idx = best_idx
    
    # Convert to actual lag and then to tempo
    interpolated_lag = min_lag + interpolated_idx
    tempo = float(60 * SR / (interpolated_lag * HOP_LENGTH))
    beat_period = float(interpolated_lag)
    
    if PLOT:
        plot_tempo_autocorrelation(
            lags, corr_bounded, smoothed_corr, best_idx, interpolated_lag,
            tempo, min_lag, min_bpm, max_bpm, SR, HOP_LENGTH,
            figsize=(12, 8), save_path=None, show=True
        )
    
    return beat_period, tempo, lags, smoothed_corr

def generate_candidate_beats(onset_env):
    # Generates a list of frame indeces which are good candidates for beats, which serve as the intial anchor points for beat tracking
    # After beat tracking we can optimise based on score
    search_indeces = np.arange(int(SEARCH_START * FRAMES_PER_SEC), int(SEARCH_END * FRAMES_PER_SEC + 1))

    # Now do peak picking within the search range
    # We choose the num_candidates highest peaks
    
    from scipy.signal import find_peaks
    peaks, _ = find_peaks(onset_env[search_indeces])
    peak_values = onset_env[search_indeces][peaks]

    # Sort by peak values
    sorted_peaks = peaks[np.argsort(peak_values)[::-1]]

    # Choose the NUM_CANDIDATES highest peaks
    candidate_beats = sorted_peaks[:NUM_CANDIDATES] + SEARCH_START * FRAMES_PER_SEC
    vprint(f"Chose {NUM_CANDIDATES} candidate beats: {candidate_beats}")

    return candidate_beats

def generate_candidate_start_idxs(onset_env, candidate_beats, beat_period, num_candidates=10):

    candidate_start_idxs = []
    for beat in candidate_beats:
        start_idx = float(beat)
        # print(f"Starting at {start_idx:.2f} frames, ie. {start_idx / FRAMES_PER_SEC:.2f} seconds")
        # Subtract the presumed beat period repeatedly until we reach the start of the signal
        # Also check onset_env is above 0.05*max(onset_env) before start_idx - this is to avoid placing the first beat too early
        while start_idx - beat_period >= 0 and (onset_env[:(int(start_idx)+1)].max() > 0.05 * onset_env.max()): ############################
            start_idx -= beat_period
        candidate_start_idxs.append(start_idx)

    vprint(f"Start indeces for candidate beats: {candidate_start_idxs}")

    return candidate_start_idxs

def evaluate_candidate_beat_frames(onset_env, beat_frames):
    num_beats = len(beat_frames)
    onsets_associated_with_beats = []
    for frame in beat_frames:
        if frame < len(onset_env):
            start = max(0, int(frame - LEEWAY * FRAMES_PER_SEC))
            end = min(len(onset_env), int(frame + LEEWAY * FRAMES_PER_SEC))
            onsets_associated_with_beats.append(start + np.argmax(onset_env[start:end]))
        else:
            print(f"Length of onset_env: {len(onset_env)}, Error: frame {frame} is greater than {len(onset_env)} therefore out of bounds")
            # print(f"All the relevant values for this particular candidate: beat_period: {beat_period:.2f}, tempo: {60 * SR / (beat_period * HOP_LENGTH):.2f}")
            # print(f"beat_frames: {beat_frames}")
            # Remove this frame and all subsequent frames
            beat_frames = beat_frames[:len(onsets_associated_with_beats)]
            num_beats = len(onsets_associated_with_beats)
            break
        
    # plt.plot(onset_env[onsets_associated_with_beats], marker='o')
    # plt.show()
    score = np.sum(np.log(onset_env[onsets_associated_with_beats]+1))/num_beats ########################
    # score = np.sum(onset_env[onsets_associated_with_beats])/num_beats

    return score

def downbeat_detection(onset_env, beat_frames, genre):
    # Finds the downbeat by dividing the beat_frames into n groups, where n is the metric of the song
    # One group being every n-th beat. Then we use evaluate_candidate_beat_frames to find the best group

    # Find the metric of the song
    print(f"Genre: {genre}")
    if genre in ['Waltz', 'VienneseWaltz']:
        metric = 3
    else:
        metric = 4
    print(f"Metric: {metric}")
    
    # One of the beats in beat_frames[:metric] is the downbeat
    # We will evaluate each group of metric beats and choose the one with the highest score
    groups = [beat_frames[i::metric] for i in range(metric)]
    scores = [evaluate_candidate_beat_frames(onset_env, group) for group in groups]
    print(f"Scores: {scores}")
    best_group_idx = np.argmax(scores)
    downbeats = groups[best_group_idx]
    downbeat_times = librosa.frames_to_time(downbeats, sr=SR, hop_length=HOP_LENGTH)

    return downbeat_times

def beat_tracker(onset_env, genre=None, ground_truth_beats=None):
    """
    Track beats using the onset strength envelope without using librosa.beat
    Returns the estimated beat times, beat frames, tempo, and evaluation metrics
    """
    
    beat_period_0, tempo_0, lags, corr = tempo_estimate_with_parabolic_interpolation(onset_env, genre=genre)

    # There are often errors of between -2% and +2% in tempo estimation, this helped quite a bit
    candidate_beat_periods = [beat_period_0 / i for i in CANDIDATE_TEMPO_MULTIPLIERS]
    candidate_beats = generate_candidate_beats(onset_env)

    candidate_beat_scores = []
    candidate_beat_frames = []

    # This iterates over candidate beat_periods and candidate beats and their associated start indeces
    # For each combo, it refines the beat positions by looking for nearby onset strength peaks
    # It then evaluates the beat frames based on the onset strength at the beat frames, and chooses thhe highest scoring beat frames
    for beat_period in candidate_beat_periods:
        candidate_start_idxs = generate_candidate_start_idxs(onset_env, candidate_beats, beat_period)

        for start_idx, candidate_beat in zip(candidate_start_idxs, candidate_beats):

            temp_beat_period = beat_period
            num_beats = int((len(onset_env) - start_idx) / temp_beat_period)
            beat_frames = start_idx + np.arange(num_beats) * temp_beat_period
            
            # Refine beat positions by looking for nearby onset strength peaks
            window_radius = int(temp_beat_period * ALLOWED_ERROR)  # Look for peaks within x% of beat period - window is actually 2x this
            
            for i in range(num_beats):
                frame = beat_frames[i]
                if frame >= len(onset_env):
                    # This means the beat frame is out of bounds
                    # Remove this frame and all subsequent frames
                    beat_frames = beat_frames[:i]
                    num_beats = i
                    break
                start = max(0, int(frame - window_radius))
                end = min(len(onset_env), int(frame + window_radius + 1))
                
                if start < end:
                    # Find local peak
                    # This chooses local peak based on onset strength but weighted by distance from current frame, so the middle of onset_end[start:end] is weighted the most, in a triangular fashion
                    # This is because I noticed that a slightly higher peak may be chosen even when there is another onset right on the beat
                    # local_peak = start + np.argmax(onset_env[start:end] * (1 - np.abs(np.arange(start, end) - frame) / window_size))
                    # local_peak = onset_frames[np.argmin(np.abs(onset_frames - frame))]

                    local_peak = start + np.argmax(onset_env[start:end])
                    difference = local_peak - frame

                    # adjust beat frames from i onwards using momentum
                    beat_frames[i:] += np.float64(difference * (MOMENTUM)**(np.arange(0, num_beats-i)))

                    # beat_frames[i:] += np.int64(difference * 0.5**np.arange(1, num_beats+1-i))
                    # beat_frames[i:] += difference * onset_env[local_peak] / max(onset_env)
                    # beat_frames[i:] += difference

                    # Update beat_period
                    if i > 0:
                        temp_beat_period = MOMENTUM * temp_beat_period + (1 - MOMENTUM) * (beat_frames[i] - beat_frames[i-1])
                    window_radius = int(temp_beat_period * ALLOWED_ERROR)
            
            # Now we get a new implied tempo based on the first and last detected beats
            avg_beat_period = (beat_frames[-1] - beat_frames[0]) / (num_beats - 1)
            implied_tempo = 60 * SR / (avg_beat_period * HOP_LENGTH)

            # Score the candidate beat frames
            score = evaluate_candidate_beat_frames(onset_env, beat_frames)

            candidate_beat_scores.append(score)
            candidate_beat_frames.append(beat_frames)

    beat_frames = candidate_beat_frames[np.argmax(candidate_beat_scores)]
    beat_times = librosa.frames_to_time(beat_frames, sr=SR, hop_length=HOP_LENGTH)

    avg_beat_period = (beat_frames[-1] - beat_frames[0]) / (len(beat_frames) - 1)
    implied_tempo = 60 * SR / (avg_beat_period * HOP_LENGTH)

    if ground_truth_beats is not None:
        f_measure, precision, recall = evaluate_beat_tracking(ground_truth_beats, beat_times)
    else:
        f_measure, precision, recall = None, None, None
    print(f"f_measure: {f_measure:.2f}, precision: {precision:.2f}, recall: {recall:.2f}")
    
    if ground_truth_beats is not None and len(ground_truth_beats) > 1:
        gt_tempo = 60 / np.mean(np.diff(ground_truth_beats[:,0]))
        print(f"Ground truth tempo: {gt_tempo:.2f} BPM")

    if PLOT or PLOT_CONDITION(f_measure, precision, recall):
        plot_beat_tracking_results(
            onset_env, beat_times, ground_truth_beats,
            lags, corr, implied_tempo, gt_tempo if ground_truth_beats is not None and len(ground_truth_beats) > 1 else None,
            f_measure, precision, recall,
            SR, HOP_LENGTH,
            figsize=(12, 8), save_path=None, show=True
        )
    
    return beat_times, beat_frames, implied_tempo, f_measure, precision, recall

def evaluate_beat_tracking(reference_beats, estimated_beats):
    """
    Evaluate beat tracking results
    Returns F-measure, precision, and recall
    """
    true_positives = 0

    reference_beats = reference_beats[:,0]
    
    for est_beat in estimated_beats:

        differences = np.abs(reference_beats - est_beat)
        min_diff = np.min(differences)
        
        if min_diff <= LEEWAY:
            true_positives += 1
    
    precision = true_positives / len(estimated_beats) if len(estimated_beats) > 0 else 0
    recall = true_positives / len(reference_beats) if len(reference_beats) > 0 else 0
    
    f_measure = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return f_measure, precision, recall

def load_ground_truth_beats(audio_file_path):
    base_filename = os.path.splitext(os.path.basename(audio_file_path))[0]
    ground_truth_path = os.path.join("data2", "BallroomAnnotations", "ballroomGTbeats", f"{base_filename}.beats")
    
    if os.path.exists(ground_truth_path):
        try:
            ground_truth_beats = np.loadtxt(ground_truth_path)
            print(f"Loaded {ground_truth_beats.shape} ground truth beats from {ground_truth_path}")
            return ground_truth_beats
        except Exception as e:
            print(f"Error loading ground truth file {ground_truth_path}: {e}")
            return None
    else:
        print(f"Ground truth file not found: {ground_truth_path}")
        return None