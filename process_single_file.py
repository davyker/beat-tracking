import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from onset_detection import *

def process_audio(file_path, genre=None, save_results=True):

    if not os.path.exists(file_path):
        print(f"Error: File {file_path} does not exist.")
        sys.exit(1)
    
    print(f"Processing file: {os.path.basename(file_path)}")
    print(f"Genre: {os.path.basename(os.path.dirname(file_path))}")
    
    ground_truth_beats = load_ground_truth_beats(file_path)
    
    # Call the onset detection function from main module
    onset_env = detect_onsets(file_path)
    beat_times, beat_frames, tempo, f_measure, precision, recall = beat_tracker(onset_env, genre=genre, ground_truth_beats=ground_truth_beats)
    print(f"Estimated tempo: {tempo:.2f} BPM")
    
    # Evaluate beat tracking against ground truth if available
    evaluation_results = {}
    if ground_truth_beats is not None and len(ground_truth_beats) > 0:
        
        downbeats = downbeat_detection(onset_env, beat_frames, genre)
        downbeat_time = downbeats[len(downbeats) // 2]
        print(f"Downbeat detected at {downbeat_time:.2f} seconds")

        # Find the closest ground truth beat to the detected downbeat
        closest_beat_idx = np.argmin(np.abs(ground_truth_beats[:, 0] - downbeat_time))
        closest_beat = ground_truth_beats[closest_beat_idx, 0]
        print(f"Closest ground truth beat to detected downbeat: {closest_beat:.2f} seconds")

        first_downbeat_time = ground_truth_beats[np.min(np.where(ground_truth_beats[:, 1] == 1)), 0]
        print(f"Time of first downbeat: {first_downbeat_time:.2f} seconds")

        is_downbeat = (ground_truth_beats[closest_beat_idx, 1] == 1)
        print(f"Detected downbeat is correct: {is_downbeat}")

        evaluation_results = {
            'f_measure': float(f_measure),
            'precision': float(precision),
            'recall': float(recall),
            'downbeat_correct': int(is_downbeat) # JSONs don't like booleans!!
        }
        
        print(f"\nBeat Tracking Evaluation:")
        print(f"    F-measure: {f_measure:.4f}")
        print(f"    Precision: {precision:.4f}")
        print(f"    Recall: {recall:.4f}")
        print(f"    Downbeat correct: {is_downbeat}")

    results = {
        'file': os.path.basename(file_path),
        'beat_times': beat_times.tolist(),
        'tempo': float(tempo),
        'downbeats': downbeats
    }
    
    if evaluation_results:
        results.update(evaluation_results)
    
    return results

# THIS IS THE FUNCTION TO BE CALLED BY TEST SCRIPT

def beatTracker(inputFile):
    genre = os.path.basename(os.path.dirname(inputFile))
    results = process_audio(inputFile, genre=genre)
    return results['beat_times'], results['downbeats']

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process a single ballroom audio file for onset detection.')
    parser.add_argument('file_path', type=str, help='Path to the audio file to process')
    parser.add_argument('--save', action='store_true', help='Save the results to a JSON file')
    
    args = parser.parse_args()

    genre = os.path.basename(os.path.dirname(args.file_path))
    results = process_audio(args.file_path, genre=genre, save_results=args.save)
    
    print(f"\nOnset Detection Summary for {os.path.basename(args.file_path)}:")
    
    if 'f_measure' in results:
        print(f"  Beat tracking F1 score: {results['f_measure']:.4f}")