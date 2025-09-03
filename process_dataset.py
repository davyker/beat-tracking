import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
import glob
import json
import pandas as pd
from tqdm import tqdm # type: ignore
from onset_detection import *
from process_single_file import process_audio
from visualisation import plot_genre_histograms, plot_genre_tempo_comparison, plot_evaluation_metrics_comparison

HIST_BINS = 50
PLOT_PER_GENRE = True

def process_dataset(data_path, genre='all', save_results=True):
    if genre == 'all':
        genre_dirs = [d for d in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, d))]
    else:
        # Make sure the genre exists
        if not os.path.isdir(os.path.join(data_path, genre)):
            print(f"Error: Genre directory '{genre}' not found in {data_path}")
            sys.exit(1)
        genre_dirs = [genre]
    
    print(f"Processing {len(genre_dirs)} genre directories: {', '.join(genre_dirs)}")
    
    results = {
        'songs': {},
        'genres': {}
    }
    
    total_songs = 0
    
    for genre_dir in genre_dirs:
        genre_path = os.path.join(data_path, genre_dir)
        wav_files = glob.glob(os.path.join(genre_path, '*.wav'))
        total_songs += len(wav_files)
    
    with tqdm(total=total_songs, desc="Processing songs") as pbar:
        for genre_dir in genre_dirs:
            genre_path = os.path.join(data_path, genre_dir)
            wav_files = glob.glob(os.path.join(genre_path, '*.wav'))
            
            genre_results = {
                'song_count': len(wav_files),
                'tempo_mean': 0,
                'tempo_std': 0,
                'f_measure_mean': 0,
                'precision_mean': 0,
                'recall_mean': 0,
                'downbeat_correct_prop': 0
            }
            
            tempos = []
            f_measures = []
            precisions = []
            recalls = []
            downbeat_corrects = []
            
            # Process each song in the genre
            for wav_file in wav_files:

                pbar.set_postfix_str(f"File: {os.path.basename(wav_file)}")
                song_result = process_audio(wav_file, genre=genre_dir, save_results=save_results)
                
                song_id = os.path.basename(wav_file)
                results['songs'][song_id] = song_result
                
                tempos.append(song_result['tempo'])
                
                if 'f_measure' in song_result:
                    f_measures.append(song_result['f_measure'])
                    precisions.append(song_result['precision'])
                    recalls.append(song_result['recall'])
                    downbeat_corrects.append(int(song_result['downbeat_correct']))
                
                pbar.update(1)

            print(f"Processed {len(wav_files)} songs in genre '{genre_dir}'")            
            
            if PLOT_PER_GENRE:
                save_path = os.path.join(data_path, f'genre_histograms_{genre_dir}_hop{HOP_LENGTH}.png')
                plot_genre_histograms(
                    genre_dir, tempos, f_measures, precisions, recalls, HOP_LENGTH,
                    hist_bins=HIST_BINS, figsize=(12, 8), save_path=save_path, show=True
                )            

            # Calculating genre aggregates
            if tempos:
                genre_results['tempo_mean'] = float(np.mean(tempos))
                genre_results['tempo_std'] = float(np.std(tempos))
                print(f"Mean tempo for genre '{genre_dir}': {genre_results['tempo_mean']:.2f} BPM")
            
            if f_measures:
                genre_results['f_measure_mean'] = float(np.mean(f_measures))
                genre_results['precision_mean'] = float(np.mean(precisions))
                genre_results['recall_mean'] = float(np.mean(recalls))
                print(f"Mean F-measure: {genre_results['f_measure_mean']:.4f}, Precision: {genre_results['precision_mean']:.4f}, Recall: {genre_results['recall_mean']:.4f}")

            if downbeat_corrects:
                genre_results['downbeat_correct_prop'] = float(np.mean(downbeat_corrects))
                print(f"Proportion of correct downbeats: {genre_results['downbeat_correct_prop']:.4f}")
            
            results['genres'][genre_dir] = genre_results
    
    if save_results:
        results_dir = 'final_results'
        os.makedirs(results_dir, exist_ok=True)
        results_path = os.path.join(results_dir, f'ballroom_results_{genre}_hop{HOP_LENGTH}_use_genres_{USE_GENRES}.json')
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {results_path}")
    
    print("\nResults Summary by Genre:")
    print("=========================")
    
    genres_df = pd.DataFrame(results['genres']).T
    genres_df = genres_df.sort_index()
    
    print(genres_df)

    overall_df = pd.DataFrame({
        'song_count': genres_df['song_count'].sum(),
        'tempo_mean': np.average(genres_df['tempo_mean'], weights=genres_df['song_count']),
        'tempo_std': np.sqrt(np.average((genres_df['tempo_std'] ** 2), weights=genres_df['song_count'])),
        'f_measure_mean': np.average(genres_df['f_measure_mean'], weights=genres_df['song_count']),
        'precision_mean': np.average(genres_df['precision_mean'], weights=genres_df['song_count']),
        'recall_mean': np.average(genres_df['recall_mean'], weights=genres_df['song_count']),
        'downbeat_correct_prop': np.average(genres_df['downbeat_correct_prop'], weights=genres_df['song_count'])
    }, index=['Overall'])

    print(overall_df)
    
    # Plotting genre comparison
    if len(genre_dirs) > 1:
        save_path = os.path.join(data_path, f'genre_tempo_comparison_hop{HOP_LENGTH}.png')
        plot_genre_tempo_comparison(
            results['genres'], HOP_LENGTH,
            figsize=(12, 6), save_path=save_path, show=True
        )
        
        # F-measure comparison if available
        if 'f_measure_mean' in results['genres'][genre_dirs[0]]:
            save_path = os.path.join(data_path, f'genre_evaluation_comparison_hop{HOP_LENGTH}.png')
            plot_evaluation_metrics_comparison(
                results['genres'], HOP_LENGTH,
                figsize=(12, 6), save_path=save_path, show=True
            )
    
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process the Ballroom dataset for onset detection and beat tracking.')
    parser.add_argument('--data_path', type=str, default='data1/BallroomData', help='Path to the Ballroom dataset directory')
    parser.add_argument('--genre', type=str, default='all', help='Genre to process (default = all)')
    parser.add_argument('--save', dest='save_results', action='store_true', help='Save the results to a json file')
    
    args = parser.parse_args()

    for arg in vars(args):
        print(f"{arg}: {getattr(args, arg)}")
    

    results = process_dataset(
        args.data_path, 
        genre=args.genre,
        save_results=args.save_results
    )