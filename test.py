from process_single_file import beatTracker
import numpy as np

beats, downbeats = beatTracker('data1/BallroomData/Rumba-American/Albums-GloriaEstefan_MiTierra-06.wav')
print(f"Beats: {np.round(beats, 3)}")
print(f"Downbeats: {np.round(downbeats, 3)}")