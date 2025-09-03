"""
Download the Ballroom audio dataset (data1).
Annotations (data2) are included in the repository.
"""

import os
import urllib.request
import tarfile
import sys
import shutil

def download_dataset():
    data1_url = "https://mtg.upf.edu/ismir2004/contest/tempoContest/data1.tar.gz"
    
    if os.path.exists("data1"):
        print("Audio dataset (data1) already exists. Delete it to re-download.")
        return
    
    def download_progress(block_num, block_size, total_size):
        downloaded = block_num * block_size
        percent = min(downloaded * 100 / total_size, 100)
        sys.stdout.write(f"\rProgress: {percent:.1f}%")
        sys.stdout.flush()
    
    try:
        # Download and organize data1
        print(f"Downloading audio files from {data1_url}...")
        urllib.request.urlretrieve(data1_url, "data1.tar.gz", download_progress)
        print("\nExtracting audio files...", end=" ")
        with tarfile.open("data1.tar.gz", "r:gz") as tar:
            tar.extractall(".")
        os.remove("data1.tar.gz")
        
        # Move BallroomData into data1
        if os.path.exists("BallroomData"):
            os.makedirs("data1", exist_ok=True)
            shutil.move("BallroomData", "data1/BallroomData")
        print("Done.")

        print("\nAudio dataset ready.")
            
    except Exception as e:
        print(f"\nError downloading dataset: {e}")
        print("Try downloading manually from:")
        print(f"  data1: {data1_url}")
        print(f"  data2: {data2_url}")
        print("Then extract the tar.gz files.")

if __name__ == "__main__":
    download_dataset()