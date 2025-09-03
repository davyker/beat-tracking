"""
Download the Ballroom dataset
"""

import os
import urllib.request
import tarfile
import sys
import tarfile
import shutil

def download_dataset():
    data1_url = "https://mtg.upf.edu/ismir2004/contest/tempoContest/data1.tar.gz"
    data2_url = "https://mtg.upf.edu/ismir2004/contest/tempoContest/data2.tar.gz"
    
    if os.path.exists("data1") and os.path.exists("data2"):
        print("Dataset folders already exist. Delete them to re-download.")
        return
    
    def download_progress(block_num, block_size, total_size):
        downloaded = block_num * block_size
        percent = min(downloaded * 100 / total_size, 100)
        sys.stdout.write(f"\rProgress: {percent:.1f}%")
        sys.stdout.flush()
    
    try:
        # Download and organize data1
        if not os.path.exists("data1"):
            print(f"Downloading data1 (audio files) from {data1_url}...")
            urllib.request.urlretrieve(data1_url, "data1.tar.gz", download_progress)
            print("\nExtracting data1...", end=" ")
            with tarfile.open("data1.tar.gz", "r:gz") as tar:
                tar.extractall(".")
            os.remove("data1.tar.gz")
            
            # Move BallroomData into data1
            if os.path.exists("BallroomData"):
                os.makedirs("data1", exist_ok=True)
                shutil.move("BallroomData", "data1/BallroomData")
            print("Done.")
        
        # Download and organize data2
        if not os.path.exists("data2"):
            print(f"Downloading data2 (annotations) from {data2_url}...")
            urllib.request.urlretrieve(data2_url, "data2.tar.gz", download_progress)
            print("\nExtracting data2...", end=" ")
            with tarfile.open("data2.tar.gz", "r:gz") as tar:
                tar.extractall(".")
            os.remove("data2.tar.gz")
            
            # Move BallroomAnnotations into data2
            if os.path.exists("BallroomAnnotations"):
                os.makedirs("data2", exist_ok=True)
                shutil.move("BallroomAnnotations", "data2/BallroomAnnotations")
            print("Done.")

        print("\nDataset ready.")
            
    except Exception as e:
        print(f"\nError downloading dataset: {e}")
        print("Try downloading manually from:")
        print(f"  data1: {data1_url}")
        print(f"  data2: {data2_url}")
        print("Then extract the tar.gz files.")

if __name__ == "__main__":
    download_dataset()