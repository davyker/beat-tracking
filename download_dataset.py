"""
Download the Ballroom dataset
"""

import os
import urllib.request
import zipfile
import sys

def download_dataset():
    dataset_url = "https://github.com/davyker/beat-tracking/releases/download/v1.0/ballroom-dataset.zip"
    zip_filename = "ballroom-dataset.zip"
    
    if os.path.exists("data1") and os.path.exists("data2"):
        print("Dataset folders already exist. Delete them to re-download.")
        return
    
    print(f"Downloading Ballroom dataset from GitHub release...")
    print(f"URL: {dataset_url}")
    
    try:
        def download_progress(block_num, block_size, total_size):
            downloaded = block_num * block_size
            percent = min(downloaded * 100 / total_size, 100)
            sys.stdout.write(f"\rProgress: {percent:.1f}%")
            sys.stdout.flush()
        
        urllib.request.urlretrieve(dataset_url, zip_filename, download_progress)
        print("\nDownload complete")
        
        print("Extracting dataset...")
        with zipfile.ZipFile(zip_filename, 'r') as zip_ref:
            zip_ref.extractall(".")
        print("Extraction complete")
        
        os.remove(zip_filename)
        print("Cleaned up zip file.")
        
        print("\nDataset ready. Folders created:")
        if os.path.exists("data1"):
            print("  - data1/")
        if os.path.exists("data2"):
            print("  - data2/")
            
    except Exception as e:
        print(f"\nError downloading dataset: {e}")
        print("\nPlease download manually from:")
        print(f"  {dataset_url}")
        print("Then extract the zip file in this directory.")

if __name__ == "__main__":
    download_dataset()