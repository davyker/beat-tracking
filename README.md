# Beat Tracking

A beat tracking system for ballroom dance music using onset detection and tempo estimation.

## Setup

### 1. Create conda environment

```bash
conda create -n beat-tracking python=3.10
conda activate beat-tracking
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Download dataset

```bash
python download_dataset.py
```

This will download the Ballroom dataset (~1.8GB) from the GitHub release and extract it to `data1/` and `data2/` folders.

## Usage

### Process single file

```python
from process_single_file import beatTracker

beats, downbeats = beatTracker('data1/BallroomData/Waltz/Album1-01.wav')
```

### Process entire dataset

```bash
python process_dataset.py --data_path data1/BallroomData --genre all --save
```

### Run test

```bash
python test.py
```

## Project Structure

- `onset_detection.py` - Core beat tracking algorithms
- `process_single_file.py` - Single file processing
- `process_dataset.py` - Batch processing for datasets
- `visualisation.py` - Modular plotting functions
- `test.py` - Example usage
- `download_dataset.py` - Dataset download script

## Dataset

The Ballroom dataset contains annotated ballroom dance music across multiple genres including Waltz, Tango, ChaChaCha, Jive, Quickstep, Rumba, Samba, and VienneseWaltz.