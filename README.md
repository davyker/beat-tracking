# Beat Tracking

A beat tracking system for ballroom dance music using onset detection and tempo estimation.

## Setup

### 1. Create Python 3.10 environment

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Download audio files

```bash
python download_dataset.py
```

This will download the Ballroom audio files (~1.8GB) and extract them to `data1/`. The annotations in `data2/` are already included in this repository.

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

### Results

![Results by Genre](https://github.com/user-attachments/assets/0f1ea826-310e-44b3-8dc3-93e985c761f0)

## Project Structure

- `onset_detection.py` - Core beat tracking algorithms
- `process_single_file.py` - Single file processing
- `process_dataset.py` - Batch processing for datasets
- `visualisation.py` - Modular plotting functions
- `test.py` - Example usage
- `download_dataset.py` - Dataset download script

## Dataset

The Ballroom dataset contains annotated ballroom dance music across multiple genres including Waltz, Tango, ChaChaCha, Jive, Quickstep, Rumba, Samba, and VienneseWaltz.
