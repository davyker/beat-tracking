# Beat Tracking

A musical beat tracking system for that combines tempo estimation via autocorrelation with Bayesian beat selection. The algorithm addresses common issues like tempo drift and offbeat errors through a momentum-based beat adjustment mechanism that adapts to local timing variations while maintaining global tempo stability, and Bayesian scoring of candidates.

## Methods
- **Tempo Estimation**: Autocorrelation of onset strength envelope with Hamming window smoothing and parabolic interpolation
- **Beat Tracking**: Dynamic beat grid with momentum-based adjustment that allows beats to snap to nearby onsets while preventing long-term drift
- **Candidate Selection**: Bayesian scoring of multiple tempo & beat candidates using log-likelihood of onset strengths
- **Genre Adaptation**: Tempo priors customized for each ballroom dance style (90-135 for Cha-cha-cha, 160-180 for Jive, etc.)

## Results

The algorithm achieces 85.6% F-measure across all genres.
ChaChaCha, Quickstep, Tango, VienneseWaltz are easier, with scoring 90%+, being more rhythmic and having high-quality recordings.
Waltz and Rumba have worse quality recordings and are either syncopated or have rubato, making the task more difficult.

![Results by Genre - Table](https://github.com/user-attachments/assets/0f1ea826-310e-44b3-8dc3-93e985c761f0)
![Results by Genre - Plot](https://github.com/user-attachments/assets/9f8df034-f31c-4a20-87bc-111519d3074d)

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

## Project Structure

- `onset_detection.py` - Core beat tracking algorithms
- `process_single_file.py` - Single file processing
- `process_dataset.py` - Batch processing for datasets
- `visualisation.py` - Modular plotting functions
- `test.py` - Example usage
- `download_dataset.py` - Dataset download script

## Dataset

The Ballroom dataset contains annotated ballroom dance music across multiple genres including Waltz, Tango, ChaChaCha, Jive, Quickstep, Rumba, Samba, and VienneseWaltz.
