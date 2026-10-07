# Grammar Scoring Engine for Spoken Audio

Predicts a **grammar score from 0 to 5** for 45–60 second recordings of people speaking English.
The task is regression, evaluated with **Pearson correlation** and **RMSE**.

The work lives in one notebook, [Grammar_recognition_pipeline.ipynb](Grammar_recognition_pipeline.ipynb), and is split into parts:

| Part | What it does | Status |
|---|---|---|
| **Part 1: Data and EDA** | Loads the labels, checks every audio file, builds a signal-level feature table, compares noise clips with speech, and defines one shared preprocessing function. | Done |
| **Part 2: Transcription and noise gate** | Transcribes all 985 clips with OpenAI Whisper, checks transcript quality, computes ASR statistics, and builds a cross-validated **noise gate** that detects the label-0 noise clips. | Done |
| Part 3: Features | Grammar and fluency features from the cleaned transcripts. | Planned |
| Part 4: Models | Ridge / gradient boosting regressors, cross-validated RMSE and Pearson, final submission. | Planned |

---

## Table of contents

1. [Repository layout](#1-repository-layout)
2. [Requirements](#2-requirements)
3. [Setup guide (local machine)](#3-setup-guide-local-machine)
4. [Getting the dataset](#4-getting-the-dataset)
5. [Running the notebook](#5-running-the-notebook)
6. [Run switches](#6-run-switches)
7. [Running on Kaggle](#7-running-on-kaggle)
8. [Generated files (`outputs/`)](#8-generated-files-outputs)
9. [Key results so far](#9-key-results-so-far)
10. [Troubleshooting](#10-troubleshooting)
11. [What is and is not in git](#11-what-is-and-is-not-in-git)

---

## 1. Repository layout

```
shl_assignment/
├── Grammar_recognition_pipeline.ipynb   # the whole pipeline (Part 1 + Part 2)
├── requirements.txt                     # every library with exact versions (PyTorch installed separately)
├── train.csv                            # 769 rows: filename, label (0.0 to 5.0)   (NOT in git, download)
├── test.csv                             # 216 rows: filename, label (always -1)    (NOT in git, download)
├── sample_submission.csv                # format example only (see note below)     (NOT in git, download)
├── train/                               # 769 .wav files, ~1.3 GB                  (NOT in git, download)
├── test/                                # 216 .wav files, ~0.3 GB                  (NOT in git, download)
├── outputs/                             # cached results (in git, ~13 MB)
│   ├── audio_features.csv               # Part 1 signal features, one row per clip
│   ├── asr_stats.csv                    # Part 2 per-clip ASR statistics
│   ├── noise_gate_predictions.csv       # noise probabilities (train out-of-fold + test)
│   └── asr/
│       ├── pilot_small_noprompt.jsonl   # Whisper pilot, 24 clips, no prompt
│       ├── pilot_small_prompt.jsonl     # Whisper pilot, 24 clips, disfluent prompt
│       ├── asr_full_small_prompt.jsonl  # full transcription of all 985 clips
│       └── transcripts.csv              # flat table of the cleaned transcripts
├── .gitignore
└── README.md
```

> **Important:** `train/` and `test/` reuse the same filenames (212 of 216 test names also exist in `train/`)
> but they are **different recordings**. Always identify a clip by `(split, filename)`, never by filename alone.

> **Note on `sample_submission.csv`:** only 25 of its 204 filenames appear in `test.csv`. Use it for the column
> format (`filename,label`) only. The final submission must be built from `test.csv`.

---

## 2. Requirements

### Hardware

| | Minimum | Used for the reference run |
|---|---|---|
| OS | Windows 10/11, Linux or macOS | Windows 11 |
| RAM | 8 GB | 16 GB |
| Disk | ~3 GB free (1.6 GB dataset + 0.5 GB Whisper weights + env) | |
| GPU | Optional. Needed only to re-run Whisper in reasonable time | NVIDIA RTX 4060 Laptop GPU (8 GB) |

- **Without a GPU** you can still run the whole notebook: the Whisper transcripts are already cached in `outputs/asr/`,
  so set `RUN_WHISPER = False` (see [Run switches](#6-run-switches)).
- **Re-transcribing** all 985 clips took **146 minutes** on the RTX 4060 (peak 1.23 GiB VRAM). On a CPU it is 10–30× slower.

### Software

- **Python 3.12** (reference run: 3.12.14 from conda-forge)
- **Conda** (Miniconda / Anaconda / Miniforge) – recommended, the reference environment is a conda env named `dlenv`
- **Git**
- **Jupyter** (JupyterLab, classic Notebook, or VS Code with the Jupyter extension)
- **PyTorch** – installed separately, matched to your CUDA version (step 3.3)
- `ffmpeg` is **not** required: audio is loaded with `librosa`/`soundfile` and passed to Whisper as an array.

Pinned package versions are in [requirements.txt](requirements.txt).

---

## 3. Setup guide (local machine)

Commands below work in **PowerShell, Git Bash, or a Linux/macOS terminal** unless noted.

### 3.1 Clone the repository

```bash
git clone https://github.com/shivaonkruger/Grammer_scoring_speech_to_text_pipeline.git
cd Grammer_scoring_speech_to_text_pipeline
```

### 3.2 Create the Python environment

**Option A – Conda (recommended, matches the reference run)**

```bash
conda create -n dlenv -c conda-forge python=3.12 -y
conda activate dlenv
```

**Option B – plain `venv`**

```bash
# Windows (PowerShell)
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1

# Linux / macOS
python3.12 -m venv .venv
source .venv/bin/activate
```

Then upgrade pip:

```bash
python -m pip install --upgrade pip
```

### 3.3 Install PyTorch first

PyTorch is **not in** `requirements.txt` because the right build depends on your GPU and CUDA driver.

1. Check your CUDA driver version (NVIDIA GPUs only):
   ```bash
   nvidia-smi
   ```
   The top-right corner shows the highest CUDA version your driver supports.
2. Go to <https://pytorch.org/get-started/locally/>, pick your OS, `pip`, and the CUDA version **at or below** the one `nvidia-smi` shows,
   and run the command it gives you. It looks like:
   ```bash
   pip install torch --index-url https://download.pytorch.org/whl/cuXXX
   ```
3. **No NVIDIA GPU?** Install the CPU build:
   ```bash
   pip install torch --index-url https://download.pytorch.org/whl/cpu
   ```

The reference run used `torch==2.14.1+cu132` (CUDA 13.2).

Verify:

```bash
python -c "import torch; print(torch.__version__, '| CUDA available:', torch.cuda.is_available())"
```

Install it **before** step 3.4: `openai-whisper` depends on PyTorch, and pip would otherwise pull a default build.

### 3.4 Install everything else

```bash
pip install -r requirements.txt
```

This installs every other library with the exact versions of the reference run, including `openai-whisper`, the spaCy
English model and `errant`.
The Whisper **`small`** weights (`small.pt`, ~461 MB) are downloaded automatically to `~/.cache/whisper` the first time
the notebook loads the model. You can pre-download them with:

```bash
python -c "import whisper; whisper.load_model('small')"
```

### 3.5 Register the environment as a Jupyter kernel

The notebook expects a kernel called **`dlenv`** (display name *Python (dlenv)*).

```bash
pip install ipykernel jupyterlab
python -m ipykernel install --user --name dlenv --display-name "Python (dlenv)"
```

If you named your environment differently, just pick your own kernel when you open the notebook.

### 3.6 Verify the install

```bash
python -c "import numpy, pandas, sklearn, librosa, soundfile, seaborn, tqdm; print('Part 1 OK')"
python -c "import torch, whisper; print('Part 2 OK | torch', torch.__version__, '| cuda', torch.cuda.is_available())"
```

---

## 4. Getting the dataset

The dataset is **not** stored in git. It is about **1.6 GB**: 985 `.wav` clips (~13 hours of audio) plus 3 CSV files.

**Download:** [Google Drive – dataset](GOOGLE_DRIVE_LINK_HERE)

1. Open the link and download the folder (Google Drive zips it automatically; large folders may be split into several zips).
2. Unzip everything.
3. Move `train/`, `test/`, `train.csv`, `test.csv` and `sample_submission.csv` into the project root
   (the same folder as the notebook), so the layout looks like this:

```
shl_assignment/
├── Grammar_recognition_pipeline.ipynb
├── train.csv
├── test.csv
├── sample_submission.csv
├── train/
│   ├── audio_0.wav
│   ├── audio_1.wav
│   └── ...            (769 files)
└── test/
    ├── audio_0.wav
    └── ...            (216 files)
```

Checks:

- `train/` must contain **769** `.wav` files and `test/` **216**.
- Make sure the folders are not nested one level too deep (e.g. `train/train/audio_0.wav`) after unzipping.
- Every clip should be 16 kHz, mono, 16-bit PCM (the notebook verifies this in Step 4).
- The notebook looks for files at `DATA_DIR/<split>/<filename>`; locally `DATA_DIR` is the folder the notebook is in (`.`).

Quick count:

```bash
# Git Bash / Linux / macOS
ls train/*.wav | wc -l     # 769
ls test/*.wav  | wc -l     # 216
```

```powershell
# PowerShell
(Get-ChildItem train -Filter *.wav).Count   # 769
(Get-ChildItem test  -Filter *.wav).Count   # 216
```

These files are listed in `.gitignore`, so they will never be committed by accident.

---

## 5. Running the notebook

1. Activate the environment: `conda activate dlenv` (or your venv).
2. Start Jupyter from the project root:
   ```bash
   jupyter lab
   ```
   or open `Grammar_recognition_pipeline.ipynb` in VS Code and select the **Python (dlenv)** kernel.
3. Decide on the [run switches](#6-run-switches) (Step 1.2 and Step 11.1).
4. Run all cells top to bottom (**Run → Run All Cells**). The notebook is designed to run in order;
   Part 2 uses tables created in Part 1.

### Expected run times

| Mode | Setting | Time |
|---|---|---|
| **Fast** (use cached results) | `RECOMPUTE_AUDIO_FEATURES = False`, `RUN_WHISPER = False` | a few minutes, no GPU needed |
| Recompute audio features | `RECOMPUTE_AUDIO_FEATURES = True` | + ~4 minutes |
| Full Whisper run | `RUN_WHISPER = True` and delete `outputs/asr/asr_full_small_prompt.jsonl` | + ~2.5 hours on an RTX 4060 |

Whisper runs are **resumable**: results are checkpointed every 25 clips, and clips already saved in `outputs/asr/`
are never transcribed again. If a run is interrupted, re-run the cell and it continues where it stopped.

### Notebook structure

**Part 1 – Data and EDA**
- Step 1 – Setup and configuration (imports, seed, Kaggle/local path detection)
- Step 2 – Load the CSV files and integrity checks
- Step 3 – Label analysis
- Step 4 – Fast metadata scan (file headers only)
- Step 5 – Signal-level feature table (cached to `outputs/audio_features.csv`)
- Step 6 – Noise versus speech on the train set (ROC AUC ranking)
- Step 7 – Visual inspection (waveforms and mel spectrograms)
- Step 8 – Listen (embedded audio players – **lower your volume**, the noise clips are loud)
- Step 9 – The reusable `load_and_preprocess` function
- Step 10 – Test set sanity check

**Part 2 – Transcription and noise gate**
- Step 11 – Run switches, imports and device
- Step 12 – Confirm the tables from Part 1
- Step 13 – Whisper pilot on 24 training clips
- Step 14 – Compare Whisper settings (with / without a disfluent prompt)
- Step 15 – Full transcription of all 985 clips
- Step 16 – Transcript quality check and prompt-artifact removal
- Step 17 – ASR statistics per clip
- Step 18 – Noise versus speech through the ASR lens
- Step 19 – Noise gate model with honest cross validation
- Step 20 – Test set preview of the noise gate

Each step has a **Goal / How / Terms** cell before the code and a **What to look for** cell after it.

---

## 6. Run switches

| Switch | Where | Default | What it does |
|---|---|---|---|
| `RECOMPUTE_AUDIO_FEATURES` | Step 1.2 | `False` | `True` recomputes the Part 1 feature table even if `outputs/audio_features.csv` exists. |
| `RUN_WHISPER` | Step 11.1 | `True` | `True` transcribes clips that have no saved transcript yet. `False` only loads saved results and never starts Whisper. |
| `PILOT_MEDIUM` | Step 11.1 | `False` | `True` adds the Whisper `medium` model to the Step 14 comparison (~1.5 GB download, needs ≥ 6 GB VRAM and plenty of RAM). |
| `CHECKPOINT_EVERY` | Step 11.1 | `25` | Number of clips transcribed between two saves to disk. |
| `SEED` | Step 1.2 | `42` | Global random seed for reproducibility. |

> Since `outputs/asr/` is already committed, `RUN_WHISPER = True` will simply load the saved transcripts and transcribe nothing.
> It only does real work if a clip is missing from the saved files.

---

## 7. Running on Kaggle

The notebook detects Kaggle automatically (it checks for `/kaggle/input`) and adapts its paths:

- `DATA_DIR` = the first folder under `/kaggle/input` that contains `train.csv` and a `train/` folder.
- `OUTPUT_DIR` = `/kaggle/working`.

Steps:

1. Create a new Kaggle notebook and upload / import `Grammar_recognition_pipeline.ipynb`.
2. **Add data**: upload the dataset from the Google Drive link as a Kaggle dataset (with `train.csv`, `test.csv`, `train/`, `test/`) and attach it.
3. *(Optional, recommended)* Upload the `outputs/asr/` folder from this repo as a Kaggle dataset and attach it.
   The notebook copies any attached `asr` folder with `.jsonl` files into `/kaggle/working/asr` automatically.
   Then set `RUN_WHISPER = False` – no GPU needed.
4. *(Optional)* To transcribe on Kaggle instead, enable a **GPU** accelerator and **Internet** (for the weights download),
   or attach a dataset containing `small.pt` – the notebook searches `/kaggle/input` for it.
   Kaggle's preinstalled PyTorch works with `openai-whisper`; you may need `!pip install openai-whisper==20250625` first.
5. Run all cells.

---

## 8. Generated files (`outputs/`)

| File | Created in | Contents |
|---|---|---|
| `audio_features.csv` | Step 5 | One row per clip (key `split, filename`): rms, rms_dbfs, peak, clip_fraction, zcr, spectral centroid / bandwidth / flatness, speech_ratio, loaded duration. |
| `asr/pilot_small_noprompt.jsonl` | Step 13 | Whisper `small`, no prompt, on 24 pilot clips. |
| `asr/pilot_small_prompt.jsonl` | Step 14 | Whisper `small` with the disfluent prompt, same 24 clips. |
| `asr/asr_full_small_prompt.jsonl` | Step 15 | Full transcription of all 985 clips: text, segments (no-speech prob, avg log-prob, compression ratio), word timestamps and word probabilities, runtime, GPU peak. One JSON object per line. |
| `asr/transcripts.csv` | Step 15/16 | Flat table of the (cleaned) transcripts. |
| `asr_stats.csv` | Step 17 | Per-clip ASR statistics: n_words, words_per_sec, mean/min word prob, mean no-speech prob, mean avg log-prob, max compression ratio, trigram repeat, filler and repeat rates. |
| `noise_gate_predictions.csv` | Step 20 | Noise probability for every train clip (out-of-fold) and test clip. |

All tables are keyed by **`(split, filename)`**.

---

## 9. Key results so far

**Data**
- 769 train clips (732 speech + 37 noise clips with label 0) and 216 test clips; all 16 kHz mono PCM_16.
- Speech labels: mean 3.48, std 1.01. Whole-number scores dominate; only 4 clips are 1.0 or 1.5.
- **Baseline to beat:** always predicting the mean gives **RMSE 1.014** on the speech clips (Pearson 0).

**Noise clips**
- Three acoustic features (zero-crossing rate, spectral centroid, spectral flatness) separate noise from speech perfectly on train (AUC 1.0).
- Whisper still recovers coherent speech from all 37 noise clips, so label 0 means "speech made unusable by added noise".

**Whisper**
- Chosen setting: `small` model + a disfluent `initial_prompt`, which keeps hesitations ("um", "uh") and repetitions that matter for grammar scoring.
- 985/985 clips transcribed with 0 errors. Prompt-copy artifacts were removed from 9 clips in Step 16.4.

**Noise gate**
- Standardized logistic regression on 6 features (5 acoustic + `mean_word_prob`).
- 5-fold stratified CV: ROC AUC 1.000, precision 1.000, recall 1.000, 0 misclassified clips (threshold 0.52).
- Test preview: **0 of 216** test clips flagged as noise (highest probability 0.026).

---

## 10. Troubleshooting

| Problem | Fix |
|---|---|
| `ModuleNotFoundError` in Step 1.1 | The wrong kernel is selected, or Part 1 requirements are not installed. Select **Python (dlenv)** and re-run `pip install -r requirements.txt`. |
| `FileNotFoundError` for `train.csv` or a `.wav` file | The dataset is not in the project root, or it is nested one folder too deep after unzipping. See [Getting the dataset](#4-getting-the-dataset). |
| Step 5 final check prints `False` | The cached `audio_features.csv` is from a different dataset. Set `RECOMPUTE_AUDIO_FEATURES = True` and re-run. |
| `openai-whisper ... NOT INSTALLED` in Step 11.2 | Run `pip install -r requirements.txt`, or set `RUN_WHISPER = False` to use the saved transcripts. |
| `device cpu` although you have an NVIDIA GPU | You installed the CPU build of PyTorch. Uninstall it (`pip uninstall torch`) and reinstall the CUDA build (step 3.3). |
| `CUDA out of memory` | Close other GPU programs, keep `PILOT_MEDIUM = False`, or run with `RUN_WHISPER = False`. |
| Warning `Failed to launch Triton kernels` | Harmless on Windows; Whisper falls back to a slower path for word timestamps. The notebook hides it. |
| Whisper run was interrupted | Just re-run the cell. Finished clips are skipped; only missing or failed clips are transcribed. |
| `faster-whisper` instead of `openai-whisper`? | Not used: its GPU engine needs CUDA 12 libraries, which did not match the CUDA 13.2 PyTorch build. |
| Notebook is slow to open in git diffs | The notebook stores cell outputs (plots, audio players). That is expected; it is ~5 MB. |
| Numbers differ from the summaries | Compare library versions printed in Step 1.3 with the pinned versions in the requirements files, especially `librosa` and `numpy`. |

---

## 11. What is and is not in git

**Tracked**
- The notebook, requirements files, this README and `.gitignore`.
- `outputs/` – cached features, transcripts and gate predictions, so the notebook can run without a GPU.

**Ignored** (see [.gitignore](.gitignore))
- The whole dataset: `train/`, `test/`, `train.csv`, `test.csv`, `sample_submission.csv`, and all audio files
  (`*.wav`, `*.mp3`, ...). Download it from the [Google Drive link](#4-getting-the-dataset).
- Model weights (`*.pt`, `*.pth`, `*.safetensors`, ...) and local caches.
- Virtual environments, `__pycache__/`, `.ipynb_checkpoints/`.
- Secrets (`.env`, `kaggle.json`), IDE folders and OS files.
- Generated submission files (`submission*.csv`).
