# Music Restoration Studio

Music Restoration Studio is a desktop app for restoring and enhancing songs from YouTube or local files. It is designed for old recordings, noisy transfers, dull masters, and compressed audio that need cleanup without losing the original source.

The app keeps the untouched source, creates a lossless working master for processing, and lets you compare the original and the latest enhanced version side by side.

## Highlights

- Download audio from YouTube or import a local audio file
- Preserve the original source for direct A/B playback
- Analyze the source and suggest a restoration plan
- Apply restoration and enhancement in versioned passes
- Undo to earlier enhanced versions
- Export the current result as `mp3`, `wav`, or `flac`

## Processing Profiles

- `Restore Old Song`
  Focused on hiss, hum, clicks, rumble, dullness, and uneven old transfers.

- `Enhance Song`
  For tracks that are mostly clean and benefit more from clarity, balance, and mastering polish.

- `Repair Compressed Audio`
  For bandwidth-limited or harsh compressed material, with optional bandwidth restoration.

- `Stem Rebalance`
  Uses separation and recombination to rebalance vocals and accompaniment.

- `Advanced / Experimental`
  Exposes the slower or more specialized options, including AudioSR and legacy fallbacks.

## Built-In Workflow

1. Paste a YouTube URL or import a local file.
2. Play the original source.
3. Run `Analyze Source` for guidance.
4. Adjust the processing plan or use the recommendation.
5. Click `Apply Plan`.
6. Compare original and enhanced playback.
7. Export the current version when you are happy with it.

## Main Features

- Compact Tkinter desktop interface
- Browser-session mode for tougher YouTube downloads
- Lossless working master for all enhancement passes
- Versioned enhancement history with undo
- RoFormer-family stem rebalance through `audio-separator`
- Optional learned source analysis with local PANNs assets
- Experimental AudioSR bandwidth restoration path
- Bundled core local models for stem rebalance, dereverb, and source analysis

## Project Layout

- [restoration_studio_tk.py](</d:/Python codes/youtube downloader/restoration_studio_tk.py>)  
  Desktop app entrypoint and Tkinter UI.

- [restoration_backend.py](</d:/Python codes/youtube downloader/restoration_backend.py>)  
  Download, analysis, restoration, enhancement, export, and session management backend.

- [test_restoration_backend.py](</d:/Python codes/youtube downloader/test_restoration_backend.py>)  
  Backend smoke tests and model-resolution checks.
- [docs/ui-screenshot.png](</d:/Python codes/youtube downloader/docs/ui-screenshot.png>)  
  UI screenshot used for documentation.

## Installation

Create the main environment and install the core dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Install the optional AudioSR environment if you want bandwidth restoration:

```powershell
python -m venv .venv_audiosr
.\.venv_audiosr\Scripts\python.exe -m pip install --upgrade pip
.\.venv_audiosr\Scripts\python.exe -m pip install -r requirements-audiosr.txt
```

## Run

Launch the desktop app:

```powershell
.\.venv\Scripts\python.exe restoration_studio_tk.py
```

## Optional Models and Tools

The app can make use of locally installed extras when available:

- `audio-separator` for RoFormer-family stem rebalance
- `onnxruntime` for lighter model paths
- local PANNs assets in `models/panns-home/panns_data/` for learned source analysis
- AudioSR inside `.venv_audiosr` for experimental bandwidth restoration

The repository includes the core local model assets used by the main restoration flow. AudioSR model caches remain local and are not committed because they are much larger generated assets.

## Tests

Run the quick backend suite:

```powershell
.\.venv\Scripts\python.exe test_restoration_backend.py --suite quick
```

Run the model-resolution checks:

```powershell
.\.venv\Scripts\python.exe test_restoration_backend.py --suite models
```

## Notes

- Generated sessions, caches, exports, and virtual environments should stay out of version control.
- The app keeps the original downloaded or imported source intact.
- Each enhancement pass creates a new version in the session history.
- AudioSR is experimental and can change timbre.
- Large model assets are intended to live locally under `models/`.
