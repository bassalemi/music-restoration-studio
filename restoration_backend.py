import contextlib
import copy
import csv
import glob
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import traceback
import urllib.request
from pathlib import Path

import numpy as np
import scipy.signal
import yt_dlp


class RestorationStudioBackend:
    RESTORATION_PRESET_LABELS = {
        "light": "Light Restore",
        "medium": "Medium Restore",
        "heavy": "Heavy Restore",
        "archive": "Archive Transfer",
    }
    RESTORATION_PRESET_ORDER = ["light", "medium", "heavy", "archive"]
    HUM_FREQUENCIES = [50, 60]
    PROFILE_LABELS = {
        "restore": "Restore Old Song",
        "enhance": "Enhance Song",
        "compression": "Repair Compressed Audio",
        "stem": "Stem Rebalance",
        "advanced": "Advanced / Experimental",
    }
    PROFILE_DEFAULTS = {
        "restore": {
            "restoration_cleanup": True,
            "clarity_mastering": True,
            "normalize_audio": True,
            "stem_rebalance": False,
            "bandwidth_restore": False,
            "ai_dereverb": False,
            "restoration_preset": "medium",
            "hum_frequency": 60,
            "backend": "roformer",
            "ai_model_preset": "balanced",
            "bass_boost": 0,
            "treble_boost": 0,
            "volume_boost": 0,
        },
        "enhance": {
            "restoration_cleanup": False,
            "clarity_mastering": True,
            "normalize_audio": True,
            "stem_rebalance": False,
            "bandwidth_restore": False,
            "ai_dereverb": False,
            "restoration_preset": "light",
            "hum_frequency": 60,
            "backend": "roformer",
            "ai_model_preset": "balanced",
            "bass_boost": 1,
            "treble_boost": 1,
            "volume_boost": 0,
        },
        "compression": {
            "restoration_cleanup": False,
            "clarity_mastering": True,
            "normalize_audio": True,
            "stem_rebalance": False,
            "bandwidth_restore": True,
            "ai_dereverb": False,
            "restoration_preset": "light",
            "hum_frequency": 60,
            "backend": "roformer",
            "ai_model_preset": "low_vram",
            "bass_boost": 1,
            "treble_boost": 1,
            "volume_boost": 0,
        },
        "stem": {
            "restoration_cleanup": False,
            "clarity_mastering": False,
            "normalize_audio": False,
            "stem_rebalance": True,
            "bandwidth_restore": False,
            "ai_dereverb": False,
            "restoration_preset": "light",
            "hum_frequency": 60,
            "backend": "roformer",
            "ai_model_preset": "quality",
            "bass_boost": 0,
            "treble_boost": 0,
            "volume_boost": 0,
        },
        "advanced": {
            "restoration_cleanup": False,
            "clarity_mastering": True,
            "normalize_audio": True,
            "stem_rebalance": False,
            "bandwidth_restore": False,
            "ai_dereverb": False,
            "restoration_preset": "medium",
            "hum_frequency": 60,
            "backend": "roformer",
            "ai_model_preset": "balanced",
            "bass_boost": 0,
            "treble_boost": 0,
            "volume_boost": 0,
        },
    }
    BROWSER_OPTIONS = ["chrome", "edge", "firefox"]
    AI_MODEL_PRESET_LABELS = {
        "quality": "Highest Quality",
        "balanced": "Balanced",
        "low_vram": "Low VRAM",
    }
    AI_MODEL_PRESET_ORDER = ["quality", "balanced", "low_vram"]
    AI_MODEL_PRESETS = {
        "quality": {
            "description": "6-stem BS-RoFormer-SW",
            "preferred_models": ["BS-Roformer-SW.ckpt"],
            "arch": "mdxc",
            "sample_rate": 48000,
            "extra_args": ["--mdxc_segment_size", "256", "--mdxc_overlap", "8", "--mdxc_batch_size", "1"],
        },
        "balanced": {
            "description": "2-stem BS-RoFormer Viperx 12.9755",
            "preferred_models": ["model_bs_roformer_ep_317_sdr_12.9755.ckpt", "BS-Roformer-SW.ckpt"],
            "arch": "mdxc",
            "sample_rate": 48000,
            "extra_args": ["--mdxc_segment_size", "256", "--mdxc_overlap", "8", "--mdxc_batch_size", "1"],
        },
        "low_vram": {
            "description": "UVR MDX KARA 2 ONNX",
            "preferred_models": ["UVR_MDXNET_KARA_2.onnx", "Kim_Vocal_2.onnx"],
            "arch": "mdx",
            "sample_rate": 44100,
            "extra_args": ["--mdx_segment_size", "128", "--mdx_overlap", "0.25", "--mdx_batch_size", "1"],
        },
    }
    DEFAULT_ROFORMER_PREFERENCES = [
        "BS-Roformer-SW.ckpt",
        "model_bs_roformer_ep_317_sdr_12.9755.ckpt",
    ]
    PANN_LABELS_URL = "https://storage.googleapis.com/us_audioset/youtube_corpus/v1/csv/class_labels_indices.csv"
    PANN_CHECKPOINT_URL = "https://zenodo.org/record/3987831/files/Cnn14_mAP%3D0.431.pth?download=1"

    def __init__(self, root_dir: Path):
        self.root_dir = root_dir
        self.sessions_dir = self.root_dir / "sessions"
        self.sessions_dir.mkdir(parents=True, exist_ok=True)
        self.audio_separator_model_dir = self.root_dir / "models" / "audio-separator"
        self.audio_separator_model_dir.mkdir(parents=True, exist_ok=True)
        self.hf_cache_dir = self.root_dir / "models" / "huggingface"
        self.hf_cache_dir.mkdir(parents=True, exist_ok=True)
        self.panns_home_dir = self.root_dir / "models" / "panns-home"
        self.panns_data_dir = self.panns_home_dir / "panns_data"
        self.panns_data_dir.mkdir(parents=True, exist_ok=True)
        self.temp_runtime_dir = self.root_dir / ".tmp_runtime"
        self.temp_runtime_dir.mkdir(parents=True, exist_ok=True)

        self.ffmpeg_location = self.find_ffmpeg()
        self.lock = threading.RLock()
        self.processing_thread = None

        self.processing = False
        self.progress = 0.0
        self.status_text = "Ready to download or import"
        self.status_color = "slate"
        self.last_error = None
        self.last_details = ""

        self.profile = "restore"
        self.download_mode = "direct"
        self.browser_source = "chrome"
        self.max_undo_steps = 5
        self.export_format = "mp3"

        self.current_session_dir = None
        self.source_file = None
        self.current_file = None
        self.downloaded_files = []
        self.enhancement_history = []
        self.enhancement_version = 0
        self.media_revision = 0
        self.last_resolved_roformer_model = "Not resolved yet"
        self.last_analysis = None
        self.analysis_engine = "hybrid"
        self.enable_learned_analysis = True
        # Reserved provider slot for future Apollo integration.
        self.compression_restore_provider = "audiosr"

    def snapshot(self):
        with self.lock:
            has_original = bool(self.source_file and Path(self.source_file).exists())
            has_enhanced = bool(
                self.current_file
                and Path(self.current_file).exists()
                and not self._same_file(self.current_file, self.source_file)
            )
            return {
                "processing": self.processing,
                "progress": round(self.progress, 1),
                "status_text": self.status_text,
                "status_color": self.status_color,
                "last_error": self.last_error,
                "last_details": self.last_details,
                "profile": self.profile,
                "download_mode": self.download_mode,
                "browser_source": self.browser_source,
                "has_original": has_original,
                "has_enhanced": has_enhanced,
                "original_name": Path(self.source_file).name if has_original else "No original source yet",
                "enhanced_name": Path(self.current_file).name if has_enhanced else "No enhanced version yet",
                "original_url": f"/media/original?rev={self.media_revision}" if has_original else None,
                "enhanced_url": f"/media/enhanced?rev={self.media_revision}" if has_enhanced else None,
                "can_undo": bool(self.enhancement_history) and not self.processing,
                "can_revert": has_enhanced and not self.processing,
                "resolved_roformer_model": self.last_resolved_roformer_model,
                "downloaded_files": list(self.downloaded_files),
                "current_export_format": self.export_format,
                "session_dir": str(self.current_session_dir) if self.current_session_dir else None,
                "analysis_ready": bool(self.last_analysis),
                "analysis_summary": self.last_analysis.get("summary", "No analysis yet.") if self.last_analysis else "No analysis yet.",
                "analysis_recommendation": self.last_analysis.get("recommendation_label", "Analyze source to get guidance.") if self.last_analysis else "Analyze source to get guidance.",
                "analysis_model_strategy": self.last_analysis.get("ai_model_label", "Balanced") if self.last_analysis else "Balanced",
                "analysis_engine": self.last_analysis.get("analysis_engine_label", "Hybrid (PANNs + DSP)") if self.last_analysis else "Hybrid (PANNs + DSP)",
            }

    def _set_status(self, text, *, progress=None, color=None, error=None, details=None):
        with self.lock:
            self.status_text = text
            if progress is not None:
                self.progress = max(0.0, min(100.0, float(progress)))
            if color is not None:
                self.status_color = color
            if error is not None:
                self.last_error = error
            if details is not None:
                self.last_details = details

    def _bump_media_revision(self):
        with self.lock:
            self.media_revision += 1

    def _same_file(self, first, second):
        if not first or not second:
            return False
        return os.path.abspath(first) == os.path.abspath(second)

    def _undo_limit(self):
        return max(0, min(20, int(self.max_undo_steps)))

    def _remember_undo_state(self, path):
        if not path:
            return
        if self.enhancement_history and self._same_file(self.enhancement_history[-1], path):
            return
        self.enhancement_history.append(path)
        limit = self._undo_limit()
        if limit == 0:
            self.enhancement_history.clear()
        elif len(self.enhancement_history) > limit:
            self.enhancement_history = self.enhancement_history[-limit:]

    def undo_last(self):
        with self.lock:
            if self.processing:
                raise RuntimeError("Processing is still running.")
            if not self.enhancement_history:
                raise RuntimeError("No enhancement history to undo.")
            previous = self.enhancement_history.pop()
            self.current_file = previous
            self._bump_media_revision()
            label = "original source" if self._same_file(previous, self.source_file) else "previous enhancement"
            self._set_status(f"Reverted to {label}.", progress=0, color="slate", error=None, details="")

    def revert_to_original(self):
        with self.lock:
            if self.processing:
                raise RuntimeError("Processing is still running.")
            if not self.source_file or not os.path.exists(self.source_file):
                raise RuntimeError("No original source to revert to.")
            self.current_file = self.source_file
            self.enhancement_history.clear()
            self._bump_media_revision()
            self._set_status("Returned to the original source.", progress=0, color="slate", error=None, details="")

    def _new_session_dir(self):
        stamp = time.strftime("%Y%m%d_%H%M%S")
        session_dir = self.sessions_dir / stamp
        counter = 1
        while session_dir.exists():
            counter += 1
            session_dir = self.sessions_dir / f"{stamp}_{counter}"
        (session_dir / "source").mkdir(parents=True, exist_ok=True)
        (session_dir / "versions").mkdir(parents=True, exist_ok=True)
        (session_dir / "exports").mkdir(parents=True, exist_ok=True)
        return session_dir

    def _start_fresh_session(self, source_path: Path):
        session_dir = self._new_session_dir()
        target = session_dir / "source" / source_path.name
        shutil.copy2(source_path, target)
        with self.lock:
            self.current_session_dir = session_dir
            self.source_file = str(target)
            self.current_file = str(target)
            self.downloaded_files = [str(target)]
            self.enhancement_history = []
            self.enhancement_version = 0
            self.last_analysis = None
            self.last_error = None
            self.last_details = ""
            self.progress = 0.0
            self.media_revision += 1
        return target

    def _next_enhanced_filename(self):
        if not self.current_session_dir:
            raise RuntimeError("No active session.")
        self.enhancement_version += 1
        return self.current_session_dir / "versions" / f"enhanced_{self.enhancement_version:03d}.wav"

    def update_ai_progress(self, stage: str, percent: float, color: str = "blue"):
        self._set_status(stage or self.status_text, progress=percent, color=color)

    def progress_hook(self, update):
        status = update.get("status")
        if status == "downloading":
            total = update.get("total_bytes") or update.get("total_bytes_estimate") or 0
            downloaded = update.get("downloaded_bytes") or 0
            if total:
                percent = (downloaded / total) * 100.0
                self._set_status(f"Downloading source audio... {percent:.1f}%", progress=percent, color="blue")
            else:
                self._set_status("Downloading source audio...", progress=10, color="blue")
        elif status == "finished":
            self._set_status("Download finished. Preparing source file...", progress=100, color="blue")

    def _friendly_download_error(self, exc: Exception, mode: str):
        text = str(exc)
        lowered = text.lower()
        if "403" in lowered or "forbidden" in lowered or "sign in to confirm" in lowered:
            if mode == "direct":
                return (
                    "YouTube blocked the direct download request. Try 'Use browser session' "
                    "with Chrome, Edge, or Firefox so yt-dlp can reuse your signed-in browser session."
                )
            return (
                "YouTube still blocked this request. Switch to a different browser session "
                "or refresh your browser login and try again."
            )
        return f"Download failed: {text}"

    def _collect_downloaded_files(self, ydl, info, output_dir: Path):
        entries = info.get("entries") if isinstance(info, dict) and info.get("entries") else [info]
        files = []
        for entry in entries:
            if not entry:
                continue
            candidates = []
            for item in entry.get("requested_downloads") or []:
                filepath = item.get("filepath")
                if filepath:
                    candidates.append(filepath)
            prepared = ydl.prepare_filename(entry)
            if prepared:
                candidates.append(prepared)
                base, _ = os.path.splitext(prepared)
                candidates.extend(glob.glob(f"{base}.*"))
            for candidate in candidates:
                if candidate and os.path.exists(candidate) and not candidate.endswith(".part"):
                    files.append(os.path.abspath(candidate))
                    break
        if not files:
            audio_files = []
            for pattern in ("*.m4a", "*.webm", "*.mp3", "*.wav", "*.flac", "*.opus"):
                audio_files.extend(output_dir.glob(pattern))
            files = [str(path.resolve()) for path in sorted(audio_files, key=lambda item: item.stat().st_mtime)]
        return files

    def download_audio(self, url: str, download_mode: str, browser_source: str):
        if not url:
            raise RuntimeError("Enter a YouTube URL first.")

        session_dir = self._new_session_dir()
        source_dir = session_dir / "source"
        ydl_opts = {
            "format": "bestaudio/best",
            "outtmpl": str(source_dir / "%(title)s.%(ext)s"),
            "progress_hooks": [self.progress_hook],
            "quiet": True,
            "no_warnings": True,
        }
        if self.ffmpeg_location and self.ffmpeg_location != "":
            ydl_opts["ffmpeg_location"] = self.ffmpeg_location
        if download_mode == "browser":
            ydl_opts["cookiesfrombrowser"] = (browser_source.lower(),)

        try:
            self._set_status("Fetching video info...", progress=2, color="blue", error=None, details="")
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=True)
                files = self._collect_downloaded_files(ydl, info, source_dir)
        except Exception as exc:
            raise RuntimeError(self._friendly_download_error(exc, download_mode)) from exc

        if not files:
            raise RuntimeError("yt-dlp finished, but no source audio file was found.")

        first_file = Path(files[0])
        with self.lock:
            self.current_session_dir = session_dir
            self.source_file = str(first_file)
            self.current_file = str(first_file)
            self.downloaded_files = files
            self.enhancement_history = []
            self.enhancement_version = 0
            self.download_mode = download_mode
            self.browser_source = browser_source
            self.last_error = None
            self.last_details = ""
            self.progress = 0.0
            self.media_revision += 1
        self._set_status("Original source is ready to play.", progress=0, color="green", error=None, details="")

    def import_audio_from_path(self, temp_path):
        temp_path = Path(temp_path)
        if not temp_path.exists():
            raise RuntimeError("The uploaded file was not saved correctly.")
        try:
            copied = self._start_fresh_session(temp_path)
        finally:
            shutil.rmtree(temp_path.parent, ignore_errors=True)
        self._set_status(f"Imported {copied.name}.", progress=0, color="green", error=None, details="")

    def find_ffmpeg(self):
        candidates = []
        found = shutil.which("ffmpeg")
        if found:
            return ""
        program_files = [
            Path(os.environ.get("ProgramFiles", "C:/Program Files")),
            Path(os.environ.get("ProgramFiles(x86)", "C:/Program Files (x86)")),
        ]
        for base in program_files:
            candidates.extend(base.glob("ffmpeg*/bin"))
            candidates.extend(base.glob("FFmpeg*/bin"))
        candidates.append(self.root_dir / "ffmpeg" / "bin")
        for candidate in candidates:
            exe = candidate / "ffmpeg.exe"
            if exe.exists():
                return str(candidate)
        return None

    def _ffmpeg_executable(self):
        if not self.ffmpeg_location or self.ffmpeg_location == "":
            return "ffmpeg"
        return os.path.join(self.ffmpeg_location, "ffmpeg.exe")

    def _audio_encode_args(self, output_path):
        ext = os.path.splitext(output_path)[1].lower()
        if ext == ".mp3":
            return ["-c:a", "libmp3lame", "-b:a", "320k"]
        if ext == ".flac":
            return ["-c:a", "flac"]
        return ["-c:a", "pcm_s16le"]

    def _convert_to_work_wav(self, input_path, output_path):
        cmd = [
            self._ffmpeg_executable(), "-y",
            "-i", input_path,
            "-vn",
            "-c:a", "pcm_s16le",
            output_path,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0 or not os.path.exists(output_path):
            detail = result.stderr[-1000:] if result.stderr else "Unknown FFmpeg error"
            raise RuntimeError(f"Failed to create working WAV:\n{detail}")

    def _encode_output(self, input_path, output_path):
        cmd = [
            self._ffmpeg_executable(), "-y",
            "-i", input_path,
            "-vn",
            *self._audio_encode_args(output_path),
            output_path,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0 or not os.path.exists(output_path):
            detail = result.stderr[-1000:] if result.stderr else "Unknown FFmpeg error"
            raise RuntimeError(f"Failed to export audio:\n{detail}")

    def export_current(self, fmt: str):
        if not self.current_file or not os.path.exists(self.current_file):
            raise RuntimeError("No active audio to export.")
        fmt = (fmt or "mp3").lower()
        if fmt not in {"mp3", "wav", "flac"}:
            raise RuntimeError("Unsupported export format.")
        if not self.current_session_dir:
            raise RuntimeError("No session is active.")
        export_dir = self.current_session_dir / "exports"
        export_dir.mkdir(parents=True, exist_ok=True)
        base_name = Path(self.current_file).stem
        export_path = export_dir / f"{base_name}.{fmt}"
        if fmt == Path(self.current_file).suffix.lower().lstrip("."):
            shutil.copy2(self.current_file, export_path)
        else:
            self._encode_output(self.current_file, str(export_path))
        self.export_format = fmt
        return export_path

    @staticmethod
    def _replace_file_with_retry(source, target, attempts=10, delay=0.2):
        last_error = None
        for _ in range(attempts):
            try:
                os.replace(source, target)
                return
            except PermissionError as exc:
                last_error = exc
                try:
                    if os.path.exists(target):
                        os.remove(target)
                    os.replace(source, target)
                    return
                except PermissionError as remove_exc:
                    last_error = remove_exc
                    try:
                        shutil.copyfile(source, target)
                        return
                    except PermissionError as copy_exc:
                        last_error = copy_exc
                time.sleep(delay)
        raise last_error

    def _build_optional_command_env(self):
        env = os.environ.copy()
        for key in ("PIP_NO_INDEX", "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "GIT_HTTP_PROXY", "GIT_HTTPS_PROXY"):
            env.pop(key, None)
            env.pop(key.lower(), None)
        env["HF_HOME"] = str(self.hf_cache_dir)
        env["TRANSFORMERS_CACHE"] = str(self.hf_cache_dir)
        env["AUDIO_SEPARATOR_MODEL_DIR"] = str(self.audio_separator_model_dir)
        env["TMP"] = str(self.temp_runtime_dir)
        env["TEMP"] = str(self.temp_runtime_dir)
        env["TMPDIR"] = str(self.temp_runtime_dir)

        nvidia_paths = [
            r"C:\Program Files\NVIDIA\CUDNN\v9.15\bin\12.9",
            r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin",
        ]
        existing = [path for path in nvidia_paths if os.path.isdir(path)]
        if existing:
            env["PATH"] = os.pathsep.join(existing + [env.get("PATH", "")])
        return env

    def _find_optional_executable(self, executable, extra_venvs=None):
        if os.path.exists(executable):
            return executable
        found = shutil.which(executable)
        if found:
            return found
        suffixes = [f"{executable}.exe", f"{executable}.cmd", executable]
        venvs = [".venv"]
        venvs.extend(extra_venvs or [])
        for venv_name in venvs:
            scripts_dir = self.root_dir / venv_name / "Scripts"
            for suffix in suffixes:
                candidate = scripts_dir / suffix
                if candidate.exists():
                    return str(candidate)
        return None

    def _run_optional_command(self, command, missing_message, extra_venvs=None, env=None):
        executable = self._find_optional_executable(command[0], extra_venvs=extra_venvs)
        if executable is None:
            raise RuntimeError(missing_message)
        full_command = [executable, *command[1:]]
        runtime_env = self._build_optional_command_env()
        if env:
            runtime_env.update(env)
        result = subprocess.run(full_command, capture_output=True, text=True, env=runtime_env)
        if result.returncode != 0:
            detail = result.stderr[-1200:] if result.stderr else result.stdout[-1200:]
            raise RuntimeError(f"{Path(executable).name} failed:\n{detail}")
        return result

    def _find_newest_audio_file(self, folder, exclude_paths=None):
        exclude = {os.path.abspath(path) for path in (exclude_paths or [])}
        candidates = []
        for pattern in ("*.wav", "*.flac", "*.mp3"):
            candidates.extend(glob.glob(os.path.join(folder, pattern)))
        candidates = [path for path in candidates if os.path.abspath(path) not in exclude]
        if not candidates:
            return None
        return max(candidates, key=os.path.getmtime)

    def _estimate_audio_duration_seconds(self, file_path):
        try:
            import soundfile as sf

            info = sf.info(file_path)
            return float(info.duration)
        except Exception:
            return None

    @contextlib.contextmanager
    def _panns_home_env(self):
        keys = ("HOME", "USERPROFILE", "HOMEDRIVE", "HOMEPATH")
        previous = {key: os.environ.get(key) for key in keys}
        try:
            os.environ["HOME"] = str(self.panns_home_dir)
            os.environ["USERPROFILE"] = str(self.panns_home_dir)
            drive, tail = os.path.splitdrive(str(self.panns_home_dir))
            if drive:
                os.environ["HOMEDRIVE"] = drive
                os.environ["HOMEPATH"] = tail or "\\"
            yield
        finally:
            for key, value in previous.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value

    def _download_if_missing(self, url, destination):
        destination = Path(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists() and destination.stat().st_size > 0:
            return destination
        urllib.request.urlretrieve(url, str(destination))
        if not destination.exists() or destination.stat().st_size == 0:
            raise RuntimeError(f"Download failed for {destination.name}")
        return destination

    def _panns_assets_ready(self):
        labels_csv = self.panns_data_dir / "class_labels_indices.csv"
        checkpoint = self.panns_data_dir / "Cnn14_mAP=0.431.pth"
        return labels_csv.exists() and labels_csv.stat().st_size > 0 and checkpoint.exists() and checkpoint.stat().st_size > 250_000_000

    def _ensure_panns_assets(self):
        labels_csv = self.panns_data_dir / "class_labels_indices.csv"
        checkpoint = self.panns_data_dir / "Cnn14_mAP=0.431.pth"
        if not labels_csv.exists() or labels_csv.stat().st_size == 0:
            raise RuntimeError("Bundled PANNs labels are missing from models/panns-home/panns_data.")
        if not checkpoint.exists() or checkpoint.stat().st_size <= 250_000_000:
            raise RuntimeError("Bundled PANNs checkpoint is missing or incomplete in models/panns-home/panns_data.")
        return labels_csv, checkpoint

    def prepare_learned_analysis_assets(self):
        self._set_status("Checking bundled learned analysis model...", progress=10, color="blue", error=None, details="")
        self._ensure_panns_assets()
        self._set_status("Bundled learned analysis model is ready.", progress=100, color="green", error=None, details="")

    def _panns_available(self):
        import importlib.util

        return importlib.util.find_spec("panns_inference") is not None

    def _load_panns_labels(self, labels_csv):
        labels = []
        with open(labels_csv, "r", newline="", encoding="utf-8") as handle:
            reader = csv.reader(handle, delimiter=",")
            next(reader, None)
            for row in reader:
                if len(row) >= 3:
                    labels.append(row[2])
        return labels

    @staticmethod
    def _safe_db(value, floor=-120.0):
        value = float(value)
        if value <= 1e-12:
            return floor
        return 20.0 * np.log10(value)

    def _gpu_vram_gb(self):
        try:
            import torch

            if not torch.cuda.is_available():
                return 0.0
            props = torch.cuda.get_device_properties(0)
            return round(props.total_memory / (1024 ** 3), 1)
        except Exception:
            return 0.0

    def _read_analysis_probe(self, file_path, target_sr=22050, segment_seconds=12):
        import soundfile as sf

        analysis_path = file_path
        temp_probe = None
        try:
            try:
                handle = sf.SoundFile(analysis_path)
            except Exception:
                temp_probe = tempfile.NamedTemporaryFile(
                    prefix="analysis_probe_",
                    suffix=".wav",
                    dir=str(self.temp_runtime_dir),
                    delete=False,
                )
                temp_probe.close()
                converted = temp_probe.name
                cmd = [
                    self._ffmpeg_executable(), "-y",
                    "-i", file_path,
                    "-ar", str(target_sr),
                    "-ac", "1",
                    "-c:a", "pcm_s16le",
                    converted,
                ]
                result = subprocess.run(cmd, capture_output=True, text=True)
                if result.returncode != 0 or not os.path.exists(converted):
                    detail = result.stderr[-1000:] if result.stderr else "Unknown FFmpeg error"
                    raise RuntimeError(f"Failed to prepare audio for analysis:\n{detail}")
                analysis_path = converted
                handle = sf.SoundFile(analysis_path)

            with handle:
                sample_rate = int(handle.samplerate)
                total_frames = len(handle)
                duration = total_frames / float(sample_rate)
                starts = [0.0]
                if duration > segment_seconds * 2.5:
                    starts.append(max(0.0, (duration * 0.5) - (segment_seconds * 0.5)))
                if duration > segment_seconds * 4:
                    starts.append(max(0.0, duration - segment_seconds))

                segments = []
                unique_starts = []
                for start in starts:
                    key = round(start, 2)
                    if key in unique_starts:
                        continue
                    unique_starts.append(key)
                    handle.seek(int(start * sample_rate))
                    frames = handle.read(int(segment_seconds * sample_rate), dtype="float32", always_2d=True)
                    if frames.size == 0:
                        continue
                    mono = np.mean(frames, axis=1)
                    if sample_rate != target_sr:
                        mono = scipy.signal.resample_poly(mono, target_sr, sample_rate).astype(np.float32)
                    segments.append(mono.astype(np.float32))
        finally:
            if temp_probe and os.path.exists(temp_probe.name):
                try:
                    os.remove(temp_probe.name)
                except OSError:
                    pass

        if not segments:
            raise RuntimeError("Could not read audio for analysis.")
        probe = np.concatenate(segments)
        if probe.size > target_sr * 45:
            probe = probe[: target_sr * 45]
        return {
            "sample_rate": target_sr,
            "duration_seconds": duration,
            "samples": np.clip(probe, -1.0, 1.0),
        }

    def _analyze_signal(self, file_path):
        analysis_input = self._read_analysis_probe(file_path)
        x = analysis_input["samples"]
        sr = analysis_input["sample_rate"]
        if x.ndim != 1 or x.size < sr:
            raise RuntimeError("Not enough audio content to analyze.")

        x = x - np.mean(x)
        abs_x = np.abs(x)
        rms = float(np.sqrt(np.mean(np.square(x)) + 1e-12))
        peak = float(np.max(abs_x) + 1e-12)
        dynamic_range_db = float(np.clip(self._safe_db(np.percentile(abs_x, 95) + 1e-9) - self._safe_db(np.percentile(abs_x, 20) + 1e-9), 0.0, 60.0))

        win = 2048
        hop = 1024
        window = np.hanning(win).astype(np.float32)
        spectra = []
        for start in range(0, max(x.size - win, 1), hop):
            frame = x[start : start + win]
            if frame.size < win:
                break
            spectrum = np.abs(np.fft.rfft(frame * window))
            spectra.append(spectrum)
        if not spectra:
            raise RuntimeError("Failed to compute the analysis spectrum.")
        mag = np.stack(spectra)
        mean_mag = np.mean(mag, axis=0)
        freqs = np.fft.rfftfreq(win, d=1.0 / sr)
        total_energy = float(np.sum(mean_mag) + 1e-9)
        centroid_hz = float(np.sum(freqs * mean_mag) / total_energy)
        cumulative = np.cumsum(mean_mag)
        rolloff_index = int(np.searchsorted(cumulative, cumulative[-1] * 0.85))
        rolloff_hz = float(freqs[min(rolloff_index, freqs.size - 1)])

        high_band_mask = freqs >= min(8000, sr * 0.35)
        mid_band_mask = (freqs >= 1500) & (freqs < min(8000, sr * 0.35))
        high_ratio = float(np.sum(mean_mag[high_band_mask]) / total_energy)
        mid_ratio = float(np.sum(mean_mag[mid_band_mask]) / total_energy)
        high_band = mean_mag[high_band_mask]
        if high_band.size:
            flatness = float(np.exp(np.mean(np.log(high_band + 1e-9))) / (np.mean(high_band) + 1e-9))
        else:
            flatness = 0.0

        def hum_score(base_hz):
            score = 0.0
            for harmonic in range(1, 5):
                target = base_hz * harmonic
                idx = int(np.argmin(np.abs(freqs - target)))
                lo = max(0, idx - 1)
                hi = min(mean_mag.size, idx + 2)
                band = float(np.mean(mean_mag[lo:hi]))
                n_lo = max(0, idx - 8)
                n_hi = min(mean_mag.size, idx + 9)
                neighborhood = np.concatenate([mean_mag[n_lo:lo], mean_mag[hi:n_hi]])
                baseline = float(np.median(neighborhood)) if neighborhood.size else 1e-9
                score += max(0.0, (band / (baseline + 1e-9)) - 1.0)
            return min(score / 8.0, 1.0)

        deriv = np.abs(np.diff(x))
        click_threshold = max(float(np.percentile(deriv, 99.8)), 0.12)
        click_events = int(np.sum(deriv > click_threshold))
        click_density = click_events / max(x.size / sr, 1.0)
        click_score = float(min(click_density / 16.0, 1.0))

        hiss_score = float(np.clip((high_ratio * 5.5) + (flatness * 0.55), 0.0, 1.0))
        dullness_score = float(np.clip(max(0.0, 9000.0 - rolloff_hz) / 5500.0, 0.0, 1.0))
        compression_score = float(np.clip((dullness_score * 0.65) + max(0.0, 12.0 - dynamic_range_db) / 20.0, 0.0, 1.0))

        hum_50 = hum_score(50)
        hum_60 = hum_score(60)
        hum_frequency = 60 if hum_60 >= hum_50 else 50
        hum_strength = max(hum_50, hum_60)

        issues = []
        if hum_strength >= 0.22:
            issues.append(("hum", hum_strength))
        if click_score >= 0.16:
            issues.append(("clicks", click_score))
        if hiss_score >= 0.18:
            issues.append(("hiss", hiss_score))
        if compression_score >= 0.40:
            issues.append(("bandwidth loss", compression_score))
        issues.sort(key=lambda item: item[1], reverse=True)

        return {
            "duration_seconds": round(float(analysis_input["duration_seconds"]), 1),
            "analysis_sample_rate": sr,
            "rms_db": round(self._safe_db(rms), 1),
            "peak_db": round(self._safe_db(peak), 1),
            "dynamic_range_db": round(dynamic_range_db, 1),
            "spectral_centroid_hz": round(centroid_hz, 0),
            "rolloff_hz": round(rolloff_hz, 0),
            "high_ratio": round(high_ratio, 4),
            "mid_ratio": round(mid_ratio, 4),
            "flatness": round(flatness, 4),
            "hum_50_score": round(hum_50, 3),
            "hum_60_score": round(hum_60, 3),
            "hum_frequency": hum_frequency,
            "hum_strength": round(hum_strength, 3),
            "click_score": round(click_score, 3),
            "hiss_score": round(hiss_score, 3),
            "dullness_score": round(dullness_score, 3),
            "compression_score": round(compression_score, 3),
            "issues": issues,
        }

    def _analyze_with_panns(self, file_path):
        labels_csv, checkpoint = self._ensure_panns_assets()
        with self._panns_home_env():
            from panns_inference import AudioTagging

        analysis_input = self._read_analysis_probe(file_path, target_sr=32000, segment_seconds=8)
        audio = analysis_input["samples"].astype(np.float32)[None, :]
        device = "cuda" if self._gpu_vram_gb() > 0 else "cpu"
        with self._panns_home_env():
            tagger = AudioTagging(checkpoint_path=str(checkpoint), device=device)
        clipwise_output, embedding = tagger.inference(audio)
        scores = clipwise_output[0]
        labels = self._load_panns_labels(labels_csv)
        pairs = sorted(zip(labels, scores), key=lambda item: float(item[1]), reverse=True)
        top_tags = [(label, float(score)) for label, score in pairs[:12]]

        groups = {
            "music": {
                "Music",
                "Song",
                "Singing",
                "Vocal music",
                "Musical instrument",
                "Piano",
                "Guitar",
                "Violin, fiddle",
            },
            "speech": {
                "Speech",
                "Narration, monologue",
                "Conversation",
                "Male speech, man speaking",
                "Female speech, woman speaking",
            },
            "noise": {
                "Noise",
                "Static",
                "Hiss",
                "Hum",
                "Buzz",
                "Crackle",
                "Distortion",
            },
            "reverb": {
                "Reverberation",
                "Echo",
                "Inside, large room or hall",
                "Cavern, echo",
            },
        }
        group_scores = {}
        for key, names in groups.items():
            group_scores[key] = max((float(score) for label, score in top_tags if label in names), default=0.0)
        return {
            "top_tags": top_tags,
            "embedding_norm": float(np.linalg.norm(embedding[0])),
            "music_score": group_scores["music"],
            "speech_score": group_scores["speech"],
            "noise_score": group_scores["noise"],
            "reverb_score": group_scores["reverb"],
            "engine": "PANNs Cnn14",
        }

    def _recommended_ai_strategy(self, analysis):
        duration = float(analysis.get("duration_seconds", 0.0))
        vram_gb = self._gpu_vram_gb()
        if vram_gb <= 0.0:
            return "low_vram" if duration >= 180 else "balanced"
        if vram_gb and vram_gb < 7.0:
            return "low_vram"
        if duration >= 420:
            return "low_vram"
        if duration >= 240:
            return "balanced"
        if analysis.get("compression_score", 0.0) >= 0.6:
            return "low_vram"
        if analysis.get("click_score", 0.0) >= 0.35:
            return "balanced"
        return "quality"

    def _recommend_from_analysis(self, analysis):
        hum_strength = float(analysis.get("hum_strength", 0.0))
        click_score = float(analysis.get("click_score", 0.0))
        hiss_score = float(analysis.get("hiss_score", 0.0))
        compression_score = float(analysis.get("compression_score", 0.0))
        duration = float(analysis.get("duration_seconds", 0.0))
        panns = analysis.get("panns") or {}
        speech_score = float(panns.get("speech_score", 0.0))
        reverb_score = float(panns.get("reverb_score", 0.0))
        noise_score_panns = float(panns.get("noise_score", 0.0))

        if compression_score >= 0.62 and max(hum_strength, click_score, hiss_score) < 0.38:
            profile = "compression"
        elif max(hum_strength, click_score, hiss_score) >= 0.18:
            profile = "restore"
        else:
            profile = "enhance"

        noise_score = max(hum_strength, click_score, hiss_score, noise_score_panns * 0.7)
        if noise_score >= 0.65:
            preset = "archive"
        elif noise_score >= 0.42:
            preset = "heavy"
        elif noise_score >= 0.24:
            preset = "medium"
        else:
            preset = "light"

        ai_model_preset = self._recommended_ai_strategy(analysis)
        defaults = copy.deepcopy(self.PROFILE_DEFAULTS[profile])
        defaults["restoration_preset"] = preset
        defaults["hum_frequency"] = int(analysis.get("hum_frequency", 60))
        defaults["ai_model_preset"] = ai_model_preset
        defaults["restoration_cleanup"] = profile == "restore" or noise_score >= 0.22
        defaults["stem_rebalance"] = profile in {"stem"} or (profile in {"restore", "enhance"} and compression_score >= 0.45 and duration <= 480)
        defaults["bandwidth_restore"] = profile in {"compression", "advanced"} or (compression_score >= 0.72 and duration <= 140)
        if duration > 180:
            defaults["bandwidth_restore"] = False
        defaults["ai_dereverb"] = reverb_score >= 0.35
        defaults["backend"] = "roformer"
        if profile == "compression":
            defaults["clarity_mastering"] = True
            defaults["normalize_audio"] = True
            defaults["treble_boost"] = max(defaults.get("treble_boost", 0), 1)
        if speech_score > 0.65 and panns.get("music_score", 0.0) < 0.45:
            defaults["stem_rebalance"] = False
            defaults["clarity_mastering"] = False
        recommendation_label = self.PROFILE_LABELS[profile]
        model_label = self.AI_MODEL_PRESET_LABELS[ai_model_preset]
        issue_names = [name for name, _score in analysis.get("issues", [])]
        if issue_names:
            issue_text = ", ".join(issue_names[:3])
            summary = f"Detected {issue_text}. Recommended: {recommendation_label} with {model_label} strategy."
        else:
            summary = f"Source looks fairly clean. Recommended: {recommendation_label} with {model_label} strategy."
        return {
            "profile": profile,
            "options": defaults,
            "summary": summary,
            "recommendation_label": recommendation_label,
            "ai_model_label": model_label,
        }

    def _format_analysis_details(self, analysis, recommendation):
        issue_lines = []
        for name, strength in analysis.get("issues", []):
            issue_lines.append(f"- {name}: {strength:.2f}")
        if not issue_lines:
            issue_lines.append("- no dominant defect pattern detected")
        panns = analysis.get("panns") or {}
        panns_lines = []
        for label, score in panns.get("top_tags", [])[:6]:
            panns_lines.append(f"- {label}: {score:.2f}")
        if not panns_lines:
            panns_lines.append(f"- {analysis.get('panns_error', 'unavailable')}")
        options = recommendation["options"]
        return "\n".join(
            [
                "Source analysis",
                f"Duration: {analysis['duration_seconds']:.1f}s",
                f"Dynamic range: {analysis['dynamic_range_db']:.1f} dB",
                f"Spectral centroid: {analysis['spectral_centroid_hz']:.0f} Hz",
                f"High-frequency rolloff: {analysis['rolloff_hz']:.0f} Hz",
                f"Hum estimate: {analysis['hum_strength']:.2f} at {analysis['hum_frequency']} Hz",
                f"Clicks estimate: {analysis['click_score']:.2f}",
                f"Hiss estimate: {analysis['hiss_score']:.2f}",
                f"Compression estimate: {analysis['compression_score']:.2f}",
                "",
                "Detected issues:",
                *issue_lines,
                "",
                "PANNs learned cues:",
                *panns_lines,
                "",
                "Recommended plan",
                f"- Profile: {self.PROFILE_LABELS[recommendation['profile']]}",
                f"- Analysis engine: {analysis.get('analysis_engine_label', 'Hybrid (PANNs + DSP)')}",
                f"- Model strategy: {self.AI_MODEL_PRESET_LABELS[options['ai_model_preset']]}",
                f"- Restore strength: {self.RESTORATION_PRESET_LABELS[options['restoration_preset']]}",
                f"- Hum frequency: {options['hum_frequency']} Hz",
                f"- Dereverb: {'on' if options.get('ai_dereverb') else 'off'}",
                f"- Stem rebalance: {'on' if options['stem_rebalance'] else 'off'}",
                f"- Bandwidth restore: {'on' if options['bandwidth_restore'] else 'off'}",
                "",
                "Model note",
                "AudioSR remains experimental and is held back on long tracks to avoid slow or memory-heavy runs.",
            ]
        )

    def analyze_current_source(self):
        source_path = self.current_file or self.source_file
        if not source_path or not os.path.exists(source_path):
            raise RuntimeError("Download or import a song first.")
        self._set_status("Running source analysis...", progress=8, color="blue", error=None, details="")
        analysis = self._analyze_signal(source_path)
        analysis["analysis_engine_label"] = "DSP heuristics"
        if self.enable_learned_analysis and self._panns_available():
            try:
                if self._panns_assets_ready():
                    self._set_status("Running learned PANNs analysis...", progress=28, color="blue", error=None, details="")
                    analysis["panns"] = self._analyze_with_panns(source_path)
                    analysis["analysis_engine_label"] = "Hybrid (PANNs + DSP)"
                else:
                    analysis["panns_error"] = "Bundled PANNs model is not available locally; using DSP fallback."
            except Exception as exc:
                analysis["panns_error"] = str(exc)
        recommendation = self._recommend_from_analysis(analysis)
        analysis_record = {
            **analysis,
            **recommendation,
            "details": self._format_analysis_details(analysis, recommendation),
            "analysis_engine_label": analysis.get("analysis_engine_label", "DSP heuristics"),
        }
        self.last_analysis = analysis_record
        self.profile = recommendation["profile"]
        self._set_status(
            recommendation["summary"],
            progress=100,
            color="green",
            error=None,
            details=analysis_record["details"],
        )

    def _list_separator_models(self):
        result = self._run_optional_command(
            ["audio-separator", "--list_models", "--list_format", "json"],
            "audio-separator is not installed. Install it with:\n  pip install audio-separator",
        )
        return json.loads(result.stdout)

    def _collect_separator_model_map(self):
        try:
            model_list = self._list_separator_models()
        except Exception:
            return {}

        available = {}
        for family_name, family_models in model_list.items():
            for model_name, model_info in family_models.items():
                filename = model_info.get("filename")
                if not filename:
                    continue
                available[filename] = {
                    "name": model_name,
                    "family": family_name,
                    "search_blob": f"{family_name} {model_name} {filename}".lower(),
                }
        return available

    def resolve_separator_model(self, preferred_filenames):
        available = self._collect_separator_model_map()
        for filename in preferred_filenames:
            info = available.get(filename)
            if info:
                self.last_resolved_roformer_model = f"{info['name']} ({filename})"
                return filename
        if preferred_filenames:
            fallback = preferred_filenames[0]
            self.last_resolved_roformer_model = f"Requested model ({fallback})"
            return fallback
        self.last_resolved_roformer_model = "audio-separator package default"
        return None

    def resolve_separator_model_by_terms(self, required_terms):
        available = self._collect_separator_model_map()
        terms = [term.lower() for term in required_terms]
        for filename, info in available.items():
            blob = info["search_blob"]
            if all(term in blob for term in terms):
                self.last_resolved_roformer_model = f"{info['name']} ({filename})"
                return filename
        return None

    def resolve_roformer_model(self):
        return self.resolve_separator_model(self.DEFAULT_ROFORMER_PREFERENCES)

    def _normalize_restoration_preset(self, value):
        value = (value or "medium").strip().lower()
        return value if value in self.RESTORATION_PRESET_LABELS else "medium"

    def _normalize_hum_frequency(self, value):
        try:
            value = int(value)
        except (TypeError, ValueError):
            value = 60
        return value if value in self.HUM_FREQUENCIES else 60

    def _restoration_settings(self, preset):
        preset = self._normalize_restoration_preset(preset)
        settings = {
            "light": {
                "highpass": 24,
                "hum_harmonics": 3,
                "hum_width": 18,
                "hum_cut_db": -10.0,
                "declick_t": 1.8,
                "declick_a": 0.8,
                "noise_floor": -18,
                "hf_trim": -0.2,
            },
            "medium": {
                "highpass": 28,
                "hum_harmonics": 4,
                "hum_width": 20,
                "hum_cut_db": -13.0,
                "declick_t": 2.3,
                "declick_a": 1.0,
                "noise_floor": -22,
                "hf_trim": -0.4,
            },
            "heavy": {
                "highpass": 32,
                "hum_harmonics": 5,
                "hum_width": 22,
                "hum_cut_db": -16.0,
                "declick_t": 3.0,
                "declick_a": 1.15,
                "noise_floor": -26,
                "hf_trim": -0.8,
            },
            "archive": {
                "highpass": 36,
                "hum_harmonics": 6,
                "hum_width": 24,
                "hum_cut_db": -19.0,
                "declick_t": 3.8,
                "declick_a": 1.3,
                "noise_floor": -30,
                "hf_trim": -1.2,
            },
        }
        return settings[preset]

    def _build_dehum_filters(self, preset="medium", hum_frequency=60):
        settings = self._restoration_settings(preset)
        hum_frequency = self._normalize_hum_frequency(hum_frequency)
        filters = [f"highpass=f={settings['highpass']}"]
        for harmonic in range(1, settings["hum_harmonics"] + 1):
            frequency = hum_frequency * harmonic
            cut_db = settings["hum_cut_db"] + ((harmonic - 1) * 1.25)
            filters.append(
                f"equalizer=f={frequency}:width_type=h:width={settings['hum_width']}:g={cut_db:.1f}"
            )
        return filters

    def _build_declick_filters(self, preset="medium"):
        settings = self._restoration_settings(preset)
        return [f"adeclick=t={settings['declick_t']}:a={settings['declick_a']}"]

    def _build_hiss_filters(self, preset="medium"):
        settings = self._restoration_settings(preset)
        filters = [f"afftdn=nf={settings['noise_floor']}"]
        if settings["hf_trim"]:
            filters.append(f"equalizer=f=9600:width_type=o:width=0.9:g={settings['hf_trim']}")
        return filters

    def _build_restoration_filters(self, preset="medium", hum_frequency=60):
        return (
            self._build_dehum_filters(preset=preset, hum_frequency=hum_frequency)
            + self._build_declick_filters(preset=preset)
            + self._build_hiss_filters(preset=preset)
        )

    def apply_restoration_pipeline(self, file_path, preset="medium", hum_frequency=60, *, base_progress=18):
        preset = self._normalize_restoration_preset(preset)
        hum_frequency = self._normalize_hum_frequency(hum_frequency)
        if base_progress <= 0:
            base_progress = 18
        stages = [
            ("Removing mains hum", self._build_dehum_filters(preset=preset, hum_frequency=hum_frequency), base_progress),
            ("Repairing clicks and crackle", self._build_declick_filters(preset=preset), base_progress + 8),
            ("Reducing hiss", self._build_hiss_filters(preset=preset), base_progress + 16),
        ]
        for label, filters, percent in stages:
            self._apply_stage_filters(file_path, label, filters, percent)

    def _build_mastering_filters(self, *, conservative, clarity_mastering, normalize_audio, bass_boost, treble_boost, volume_boost):
        filters = []
        if clarity_mastering:
            if conservative:
                filters.extend([
                    "equalizer=f=220:width_type=o:width=1.3:g=-0.8",
                    "equalizer=f=2800:width_type=o:width=1.0:g=0.7",
                    "equalizer=f=8500:width_type=o:width=0.9:g=0.4",
                    "acompressor=threshold=-20dB:ratio=1.8:attack=18:release=220",
                    "stereotools=mlev=0.98",
                ])
            else:
                filters.extend([
                    "equalizer=f=250:width_type=o:width=1.2:g=-1.4",
                    "equalizer=f=3200:width_type=o:width=1.0:g=1.2",
                    "equalizer=f=9000:width_type=o:width=0.8:g=0.8",
                    "acompressor=threshold=-18dB:ratio=2.4:attack=15:release=180",
                    "bass=g=0.8",
                    "treble=g=0.8",
                    "stereotools=mlev=0.96",
                ])
        if bass_boost:
            filters.append(f"bass=g={bass_boost}")
        if treble_boost:
            filters.append(f"treble=g={treble_boost}")
        if normalize_audio:
            filters.append("loudnorm=I=-16:TP=-1.0:LRA=11")
        if volume_boost:
            filters.append(f"volume={volume_boost}dB")
        if filters:
            filters.extend(["aresample=48000", "alimiter=limit=0.891"])
        return filters

    def _apply_ffmpeg_filter_chain(self, file_path, filters):
        if not filters:
            return
        base, ext = os.path.splitext(file_path)
        temp_output = f"{base}_stage{ext or '.wav'}"
        cmd = [
            self._ffmpeg_executable(), "-y",
            "-i", file_path,
            "-af", ",".join(filters),
            *self._audio_encode_args(temp_output),
            temp_output,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0 or not os.path.exists(temp_output):
            if os.path.exists(temp_output):
                os.remove(temp_output)
            detail = result.stderr[-1200:] if result.stderr else "Unknown FFmpeg error"
            raise RuntimeError(f"FFmpeg enhancement failed:\n{detail}")
        self._replace_file_with_retry(temp_output, file_path)

    def _apply_audiosr(self, file_path):
        output_dir = tempfile.mkdtemp(prefix="audiosr_")
        try:
            audiosr_python = self.root_dir / ".venv_audiosr" / "Scripts" / "python.exe"
            audiosr_input = os.path.join(output_dir, "audiosr_input_24k.wav")
            downsample_cmd = [
                self._ffmpeg_executable(), "-y",
                "-i", file_path,
                "-ar", "24000",
                "-ac", "2",
                "-c:a", "pcm_s16le",
                audiosr_input,
            ]
            downsample = subprocess.run(downsample_cmd, capture_output=True, text=True)
            if downsample.returncode != 0 or not os.path.exists(audiosr_input):
                detail = downsample.stderr[-1000:] if downsample.stderr else "Unknown FFmpeg error"
                raise RuntimeError(f"Failed to prepare AudioSR input:\n{detail}")

            device = "cpu"
            device_check = subprocess.run(
                [str(audiosr_python), "-c", "import torch; print('cuda' if torch.cuda.is_available() else 'cpu')"],
                capture_output=True,
                text=True,
                env=self._build_optional_command_env(),
            )
            if device_check.returncode == 0 and device_check.stdout.strip().endswith("cuda"):
                device = "cuda"

            def run_audiosr(target_device, ddim_steps):
                self._run_optional_command(
                    [
                        str(audiosr_python), "-m", "audiosr",
                        "-i", audiosr_input,
                        "-s", output_dir,
                        "--model_name", "basic",
                        "--ddim_steps", str(ddim_steps),
                        "-d", target_device,
                    ],
                    "AudioSR is not installed. Install it with:\n  .\\.venv_audiosr\\Scripts\\python.exe -m pip install -r requirements-audiosr.txt",
                    extra_venvs=[".venv_audiosr"],
                )

            if device == "cuda":
                try:
                    self.update_ai_progress("Bandwidth Restore on GPU", 72)
                    run_audiosr("cuda", 30)
                except Exception as first_error:
                    detail = str(first_error).lower()
                    if "cuda out of memory" in detail or "outofmemoryerror" in detail or "cudaerror_memoryallocation" in detail:
                        self.update_ai_progress("AudioSR retried with lower memory", 78, color="orange")
                        try:
                            run_audiosr("cuda", 10)
                        except Exception:
                            self.update_ai_progress("AudioSR falling back to CPU", 82, color="orange")
                            run_audiosr("cpu", 10)
                    else:
                        raise
            else:
                run_audiosr("cpu", 10)

            enhanced = self._find_newest_audio_file(output_dir, exclude_paths=[file_path])
            if not enhanced:
                raise RuntimeError("AudioSR finished but did not create an output audio file.")
            self._convert_to_work_wav(enhanced, file_path)
        finally:
            shutil.rmtree(output_dir, ignore_errors=True)

    def apply_bandwidth_restore(self, file_path, provider=None):
        provider = provider or self.compression_restore_provider
        if provider == "apollo":
            raise RuntimeError("Apollo is reserved as the future compression-restoration provider but is not integrated yet.")
        self._apply_audiosr(file_path)

    @staticmethod
    def _is_memory_pressure_error(exc):
        detail = str(exc).lower()
        markers = [
            "cuda out of memory",
            "outofmemoryerror",
            "cudaerror_memoryallocation",
            "defaultcpuallocator: not enough memory",
            "not enough memory",
            "tried to allocate",
        ]
        return any(marker in detail for marker in markers)

    @staticmethod
    def _match_length(chunk, target_length):
        if chunk.shape[1] == target_length:
            return chunk
        if chunk.shape[1] > target_length:
            return chunk[:, :target_length]
        pad_width = target_length - chunk.shape[1]
        return __import__("numpy").pad(chunk, ((0, 0), (0, pad_width)), mode="constant")

    def separate_audio_demucs_cli(self, file_path):
        import numpy as np
        import soundfile as sf

        output_dir = tempfile.mkdtemp(prefix="demucs_stems_")
        try:
            try:
                import torch

                device = "cuda" if torch.cuda.is_available() else "cpu"
            except Exception:
                device = "cpu"

            self.update_ai_progress(f"Running Demucs legacy fallback on {device.upper()}", 42)
            command = [
                sys.executable,
                "-m", "demucs",
                "-n", "htdemucs_ft",
                "-d", device,
                "--overlap", "0.25",
                "--shifts", "1",
                "--filename", "{stem}.{ext}",
                "-o", output_dir,
                file_path,
            ]
            result = subprocess.run(command, capture_output=True, text=True, env=self._build_optional_command_env())
            if result.returncode != 0:
                detail = result.stderr[-1200:] if result.stderr else result.stdout[-1200:]
                raise RuntimeError(f"Demucs failed:\n{detail}")

            stem_files = {}
            for candidate in glob.glob(os.path.join(output_dir, "**", "*.wav"), recursive=True):
                stem_name = os.path.splitext(os.path.basename(candidate))[0].lower()
                if stem_name in {"vocals", "drums", "bass", "other"}:
                    stem_files[stem_name] = candidate
            missing = [stem for stem in ("vocals", "drums", "bass", "other") if stem not in stem_files]
            if missing:
                raise RuntimeError(f"Demucs did not produce expected stems: {', '.join(missing)}")

            original_audio, original_sr = sf.read(file_path, always_2d=True)
            target_channels = original_audio.shape[1]
            target_len = original_audio.shape[0]
            stem_gains_db = {"vocals": 1.2, "drums": 0.4, "bass": 0.8, "other": -0.7}
            cleaned = np.zeros((target_channels, target_len), dtype=np.float32)
            stem_sum = np.zeros_like(cleaned)

            for stem_name in ("vocals", "drums", "bass", "other"):
                stem_audio, stem_sr = sf.read(stem_files[stem_name], always_2d=True)
                if stem_sr != original_sr:
                    converted = os.path.join(output_dir, f"{stem_name}_resampled.wav")
                    cmd = [
                        self._ffmpeg_executable(), "-y",
                        "-i", stem_files[stem_name],
                        "-ar", str(original_sr),
                        "-ac", str(target_channels),
                        "-c:a", "pcm_s16le",
                        converted,
                    ]
                    resample = subprocess.run(cmd, capture_output=True, text=True)
                    if resample.returncode != 0 or not os.path.exists(converted):
                        detail = resample.stderr[-1000:] if resample.stderr else "Unknown FFmpeg error"
                        raise RuntimeError(f"Failed to resample {stem_name} stem:\n{detail}")
                    stem_audio, _ = sf.read(converted, always_2d=True)

                stem = stem_audio.T.astype(np.float32)
                if stem.shape[0] == 1 and target_channels > 1:
                    stem = np.repeat(stem, target_channels, axis=0)
                elif stem.shape[0] != target_channels:
                    stem = stem[:target_channels, :]
                stem = self._match_length(stem, target_len)
                gain = 10 ** (stem_gains_db.get(stem_name, 0.0) / 20.0)
                cleaned += stem * gain
                stem_sum += stem

            original = original_audio.T.astype(np.float32)
            residual = original - stem_sum
            cleaned += residual * 0.25
            input_peak = max(float(np.max(np.abs(original))), 1e-6)
            output_peak = max(float(np.max(np.abs(cleaned))), 1e-6)
            if output_peak > input_peak:
                cleaned *= input_peak / output_peak
            cleaned = np.clip(cleaned, -0.99, 0.99)
            sf.write(file_path, cleaned.T, original_sr, subtype="PCM_16")
        finally:
            shutil.rmtree(output_dir, ignore_errors=True)

    def _run_separator_model(self, file_path, preferred_model, *, label, sample_rate, extra_args, progress=42):
        output_dir = tempfile.mkdtemp(prefix="ai_separator_")
        self.update_ai_progress(label, progress)
        command = [
            "audio-separator",
            file_path,
            "--model_file_dir", str(self.audio_separator_model_dir),
            "--output_format", "WAV",
            "--output_dir", output_dir,
            "--sample_rate", str(sample_rate),
            "--log_level", "warning",
        ]
        if preferred_model:
            command.extend(["--model_filename", preferred_model])
        command.extend(extra_args)
        try:
            import torch

            if torch.cuda.is_available():
                command.append("--use_autocast")
        except Exception:
            pass
        self._run_optional_command(
            command,
            "audio-separator is not installed. Install it with:\n  pip install audio-separator",
        )
        return output_dir

    def apply_roformer_dereverb(self, file_path):
        preferred_model = self.resolve_separator_model_by_terms(["dereverb"])
        if not preferred_model:
            raise RuntimeError("No RoFormer dereverb model is available in the installed separator registry.")
        output_dir = self._run_separator_model(
            file_path,
            preferred_model,
            label="Running dereverb model",
            sample_rate=48000,
            extra_args=["--mdxc_segment_size", "256", "--mdxc_overlap", "8", "--mdxc_batch_size", "1"],
            progress=38,
        )
        try:
            enhanced = self._find_newest_audio_file(output_dir, exclude_paths=[file_path])
            if not enhanced:
                raise RuntimeError("The dereverb model finished but did not create an output file.")
            if os.path.abspath(enhanced) != os.path.abspath(file_path):
                self._replace_file_with_retry(enhanced, file_path)
        finally:
            shutil.rmtree(output_dir, ignore_errors=True)

    def separate_audio_roformer_cli(self, file_path, ai_model_preset="balanced"):
        import re
        import soundfile as sf

        output_dir = None
        try:
            ai_model_preset = (ai_model_preset or "balanced").strip().lower()
            if ai_model_preset not in self.AI_MODEL_PRESETS:
                ai_model_preset = "balanced"
            preset = self.AI_MODEL_PRESETS[ai_model_preset]
            preferred_model = self.resolve_separator_model(preset["preferred_models"])
            label = self.AI_MODEL_PRESET_LABELS[ai_model_preset]
            output_dir = self._run_separator_model(
                file_path,
                preferred_model,
                label=f"Running {label} stem model",
                sample_rate=preset["sample_rate"],
                extra_args=preset["extra_args"],
                progress=42,
            )

            stem_files = {}
            for candidate in glob.glob(os.path.join(output_dir, "*.wav")):
                name = os.path.basename(candidate).lower()
                match = re.search(r"\(([^)]+)\)", name)
                if match:
                    stem_files[match.group(1)] = candidate
                elif "vocals" in name or "vocal" in name:
                    stem_files["vocals"] = candidate
                elif "instrumental" in name or "no_vocals" in name:
                    stem_files["instrumental"] = candidate

            if not stem_files:
                raise RuntimeError("Stem separation did not produce any recognized stems.")

            if "vocals" in stem_files and "instrumental" in stem_files:
                filters = (
                    "[0:a]volume=1.15[v];"
                    "[1:a]volume=0.98[i];"
                    "[v][i]amix=inputs=2:duration=longest:normalize=0,"
                    "alimiter=limit=0.891,aresample=48000"
                )
                temp_mix = os.path.join(output_dir, "roformer_mix.wav")
                cmd = [
                    self._ffmpeg_executable(), "-y",
                    "-i", stem_files["vocals"],
                    "-i", stem_files["instrumental"],
                    "-filter_complex", filters,
                    "-c:a", "pcm_s16le",
                    temp_mix,
                ]
                result = subprocess.run(cmd, capture_output=True, text=True)
                if result.returncode != 0 or not os.path.exists(temp_mix):
                    detail = result.stderr[-1200:] if result.stderr else "Unknown FFmpeg error"
                    raise RuntimeError(f"Failed to recombine stems:\n{detail}")
                self._replace_file_with_retry(temp_mix, file_path)
                return

            if "vocals" not in stem_files:
                raise RuntimeError("The stem model ran, but no vocals stem was found for recombination.")

            original_audio, original_sr = sf.read(file_path, always_2d=True)
            target_channels = original_audio.shape[1]
            target_len = original_audio.shape[0]
            stem_gains_db = {
                "vocals": 1.15,
                "bass": 0.7,
                "drums": 0.35,
                "guitar": 0.15,
                "piano": 0.15,
                "other": -0.6,
            }
            cleaned = np.zeros((target_channels, target_len), dtype=np.float32)
            stem_sum = np.zeros_like(cleaned)

            for stem_name, stem_path in stem_files.items():
                stem_audio, stem_sr = sf.read(stem_path, always_2d=True)
                if stem_sr != original_sr:
                    converted = os.path.join(output_dir, f"{stem_name}_resampled.wav")
                    cmd = [
                        self._ffmpeg_executable(), "-y",
                        "-i", stem_path,
                        "-ar", str(original_sr),
                        "-ac", str(target_channels),
                        "-c:a", "pcm_s16le",
                        converted,
                    ]
                    resample = subprocess.run(cmd, capture_output=True, text=True)
                    if resample.returncode != 0 or not os.path.exists(converted):
                        detail = resample.stderr[-1000:] if resample.stderr else "Unknown FFmpeg error"
                        raise RuntimeError(f"Failed to resample stem '{stem_name}':\n{detail}")
                    stem_audio, _ = sf.read(converted, always_2d=True)

                stem = stem_audio.T.astype(np.float32)
                if stem.shape[0] == 1 and target_channels > 1:
                    stem = np.repeat(stem, target_channels, axis=0)
                elif stem.shape[0] != target_channels:
                    stem = stem[:target_channels, :]
                stem = self._match_length(stem, target_len)

                gain = 10 ** (stem_gains_db.get(stem_name, 0.0) / 20.0)
                cleaned += stem * gain
                stem_sum += stem

            original = original_audio.T.astype(np.float32)
            residual = original - stem_sum
            cleaned += residual * 0.2
            input_peak = max(float(np.max(np.abs(original))), 1e-6)
            output_peak = max(float(np.max(np.abs(cleaned))), 1e-6)
            if output_peak > input_peak:
                cleaned *= input_peak / output_peak
            cleaned = np.clip(cleaned, -0.99, 0.99)
            sf.write(file_path, cleaned.T, original_sr, subtype="PCM_16")
        finally:
            if output_dir:
                shutil.rmtree(output_dir, ignore_errors=True)

    def _normalize_options(self, raw):
        profile = raw.get("profile", self.profile)
        if profile not in self.PROFILE_DEFAULTS:
            profile = "restore"
        options = copy.deepcopy(self.PROFILE_DEFAULTS[profile])
        for key in ("restoration_cleanup", "clarity_mastering", "normalize_audio", "stem_rebalance", "bandwidth_restore", "ai_dereverb"):
            if key in raw:
                options[key] = bool(raw.get(key))
        for key in ("bass_boost", "treble_boost", "volume_boost"):
            if key in raw:
                try:
                    options[key] = int(raw.get(key))
                except (TypeError, ValueError):
                    options[key] = self.PROFILE_DEFAULTS[profile][key]
        if "restoration_preset" in raw:
            options["restoration_preset"] = self._normalize_restoration_preset(raw.get("restoration_preset"))
        if "hum_frequency" in raw:
            options["hum_frequency"] = self._normalize_hum_frequency(raw.get("hum_frequency"))
        ai_model_preset = raw.get("ai_model_preset", options.get("ai_model_preset", "balanced"))
        options["ai_model_preset"] = ai_model_preset if ai_model_preset in self.AI_MODEL_PRESETS else "balanced"
        backend = raw.get("backend", options["backend"])
        options["backend"] = backend if backend in {"roformer", "demucs_legacy"} else "roformer"

        if profile != "advanced":
            if profile == "stem":
                options["restoration_cleanup"] = False
                options["clarity_mastering"] = False
                options["normalize_audio"] = False
                options["stem_rebalance"] = True
                options["bandwidth_restore"] = False
                options["ai_dereverb"] = False
            elif profile == "compression":
                options["backend"] = "roformer"
            elif profile == "enhance":
                options["bandwidth_restore"] = False
                options["backend"] = "roformer"
            elif profile == "restore":
                options["backend"] = "roformer"
        if profile == "compression":
            options["ai_dereverb"] = False
        if profile not in {"restore", "advanced", "compression"}:
            options["bandwidth_restore"] = False
        if profile != "advanced" and options["backend"] == "demucs_legacy":
            options["backend"] = "roformer"
        return profile, options

    def _apply_stage_filters(self, file_path, label, filters, percent):
        if not filters:
            return
        self.update_ai_progress(label, percent)
        self._apply_ffmpeg_filter_chain(file_path, filters)

    def enhance_current(self, raw_options):
        if not self.current_file or not os.path.exists(self.current_file):
            raise RuntimeError("Download or import a song first.")

        previous_file = self.current_file
        profile, options = self._normalize_options(raw_options)
        temp_dir = tempfile.mkdtemp(prefix="restoration_work_")
        temp_work = os.path.join(temp_dir, "working.wav")
        try:
            self.profile = profile
            duration_seconds = self._estimate_audio_duration_seconds(previous_file)
            if duration_seconds and duration_seconds > 240:
                minutes = int(round(duration_seconds / 60.0))
                prep_msg = f"Preparing lossless working master for a {minutes} min track (this can take a while)..."
            else:
                prep_msg = "Preparing lossless working master (first pass may take up to a minute)..."
            self._set_status(prep_msg, progress=5, color="blue", error=None, details="")
            self._convert_to_work_wav(previous_file, temp_work)

            if profile == "restore":
                if options["restoration_cleanup"]:
                    self.apply_restoration_pipeline(
                        temp_work,
                        preset=options["restoration_preset"],
                        hum_frequency=options["hum_frequency"],
                        base_progress=14,
                    )
                if options["ai_dereverb"]:
                    self.apply_roformer_dereverb(temp_work)
                if options["stem_rebalance"]:
                    if options["backend"] == "demucs_legacy":
                        self.separate_audio_demucs_cli(temp_work)
                    else:
                        self.separate_audio_roformer_cli(temp_work, options["ai_model_preset"])
                if options["bandwidth_restore"]:
                    duration_seconds = self._estimate_audio_duration_seconds(temp_work)
                    if duration_seconds and duration_seconds > 180:
                        self.update_ai_progress("AudioSR skipped for long track", 74, color="orange")
                    else:
                        try:
                            self.update_ai_progress("Bandwidth Restore (Experimental)", 70, color="purple")
                            self.apply_bandwidth_restore(temp_work)
                        except Exception as exc:
                            if self._is_memory_pressure_error(exc):
                                self.update_ai_progress("AudioSR skipped due to memory limits", 78, color="orange")
                            else:
                                raise
                mastering_filters = self._build_mastering_filters(
                    conservative=True,
                    clarity_mastering=options["clarity_mastering"],
                    normalize_audio=options["normalize_audio"],
                    bass_boost=options["bass_boost"],
                    treble_boost=options["treble_boost"],
                    volume_boost=options["volume_boost"],
                )
                self._apply_stage_filters(temp_work, "Applying conservative mastering", mastering_filters, 86)
            elif profile == "enhance":
                if options["ai_dereverb"]:
                    self.apply_roformer_dereverb(temp_work)
                mastering_filters = self._build_mastering_filters(
                    conservative=False,
                    clarity_mastering=options["clarity_mastering"],
                    normalize_audio=options["normalize_audio"],
                    bass_boost=options["bass_boost"],
                    treble_boost=options["treble_boost"],
                    volume_boost=options["volume_boost"],
                )
                self._apply_stage_filters(temp_work, "Applying clarity and mastering", mastering_filters, 22)
                if options["stem_rebalance"]:
                    self.separate_audio_roformer_cli(temp_work, options["ai_model_preset"])
            elif profile == "compression":
                if options["restoration_cleanup"]:
                    self.apply_restoration_pipeline(
                        temp_work,
                        preset=options["restoration_preset"],
                        hum_frequency=options["hum_frequency"],
                        base_progress=12,
                    )
                if options["ai_dereverb"]:
                    self.apply_roformer_dereverb(temp_work)
                if options["stem_rebalance"]:
                    self.separate_audio_roformer_cli(temp_work, options["ai_model_preset"])
                if options["bandwidth_restore"]:
                    duration_seconds = self._estimate_audio_duration_seconds(temp_work)
                    if duration_seconds and duration_seconds > 180:
                        self.update_ai_progress("Bandwidth restore skipped for long track", 70, color="orange")
                    else:
                        try:
                            self.update_ai_progress("Bandwidth repair", 66, color="purple")
                            self.apply_bandwidth_restore(temp_work)
                        except Exception as exc:
                            if self._is_memory_pressure_error(exc):
                                self.update_ai_progress("Bandwidth repair skipped due to memory limits", 74, color="orange")
                            else:
                                raise
                mastering_filters = self._build_mastering_filters(
                    conservative=False,
                    clarity_mastering=options["clarity_mastering"],
                    normalize_audio=options["normalize_audio"],
                    bass_boost=options["bass_boost"],
                    treble_boost=options["treble_boost"],
                    volume_boost=options["volume_boost"],
                )
                self._apply_stage_filters(temp_work, "Applying repair mastering", mastering_filters, 84)
            elif profile == "stem":
                self.update_ai_progress("Stem rebalance", 24, color="purple")
                self.separate_audio_roformer_cli(temp_work, options["ai_model_preset"])
            else:
                if options["restoration_cleanup"]:
                    self.apply_restoration_pipeline(
                        temp_work,
                        preset=options["restoration_preset"],
                        hum_frequency=options["hum_frequency"],
                        base_progress=12,
                    )
                if options["ai_dereverb"]:
                    self.apply_roformer_dereverb(temp_work)
                if options["stem_rebalance"]:
                    if options["backend"] == "demucs_legacy":
                        self.separate_audio_demucs_cli(temp_work)
                    else:
                        self.separate_audio_roformer_cli(temp_work, options["ai_model_preset"])
                if options["bandwidth_restore"]:
                    duration_seconds = self._estimate_audio_duration_seconds(temp_work)
                    if duration_seconds and duration_seconds > 180:
                        self.update_ai_progress("AudioSR skipped for long track", 74, color="orange")
                    else:
                        try:
                            self.update_ai_progress("Bandwidth Restore (Experimental)", 70, color="purple")
                            self.apply_bandwidth_restore(temp_work)
                        except Exception as exc:
                            if self._is_memory_pressure_error(exc):
                                self.update_ai_progress("AudioSR skipped due to memory limits", 78, color="orange")
                            else:
                                raise
                mastering_filters = self._build_mastering_filters(
                    conservative=False,
                    clarity_mastering=options["clarity_mastering"],
                    normalize_audio=options["normalize_audio"],
                    bass_boost=options["bass_boost"],
                    treble_boost=options["treble_boost"],
                    volume_boost=options["volume_boost"],
                )
                self._apply_stage_filters(temp_work, "Advanced mastering stage", mastering_filters, 84)

            output_file = self._next_enhanced_filename()
            self.update_ai_progress("Saving enhanced working version", 95, color="blue")
            shutil.copy2(temp_work, output_file)
            self._remember_undo_state(previous_file)
            self.current_file = str(output_file)
            self._bump_media_revision()
            self._set_status("Enhanced version ready for A/B compare.", progress=100, color="green", error=None, details="")
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def run_job(self, target, *args, **kwargs):
        with self.lock:
            if self.processing:
                raise RuntimeError("A job is already running.")
            self.processing = True
            self.progress = 0.0
            self.last_error = None
            self.last_details = ""

        def worker():
            try:
                target(*args, **kwargs)
            except Exception as exc:
                traceback_text = traceback.format_exc()
                self._set_status(
                    self._friendly_job_error(exc),
                    progress=0,
                    color="red",
                    error=str(exc),
                    details=traceback_text,
                )
            finally:
                with self.lock:
                    self.processing = False

        self.processing_thread = threading.Thread(target=worker, daemon=True)
        self.processing_thread.start()

    @staticmethod
    def _friendly_job_error(exc: Exception):
        message = str(exc).strip()
        if not message:
            return "Something went wrong."
        return message.splitlines()[0]


ROOT = Path(__file__).resolve().parent
