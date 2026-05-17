import copy
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
from pathlib import Path

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
        "restore": "Restore Old Recording",
        "enhance": "Enhance Song",
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
            "restoration_preset": "medium",
            "hum_frequency": 60,
            "backend": "roformer",
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
            "restoration_preset": "light",
            "hum_frequency": 60,
            "backend": "roformer",
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
            "restoration_preset": "light",
            "hum_frequency": 60,
            "backend": "roformer",
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
            "restoration_preset": "medium",
            "hum_frequency": 60,
            "backend": "roformer",
            "bass_boost": 0,
            "treble_boost": 0,
            "volume_boost": 0,
        },
    }
    BROWSER_OPTIONS = ["chrome", "edge", "firefox"]
    DEFAULT_ROFORMER_PREFERENCES = [
        "BS-Roformer-SW.ckpt",
        "model_bs_roformer_ep_317_sdr_12.9755.ckpt",
    ]

    def __init__(self, root_dir: Path):
        self.root_dir = root_dir
        self.sessions_dir = self.root_dir / "sessions"
        self.sessions_dir.mkdir(parents=True, exist_ok=True)
        self.audio_separator_model_dir = self.root_dir / "models" / "audio-separator"
        self.audio_separator_model_dir.mkdir(parents=True, exist_ok=True)
        self.hf_cache_dir = self.root_dir / "models" / "huggingface"
        self.hf_cache_dir.mkdir(parents=True, exist_ok=True)

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

    def _list_separator_models(self):
        result = self._run_optional_command(
            ["audio-separator", "--list_models", "--list_format", "json"],
            "audio-separator is not installed. Install it with:\n  pip install audio-separator",
        )
        return json.loads(result.stdout)

    def resolve_roformer_model(self):
        try:
            model_list = self._list_separator_models()
        except Exception:
            self.last_resolved_roformer_model = "audio-separator package default"
            return None

        available = {}
        for family_name, family_models in model_list.items():
            for model_name, model_info in family_models.items():
                filename = model_info.get("filename")
                search_blob = f"{family_name} {model_name} {filename or ''}".lower()
                if filename and "roformer" in search_blob:
                    available[filename] = model_name
        for filename in self.DEFAULT_ROFORMER_PREFERENCES:
            if filename in available:
                self.last_resolved_roformer_model = f"{available[filename]} ({filename})"
                return filename
        self.last_resolved_roformer_model = "audio-separator package default"
        return None

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

    def separate_audio_roformer_cli(self, file_path):
        import re
        import numpy as np
        import soundfile as sf

        output_dir = tempfile.mkdtemp(prefix="roformer_stems_")
        try:
            preferred_model = self.resolve_roformer_model()
            self.update_ai_progress("Running RoFormer stem rebalance", 42)
            command = [
                "audio-separator",
                file_path,
                "--model_file_dir", str(self.audio_separator_model_dir),
                "--output_format", "WAV",
                "--output_dir", output_dir,
                "--sample_rate", "48000",
                "--log_level", "warning",
            ]
            if preferred_model:
                command.extend(["--model_filename", preferred_model])
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
                raise RuntimeError("RoFormer separation did not produce any recognized stems.")

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
                    raise RuntimeError(f"Failed to recombine RoFormer stems:\n{detail}")
                self._replace_file_with_retry(temp_mix, file_path)
                return

            if "vocals" not in stem_files:
                raise RuntimeError("RoFormer produced stems, but no vocals stem was found for recombination.")

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
                        raise RuntimeError(f"Failed to resample RoFormer stem '{stem_name}':\n{detail}")
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
            shutil.rmtree(output_dir, ignore_errors=True)

    def _normalize_options(self, raw):
        profile = raw.get("profile", self.profile)
        if profile not in self.PROFILE_DEFAULTS:
            profile = "restore"
        options = copy.deepcopy(self.PROFILE_DEFAULTS[profile])
        for key in ("restoration_cleanup", "clarity_mastering", "normalize_audio", "stem_rebalance", "bandwidth_restore"):
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
        backend = raw.get("backend", options["backend"])
        options["backend"] = backend if backend in {"roformer", "demucs_legacy"} else "roformer"

        if profile != "advanced":
            if profile == "stem":
                options["restoration_cleanup"] = False
                options["clarity_mastering"] = False
                options["normalize_audio"] = False
                options["stem_rebalance"] = True
                options["bandwidth_restore"] = False
            elif profile == "enhance":
                options["bandwidth_restore"] = False
                options["backend"] = "roformer"
            elif profile == "restore":
                options["backend"] = "roformer"
        if profile not in {"restore", "advanced"}:
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
            self._set_status("Preparing lossless working master...", progress=5, color="blue", error=None, details="")
            self._convert_to_work_wav(previous_file, temp_work)

            if profile == "restore":
                if options["restoration_cleanup"]:
                    self.apply_restoration_pipeline(
                        temp_work,
                        preset=options["restoration_preset"],
                        hum_frequency=options["hum_frequency"],
                        base_progress=14,
                    )
                if options["stem_rebalance"]:
                    if options["backend"] == "demucs_legacy":
                        self.separate_audio_demucs_cli(temp_work)
                    else:
                        self.separate_audio_roformer_cli(temp_work)
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
                    self.separate_audio_roformer_cli(temp_work)
            elif profile == "stem":
                self.update_ai_progress("Stem rebalance", 24, color="purple")
                self.separate_audio_roformer_cli(temp_work)
            else:
                if options["restoration_cleanup"]:
                    self.apply_restoration_pipeline(
                        temp_work,
                        preset=options["restoration_preset"],
                        hum_frequency=options["hum_frequency"],
                        base_progress=12,
                    )
                if options["stem_rebalance"]:
                    if options["backend"] == "demucs_legacy":
                        self.separate_audio_demucs_cli(temp_work)
                    else:
                        self.separate_audio_roformer_cli(temp_work)
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
