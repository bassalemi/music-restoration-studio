import argparse
import json
import math
import shutil
import struct
import tempfile
import time
import wave
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def make_backend():
    from restoration_backend import RestorationStudioBackend

    temp_root = Path(tempfile.mkdtemp(prefix="restoration_backend_test_"))
    backend = RestorationStudioBackend(temp_root)
    backend.enable_learned_analysis = False
    return backend, temp_root


def make_sample_wav(root: Path, *, seconds: float = 2.0, sample_rate: int = 24000) -> Path:
    sample_path = root / "sample_input.wav"
    total_frames = int(seconds * sample_rate)
    with wave.open(str(sample_path), "wb") as handle:
        handle.setnchannels(2)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        frames = bytearray()
        for index in range(total_frames):
            t = index / sample_rate
            tone = 0.35 * math.sin(2 * math.pi * 220 * t)
            hum = 0.05 * math.sin(2 * math.pi * 60 * t)
            click = 0.65 if index % 4096 == 0 else 0.0
            sample = max(-0.99, min(0.99, tone + hum + click))
            pcm = int(sample * 32767)
            frames.extend(struct.pack("<hh", pcm, pcm))
        handle.writeframes(frames)
    return sample_path


def assert_exists(path):
    path = Path(path)
    if not path.exists() or path.stat().st_size <= 0:
        raise AssertionError(f"Expected file to exist with data: {path}")


def test_session_and_history():
    backend, temp_root = make_backend()
    try:
        copied = backend._start_fresh_session(make_sample_wav(temp_root))
        assert copied.exists()
        assert backend.source_file == str(copied)
        assert backend.current_file == str(copied)

        first = backend._next_enhanced_filename()
        shutil.copy2(copied, first)
        backend._remember_undo_state(backend.current_file)
        backend.current_file = str(first)

        second = backend._next_enhanced_filename()
        shutil.copy2(first, second)
        backend._remember_undo_state(backend.current_file)
        backend.current_file = str(second)

        backend.undo_last()
        assert backend.current_file == str(first)
        backend.revert_to_original()
        assert backend.current_file == backend.source_file
        return {"session_dir": str(backend.current_session_dir)}
    finally:
        shutil.rmtree(temp_root, ignore_errors=True)


def test_download_options_and_profiles():
    backend, temp_root = make_backend()
    try:
        profile, options = backend._normalize_options(
            {"profile": "restore", "bandwidth_restore": True, "restoration_preset": "heavy", "hum_frequency": 50}
        )
        assert profile == "restore"
        assert options["bandwidth_restore"] is True
        assert options["restoration_preset"] == "heavy"
        assert options["hum_frequency"] == 50

        profile, options = backend._normalize_options({"profile": "enhance", "bandwidth_restore": True, "backend": "demucs_legacy"})
        assert profile == "enhance"
        assert options["bandwidth_restore"] is False
        assert options["backend"] == "roformer"
        assert options["ai_model_preset"] == "balanced"

        profile, options = backend._normalize_options({"profile": "compression", "bandwidth_restore": True, "ai_model_preset": "low_vram"})
        assert profile == "compression"
        assert options["bandwidth_restore"] is True
        assert options["ai_model_preset"] == "low_vram"
        assert options["ai_dereverb"] is False

        profile, options = backend._normalize_options({"profile": "stem", "clarity_mastering": True})
        assert profile == "stem"
        assert options["stem_rebalance"] is True
        assert options["clarity_mastering"] is False

        direct_error = backend._friendly_download_error(RuntimeError("HTTP Error 403: Forbidden"), "direct")
        browser_error = backend._friendly_download_error(RuntimeError("HTTP Error 403: Forbidden"), "browser")
        assert "browser session" in direct_error.lower()
        assert "different browser session" in browser_error.lower()
        return {"profiles": "ok"}
    finally:
        shutil.rmtree(temp_root, ignore_errors=True)


def test_ai_analysis_and_recommendation():
    backend, temp_root = make_backend()
    try:
        copied = backend._start_fresh_session(make_sample_wav(temp_root, seconds=4.0))
        backend.analyze_current_source()
        analysis = backend.last_analysis
        assert analysis is not None
        assert analysis["profile"] in {"restore", "enhance", "compression", "stem", "advanced"}
        assert analysis["options"]["ai_model_preset"] in backend.AI_MODEL_PRESETS
        assert analysis["options"]["hum_frequency"] in backend.HUM_FREQUENCIES
        assert analysis["analysis_engine_label"] == "DSP heuristics"
        assert "AI source analysis" in analysis["details"]
        assert analysis["hum_strength"] >= 0.0
        assert Path(copied).exists()
        return {
            "summary": analysis["summary"],
            "profile": analysis["profile"],
            "ai_model_preset": analysis["options"]["ai_model_preset"],
        }
    finally:
        shutil.rmtree(temp_root, ignore_errors=True)


def test_filters_and_export():
    backend, temp_root = make_backend()
    try:
        copied = backend._start_fresh_session(make_sample_wav(temp_root))
        wav_work = temp_root / "work.wav"
        backend._convert_to_work_wav(str(copied), str(wav_work))
        assert_exists(wav_work)

        dehum_filters = backend._build_dehum_filters(preset="archive", hum_frequency=50)
        declick_filters = backend._build_declick_filters(preset="archive")
        hiss_filters = backend._build_hiss_filters(preset="archive")
        restoration_filters = backend._build_restoration_filters(preset="archive", hum_frequency=50)
        mastering_filters = backend._build_mastering_filters(
            conservative=True,
            clarity_mastering=True,
            normalize_audio=True,
            bass_boost=0,
            treble_boost=0,
            volume_boost=0,
        )
        assert dehum_filters
        assert declick_filters
        assert hiss_filters
        assert restoration_filters
        assert mastering_filters

        backend.current_file = str(wav_work)
        export_path = backend.export_current("mp3")
        assert_exists(export_path)
        return {"export": str(export_path)}
    finally:
        shutil.rmtree(temp_root, ignore_errors=True)


def test_roformer_resolution():
    from restoration_backend import RestorationStudioBackend

    backend = RestorationStudioBackend(ROOT)
    model = backend.resolve_roformer_model()
    return {"resolved_model": model, "label": backend.last_resolved_roformer_model}


def main():
    parser = argparse.ArgumentParser(description="Quick backend checks for the restoration backend.")
    parser.add_argument(
        "--suite",
        choices=["quick", "models", "all"],
        default="quick",
    )
    args = parser.parse_args()

    started = time.time()
    results = {
        "session_and_history": test_session_and_history(),
        "download_options_and_profiles": test_download_options_and_profiles(),
        "ai_analysis_and_recommendation": test_ai_analysis_and_recommendation(),
        "filters_and_export": test_filters_and_export(),
    }
    if args.suite in {"models", "all"}:
        results["roformer_resolution"] = test_roformer_resolution()

    print(json.dumps({
        "suite": args.suite,
        "seconds": round(time.time() - started, 2),
        "results": results,
    }, indent=2))


if __name__ == "__main__":
    main()
