"""
Audio signal generation and playback utilities.
"""

from __future__ import annotations

import math
import os
import sys
import tempfile
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

if sys.platform == "darwin":
    # Match xbox_test.py behaviour to avoid SDL/Tk conflicts on macOS.
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import numpy as np
from scipy.io import wavfile

try:
    from .signal import generate_tone_signal, generate_xbox_signal
except ImportError:
    # Allow running as a script by fixing sys.path
    repo_root = Path(__file__).resolve().parents[2]
    src_path = repo_root / "src"
    if src_path.exists():
        sys.path.insert(0, str(src_path))
    from preference_learning.audio.signal import generate_tone_signal, generate_xbox_signal
try:
    import pygame

    PYGAME_AVAILABLE = True
except Exception:
    PYGAME_AVAILABLE = False

try:
    import sounddevice as sd

    SOUNDDEVICE_AVAILABLE = True
except Exception:
    SOUNDDEVICE_AVAILABLE = False


ParameterRanges = Dict[str, Tuple[float, float]]

DEFAULT_OUTPUT_DEVICE = "xbox_controller"
OUTPUT_DEVICE_LABELS = {
    "bluetooth_vibrator": "Bluetooth vibrator / audio out",
    "xbox_controller": "Microsoft Xbox gamepad",
}


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def _norm_slider(val: float) -> float:
    # Expect slider-like values 20..100 -> 0..1
    return _clamp01((float(val) - 20.0) / 80.0)


def _map_intensity(slider_val: float) -> float:
    return 0.20 + _norm_slider(slider_val) * 0.80


def _map_balance_left(slider_val: float) -> float:
    return _norm_slider(slider_val)


def _map_rhythm_hz(slider_val: float) -> float:
    return 0.60 + _norm_slider(slider_val) * 3.40


def _map_grain_duty(slider_val: float) -> float:
    return 0.10 + _norm_slider(slider_val) * 0.60


def generate_xbox_rumble_segments(intensity_slider, texture_slider, rhythm_slider, grain_slider, duration_s):
    actual_intensity = _map_intensity(intensity_slider)
    left_share = _map_balance_left(texture_slider)
    actual_speed_hz = _map_rhythm_hz(rhythm_slider)
    actual_duty = _map_grain_duty(grain_slider)

    motor_left = actual_intensity * left_share
    motor_right = actual_intensity * (1.0 - left_share)

    kick_left = motor_left
    kick_right = motor_right

    cycle_ms = 1000.0 / actual_speed_hz
    physical_min_gap_ms = 45.0
    attack_ms = 20.0

    target_pulse_ms = cycle_ms * actual_duty
    max_pulse_ms_normal = max(20.0, cycle_ms - physical_min_gap_ms)
    actual_pulse_ms = min(target_pulse_ms, max_pulse_ms_normal)

    total_cycles = max(1, int(math.ceil(duration_s * actual_speed_hz)))

    segments = []
    current_time = 0.0

    for i in range(total_cycles):
        if current_time >= duration_s:
            break

        remaining_ms = (duration_s - current_time) * 1000.0
        if remaining_ms < actual_pulse_ms:
            break

        dur_1_ms = min(actual_pulse_ms, attack_ms)
        segments.append(
            {
                "type": "rumble",
                "start": current_time,
                "duration": dur_1_ms / 1000.0,
                "left": kick_left,
                "right": kick_right,
                "continuous_next": True,
            }
        )
        current_time += dur_1_ms / 1000.0

        dur_2_ms = actual_pulse_ms - dur_1_ms
        if dur_2_ms > 0:
            segments.append(
                {
                    "type": "rumble",
                    "start": current_time,
                    "duration": dur_2_ms / 1000.0,
                    "left": motor_left,
                    "right": motor_right,
                    "continuous_next": False,
                }
            )
            current_time += dur_2_ms / 1000.0
        else:
            segments[-1]["continuous_next"] = False

        next_cycle_start = (i + 1) * (cycle_ms / 1000.0)
        if i < total_cycles - 1 and next_cycle_start <= current_time:
            next_cycle_start = current_time + (physical_min_gap_ms / 1000.0)

        current_time = next_cycle_start

    return segments, duration_s


def _segments_to_waveform(segments, sample_rate: int, total_duration: float):
    total_time = float(max(total_duration, 0.001))
    if segments:
        max_end = max(seg["start"] + seg["duration"] for seg in segments)
        total_time = max(total_time, max_end)
    n_samples = max(int(total_time * sample_rate), 1)
    t = np.arange(n_samples) / float(sample_rate)
    data = np.zeros((n_samples, 2), dtype=float)
    for seg in segments:
        start_idx = int(seg["start"] * sample_rate)
        end_idx = int((seg["start"] + seg["duration"]) * sample_rate)
        end_idx = min(end_idx, n_samples)
        if start_idx >= n_samples or end_idx <= start_idx:
            continue
        data[start_idx:end_idx, 0] = float(seg.get("left", 0.0))
        data[start_idx:end_idx, 1] = float(seg.get("right", 0.0))
    return t, data


def precise_wait(target_time: float, stop_event: Optional[threading.Event] = None) -> bool:
    while True:
        if stop_event is not None and stop_event.is_set():
            return False
        now = time.perf_counter()
        dt = target_time - now
        if dt <= 0:
            return True
        if dt > 0.01:
            time.sleep(min(0.005, dt / 2))
        elif dt > 0.002:
            time.sleep(0.001)


def _safe_pump_events() -> None:
    """
    Pump pygame events on the main thread to avoid macOS AppKit crashes.
    """
    if not PYGAME_AVAILABLE:
        return
    if threading.current_thread() is not threading.main_thread():
        return
    try:
        pygame.event.pump()
    except Exception:
        pass


@dataclass
class AudioGenerator:
    """Generate and play audio signals for preference learning."""

    duration: int = 3
    cycles: int = 1
    sample_rate: int = 44100
    param_ranges: ParameterRanges = field(
        default_factory=lambda: {
            "amplitude": (20.0, 100.0),
            "frequency": (20.0, 100.0),
            "density": (20.0, 100.0),
            "gradient": (20.0, 100.0),
        }
    )

    def __post_init__(self) -> None:
        self.audio_backend: Optional[str] = None
        self.output_device: str = DEFAULT_OUTPUT_DEVICE
        self._xbox: Optional[object] = None
        self._xbox_thread: Optional[threading.Thread] = None
        self._xbox_stop_event = threading.Event()
        self._xbox_guard_token: int = 0
        self._xbox_lock = threading.Lock()

        if PYGAME_AVAILABLE:
            try:
                pygame.init()
            except Exception:
                pass
            try:
                pygame.mixer.init(frequency=self.sample_rate, size=-16, channels=1, buffer=1024)
                self.audio_backend = "pygame"
            except Exception:
                self.audio_backend = None
            try:
                pygame.joystick.init()
            except Exception:
                pass
            try:
                if hasattr(pygame, "controller"):
                    pygame.controller.init()
            except Exception:
                pass
        elif SOUNDDEVICE_AVAILABLE:
            self.audio_backend = "sounddevice"

    def _param_to_slider(self, key: str, value: float) -> float:
        bounds = self.param_ranges.get(key, (20.0, 100.0))
        low, high = float(bounds[0]), float(bounds[1])
        if high == low:
            return 20.0
        n = _clamp01((float(value) - low) / (high - low))
        return 20.0 + n * 80.0

    # ------------------------------------------------------------------ #
    # Signal synthesis
    # ------------------------------------------------------------------ #
    def generate_signal(self, amplitude: float, frequency: float, density: float, gradient: float):
        """Generate an audio signal with the provided parameters."""
        amplitude = np.clip(amplitude, *self.param_ranges["amplitude"])
        frequency = np.clip(frequency, *self.param_ranges["frequency"])
        density = np.clip(density, *self.param_ranges["density"])
        gradient = np.clip(gradient, *self.param_ranges["gradient"])

        if self.output_device == "xbox_controller":
            duration_s = float(self.duration)
            intensity_slider = self._param_to_slider("amplitude", amplitude)
            texture_slider = self._param_to_slider("frequency", frequency)
            rhythm_slider = self._param_to_slider("density", density)
            grain_slider = self._param_to_slider("gradient", gradient)

            segments, total_time = generate_xbox_rumble_segments(
                intensity_slider, texture_slider, rhythm_slider, grain_slider, duration_s
            )
            fs_plot = 200
            t_vec, data = _segments_to_waveform(segments, fs_plot, total_time)
            plot_waveform = np.mean(data, axis=1) if isinstance(data, np.ndarray) and data.ndim > 1 else data
            metadata = {
                "parameters": {
                    "amplitude": amplitude,
                    "frequency": frequency,
                    "density": density,
                    "gradient": gradient,
                },
                "duration": total_time,
                "fs": fs_plot,
                "segments": segments,
                "for_plot": {"plot_waveform": plot_waveform},
                "sliders": {
                    "intensity": intensity_slider,
                    "texture": texture_slider,
                    "rhythm": rhythm_slider,
                    "grain": grain_slider,
                },
            }
            return t_vec, data, metadata

        t_vec, data, for_plot = generate_tone_signal(
            filler_amplitude=amplitude,
            filler_frequency=frequency,
            filler_density=density,
            filler_env_gradient=gradient,
            duration=self.duration,
            cycles=self.cycles,
            fs=self.sample_rate,
        )

        max_abs = np.max(np.abs(data))
        if max_abs > 0:
            data = data / max_abs * 0.8

        metadata = {
            "parameters": {
                "amplitude": amplitude,
                "frequency": frequency,
                "density": density,
                "gradient": gradient,
            },
            "duration": self.duration,
            "fs": self.sample_rate,
            "for_plot": for_plot,
        }
        return t_vec, data, metadata

    # ------------------------------------------------------------------ #
    # Persistence helpers
    # ------------------------------------------------------------------ #
    def save_audio(self, data: np.ndarray, filename: Optional[str] = None) -> str:
        target = filename
        if target is None:
            temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            target = temp_file.name
            temp_file.close()

        audio_int16 = (data * 32767).astype(np.int16)
        wavfile.write(target, self.sample_rate, audio_int16)
        return target

    # ------------------------------------------------------------------ #
    # Playback
    # ------------------------------------------------------------------ #
    def play_audio(self, data: np.ndarray, metadata: Optional[Dict] = None, blocking: bool = True) -> bool:
        if self.output_device == "xbox_controller":
            # Ensure any previous rumble is stopped before starting a new one.
            self._stop_xbox_rumble()
            return self._play_xbox(data, metadata, blocking)

        if self.audio_backend is None:
            print("Warning: no audio backend available.")
            return False

        try:
            if self.audio_backend == "pygame":
                return self._play_pygame(data, blocking)
            if self.audio_backend == "sounddevice":
                return self._play_sounddevice(data, blocking)
        except Exception as exc:
            print(f"Audio playback error: {exc}")
            return False
        return False

    def set_output_device(self, device: str) -> str:
        raw = device.strip().lower()
        normalized = raw.replace(" ", "_")
        if normalized in OUTPUT_DEVICE_LABELS:
            self.output_device = normalized
        else:
            matched = None
            for key, label in OUTPUT_DEVICE_LABELS.items():
                if raw == label.lower() or key.replace("_", " ") in raw:
                    matched = key
                    break
            self.output_device = matched or DEFAULT_OUTPUT_DEVICE
        self.stop_audio()
        return self.output_device

    def _play_pygame(self, data: np.ndarray, blocking: bool) -> bool:
        temp_file = self.save_audio(data)
        try:
            pygame.mixer.music.load(temp_file)
            pygame.mixer.music.play()

            if blocking:
                while pygame.mixer.music.get_busy():
                    time.sleep(0.1)
            return True
        finally:
            try:
                os.unlink(temp_file)
            except Exception:
                pass

    def _play_sounddevice(self, data: np.ndarray, blocking: bool) -> bool:
        try:
            sd.play(data, self.sample_rate)
            if blocking:
                sd.wait()
            return True
        except Exception as exc:
            print(f"Sounddevice playback error: {exc}")
            return False

    def _ensure_xbox_controller(self) -> Optional[object]:
        if not PYGAME_AVAILABLE:
            return None
        try:
            _safe_pump_events()
            if not pygame.joystick.get_init():
                pygame.joystick.init()
            if self._xbox is not None and getattr(self._xbox, "get_init", lambda: False)():
                return self._xbox

            count = pygame.joystick.get_count()
            fallback = None
            for idx in range(count):
                js = pygame.joystick.Joystick(idx)
                if not js.get_init():
                    js.init()
                name = js.get_name().lower()
                if fallback is None:
                    fallback = js
                if "xbox" in name or "x-box" in name or "microsoft" in name:
                    self._xbox = js
                    return self._xbox

            if hasattr(pygame, "controller"):
                try:
                    if not pygame.controller.get_init():
                        pygame.controller.init()
                    for idx in range(pygame.controller.get_count()):
                        c = pygame.controller.Controller(idx)
                        name = c.get_name().lower()
                        if fallback is None:
                            fallback = c
                        if "xbox" in name or "microsoft" in name:
                            self._xbox = c
                            return self._xbox
                except Exception:
                    pass

            if fallback is not None:
                self._xbox = fallback
                return self._xbox
        except Exception as exc:
            print(f"Xbox controller init failed: {exc}")
            return None
        return None

    def _play_xbox(self, data: np.ndarray, metadata: Optional[Dict], blocking: bool) -> bool:
        controller = self._ensure_xbox_controller()
        if controller is None:
            return False
        if not hasattr(controller, "rumble"):
            print("Pygame build does not expose Joystick.rumble.")
            return False

        segments = None
        duration_s = float(self.duration)
        if metadata:
            segments = metadata.get("segments")
            duration_s = float(metadata.get("duration", duration_s))
        if isinstance(data, dict) and segments is None:
            segments = data.get("segments")

        if segments is not None:
            return self._play_xbox_segments(controller, segments, duration_s, blocking)

        if data is None:
            return False

        duration_s = max(duration_s, self.duration)
        self._xbox_stop_event.clear()
        play_token = self._xbox_guard_token + 1
        self._xbox_guard_token = play_token

        def runner() -> None:
            try:
                is_stereo = data.ndim == 2 and data.shape[1] == 2
                total_samples = len(data)
                if total_samples < 500:
                    step_ms = max(int(duration_s / max(total_samples, 1) * 1000), 20)
                else:
                    step_ms = 40
                effective_fps = 1000.0 / step_ms if step_ms > 0 else 25.0
                step_stride = max(int(total_samples / max(duration_s * effective_fps, 1)), 1)
                rumble_duration_ms = step_ms + 100
                deadline = time.monotonic() + duration_s + 0.5

                for i in range(0, total_samples, step_stride):
                    if self._xbox_stop_event.is_set() or time.monotonic() > deadline:
                        break
                    _safe_pump_events()
                    if is_stereo:
                        sample = data[i]
                        left = float(np.clip(sample[0], 0.0, 1.0))
                        right = float(np.clip(sample[1], 0.0, 1.0))
                    else:
                        segment = data[i : i + step_stride]
                        val = float(np.max(np.abs(segment))) if len(segment) > 0 else 0.0
                        left = right = val
                    try:
                        controller.rumble(left, right, rumble_duration_ms)
                    except Exception:
                        break
                    time.sleep(step_ms / 1000.0)
            except Exception as exc:
                print(f"Xbox loop error: {exc}")
            finally:
                self._stop_xbox_rumble(play_token)

        if blocking:
            runner()
            return True

        self._xbox_thread = threading.Thread(target=runner, daemon=True)
        self._xbox_thread.start()

        def guard() -> None:
            time.sleep(duration_s + 0.8)
            if self._xbox_guard_token == play_token:
                self._stop_xbox_rumble(play_token)

        threading.Thread(target=guard, daemon=True).start()
        return True

    def _play_xbox_segments(self, controller, segments, duration_s: float, blocking: bool) -> bool:
        self._xbox_stop_event.clear()
        play_token = self._xbox_guard_token + 1
        self._xbox_guard_token = play_token

        def runner() -> None:
            start_global = time.perf_counter()
            for seg in segments:
                if self._xbox_stop_event.is_set():
                    break
                t_start = start_global + float(seg.get("start", 0.0))
                seg_duration = float(seg.get("duration", 0.0))
                t_end = t_start + seg_duration
                if not precise_wait(t_start, self._xbox_stop_event):
                    break
                left = float(np.clip(seg.get("left", 0.0), 0.0, 1.0))
                right = float(np.clip(seg.get("right", 0.0), 0.0, 1.0))
                dur_ms = max(0, int(seg_duration * 1000))
                if seg.get("continuous_next", False):
                    dur_ms += 20
                try:
                    with self._xbox_lock:
                        controller.rumble(left, right, dur_ms)
                except Exception:
                    break
                if not precise_wait(t_end, self._xbox_stop_event):
                    break

            if not self._xbox_stop_event.is_set():
                precise_wait(start_global + duration_s, self._xbox_stop_event)

            try:
                _safe_pump_events()
                with self._xbox_lock:
                    if hasattr(controller, "stop_rumble"):
                        controller.stop_rumble()
            except Exception:
                pass

        if blocking:
            runner()
            return True

        self._xbox_thread = threading.Thread(target=runner, daemon=True)
        self._xbox_thread.start()

        def guard() -> None:
            time.sleep(duration_s + 0.8)
            if self._xbox_guard_token == play_token:
                self._stop_xbox_rumble(play_token)

        threading.Thread(target=guard, daemon=True).start()
        return True

    def stop_audio(self) -> None:
        try:
            if self.output_device == "xbox_controller":
                self._stop_xbox_rumble()
            elif self.audio_backend == "pygame":
                pygame.mixer.music.stop()
            elif self.audio_backend == "sounddevice":
                sd.stop()
        except Exception:
            pass

    def _stop_xbox_rumble(self, expected_token: Optional[int] = None) -> None:
        """Stop Xbox vibration (simple stop)."""
        if expected_token is not None and expected_token != self._xbox_guard_token:
            return
        self._xbox_stop_event.set()
        current = threading.current_thread()
        if self._xbox_thread and self._xbox_thread.is_alive() and self._xbox_thread is not current:
            self._xbox_thread.join(timeout=0.5)
        self._xbox_thread = None
        self._xbox_guard_token += 1

        controller = self._xbox
        if controller is None:
            return

        try:
            _safe_pump_events()
            with self._xbox_lock:
                if hasattr(controller, "stop_rumble"):
                    controller.stop_rumble()
        except Exception:
            pass

    def close(self) -> None:
        try:
            self.stop_audio()
            if self.audio_backend == "pygame":
                pygame.mixer.quit()
            elif self.audio_backend == "sounddevice":
                sd.stop()
        except Exception:
            pass

    # ------------------------------------------------------------------ #
    # Feature extraction
    # ------------------------------------------------------------------ #
    def calculate_audio_features(self, data: np.ndarray) -> Dict[str, float]:
        if len(data) == 0:
            return {"rms": 0.0, "peak": 0.0, "energy": 0.0}
        features = {
            "rms": float(np.sqrt(np.mean(data**2))),
            "peak": float(np.max(np.abs(data))),
            "energy": float(np.sum(data**2)),
        }
        return features


if __name__ == "__main__":
    print("AudioGenerator module is part of the preference_learning package.")
    print("Run it via the package (python -m preference_learning.interface.ui_study)")
    print("or import AudioGenerator from preference_learning.audio.generator.")
