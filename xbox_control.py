import argparse
import json
import tkinter as tk
from datetime import datetime
from pathlib import Path
from tkinter import ttk, messagebox
import pygame
import sys
import threading
import time
import math
import os

# --- Matplotlib ---
import matplotlib
import matplotlib.style

matplotlib.use("TkAgg")
matplotlib.style.use("default")
matplotlib.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.unicode_minus": False,
    }
)
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

try:
    from PIL import Image, ImageTk
except Exception:
    Image = None
    ImageTk = None

# --- Colors & Styles (match study UI) ---
COLOR_BG = "#f3f4f8"
COLOR_PANEL = "#ffffff"
COLOR_ACCENT = "#3b82f6"
COLOR_ACCENT_HOVER = "#2563eb"
COLOR_TEXT = "#1f2933"
COLOR_MUTED = "#6b7280"
COLOR_BORDER = "#e5e7eb"


# ==========================================
# Mapping: UI slider (20..100) -> real units
# ==========================================
def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def norm_20_100(x: float) -> float:
    # 20 -> 0.0, 100 -> 1.0
    return _clamp01((float(x) - 20.0) / 80.0)


def map_intensity(slider_val: float) -> float:
    # 0.20 .. 1.00
    n = norm_20_100(slider_val)
    return 0.20 + n * 0.80


def map_balance_left(slider_val: float) -> float:
    # 0.00 .. 1.00 (left share)
    return norm_20_100(slider_val)


def map_rhythm_hz(slider_val: float) -> float:
    # 0.60 .. 4.00 Hz
    n = norm_20_100(slider_val)
    return 0.60 + n * 3.40


def map_grain_duty(slider_val: float) -> float:
    # 10% .. 70%
    n = norm_20_100(slider_val)
    return 0.10 + n * 0.60


# ==========================================
# Core signal generator (segment-based)
# Rules:
#  - If the last pulse (non-zero part) can't fully finish within duration, drop it.
# ==========================================
def generate_xbox_rumble_segments(intensity_slider, texture_slider, rhythm_slider, grain_slider, duration_s):
    # --- 1) Map to real values ---
    actual_intensity = map_intensity(intensity_slider)      # 0.20..1.00
    a = map_balance_left(texture_slider)                    # left share 0..1
    actual_speed_hz = map_rhythm_hz(rhythm_slider)          # 0.60..4.00
    actual_duty = map_grain_duty(grain_slider)              # 0.10..0.70

    motor_left = actual_intensity * a
    motor_right = actual_intensity * (1.0 - a)

    # "Kick" is just the first stage values (no extra boost)
    kick_left = motor_left
    kick_right = motor_right

    # --- 2) Timing ---
    cycle_ms = 1000.0 / actual_speed_hz
    PHYSICAL_MIN_GAP_MS = 45.0
    ATTACK_MS = 20.0

    target_pulse_ms = cycle_ms * actual_duty
    max_pulse_ms_normal = max(20.0, cycle_ms - PHYSICAL_MIN_GAP_MS)
    actual_pulse_ms = min(target_pulse_ms, max_pulse_ms_normal)
    # NOTE: no "min 25ms" clamp (per your request)

    total_cycles = max(1, int(math.ceil(duration_s * actual_speed_hz)))

    segments = []
    current_time = 0.0

    for i in range(total_cycles):
        if current_time >= duration_s:
            break

        remaining_ms = (duration_s - current_time) * 1000.0

        # ✅ Rule: if the remaining time can't fit the full non-zero pulse, drop the last pulse.
        if remaining_ms < actual_pulse_ms:
            break

        # Stage 1: Kick (attack)
        dur_1_ms = min(actual_pulse_ms, ATTACK_MS)

        segments.append({
            "type": "rumble",
            "start": current_time,
            "duration": dur_1_ms / 1000.0,
            "left": kick_left,
            "right": kick_right,
            "continuous_next": True,
        })
        current_time += dur_1_ms / 1000.0

        # Stage 2: Sustain
        dur_2_ms = actual_pulse_ms - dur_1_ms
        if dur_2_ms > 0:
            segments.append({
                "type": "rumble",
                "start": current_time,
                "duration": dur_2_ms / 1000.0,
                "left": motor_left,
                "right": motor_right,
                "continuous_next": False,
            })
            current_time += dur_2_ms / 1000.0
        else:
            segments[-1]["continuous_next"] = False

        # Stage 3: Gap (align to next cycle; enforce minimum gap if needed)
        next_cycle_start = (i + 1) * (cycle_ms / 1000.0)
        if i < total_cycles - 1 and next_cycle_start <= current_time:
            next_cycle_start = current_time + (PHYSICAL_MIN_GAP_MS / 1000.0)

        current_time = next_cycle_start

    return segments, duration_s


# ==========================================
# Timing helper
# ==========================================
def precise_wait(target_time, stop_event=None):
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
        else:
            pass


# ==========================================
# App
# ==========================================
class XboxVibrationApp:
    def __init__(
        self,
        root,
        output_dir=None,
        output_filename="favorite_signal.json",
        complete_on_record: bool = False,
        max_records: int = 3,
    ):
        self.root = root
        self.root.title("Haptic Preference Learning")
        self.root.geometry("1120x840")
        self.root.configure(bg=COLOR_BG)

        self.closing = False
        self.joystick = None
        self.joystick_name = tk.StringVar(value="Controller: not detected")

        self.is_playing = False
        self.play_thread = None
        self.current_stop_event = None
        self.current_session_id = 0
        self.joy_lock = threading.Lock()
        self._icons = {}

        self.output_dir = Path(output_dir) if output_dir else None
        self.output_filename = output_filename
        self.complete_on_record = complete_on_record
        self.max_records = max(1, int(max_records))
        self.record_count = 0
        self.record_paths = []
        self.record_status_var = tk.StringVar(value=f"Favorites recorded: 0/{self.max_records}")

        # live preview debounce
        self._preview_after_id = None

        if sys.platform == "darwin":
            os.environ["SDL_VIDEODRIVER"] = "dummy"

        try:
            pygame.init()
            pygame.joystick.init()
        except Exception as e:
            print("pygame init failed:", repr(e))
            sys.exit(1)

        self._setup_layout()
        self.setup_matplotlib(self.graph_frame)

        self.refresh_controllers()
        self.poll_pygame_events()

        # ✅ initial live preview
        self.update_preview_now()

    def _load_icon(self, name: str, max_height: int = 60):
        if name in self._icons:
            return self._icons[name]
        try:
            assets_dir = (
                Path(__file__).resolve().parent / "src" / "preference_learning" / "interface" / "logo"
            )
            for ext in [".png", ".jpg", ""]:
                path = assets_dir / (name + ext)
                if path.exists():
                    if Image:
                        pil_img = Image.open(path).convert("RGBA")
                        ratio = max_height / pil_img.height
                        new_w = int(pil_img.width * ratio)
                        pil_img = pil_img.resize((new_w, max_height), Image.LANCZOS)
                        icon = ImageTk.PhotoImage(pil_img)
                        self._icons[name] = icon
                        return icon
                    break
        except Exception:
            pass
        return None

    def _make_button(self, parent, text, command, primary=False, width=None):
        bg = COLOR_ACCENT if primary else COLOR_PANEL
        fg = "#ffffff" if primary else COLOR_TEXT
        active_bg = COLOR_ACCENT_HOVER if primary else COLOR_BORDER
        btn = tk.Button(
            parent,
            text=text,
            command=command,
            bg=bg,
            fg=fg,
            activebackground=active_bg,
            activeforeground=fg,
            relief="flat",
            bd=0,
            highlightthickness=1,
            highlightbackground=COLOR_BORDER,
            font=("Helvetica", 11, "bold"),
            disabledforeground=COLOR_MUTED,
            width=width,
        )
        return btn

    def _make_card(self, parent, title: str):
        frame = tk.LabelFrame(
            parent,
            text=f" {title} ",
            bg=COLOR_PANEL,
            fg=COLOR_MUTED,
            font=("Helvetica", 10, "bold"),
            bd=1,
            relief="solid",
        )
        frame.pack(fill=tk.X, pady=10)
        return frame

    def _setup_layout(self):
        header = tk.Frame(self.root, bg=COLOR_BG)
        header.pack(fill=tk.X, padx=20, pady=(10, 5))

        title_frame = tk.Frame(header, bg=COLOR_BG)
        title_frame.pack(side=tk.LEFT, anchor="w")
        tk.Label(
            title_frame,
            text="Haptic Preference Learning",
            fg=COLOR_TEXT,
            bg=COLOR_BG,
            font=("Helvetica", 22, "bold"),
        ).pack(anchor="w")
        tk.Label(
            title_frame,
            text="Favorite Signal Tuning",
            fg=COLOR_MUTED,
            bg=COLOR_BG,
            font=("Helvetica", 12),
        ).pack(anchor="w", pady=(2, 0))

        brand_frame = tk.Frame(header, bg=COLOR_BG)
        brand_frame.pack(side=tk.RIGHT)
        usc_icon = self._load_icon("USC", 56)
        harvi_icon = self._load_icon("harvi", 56)
        if usc_icon:
            tk.Label(brand_frame, image=usc_icon, bg=COLOR_BG).pack(side=tk.LEFT, padx=8)
        else:
            tk.Label(
                brand_frame,
                text="USC",
                fg=COLOR_TEXT,
                bg=COLOR_BG,
                font=("Helvetica", 18, "bold"),
            ).pack(side=tk.LEFT, padx=8)
        if harvi_icon:
            tk.Label(brand_frame, image=harvi_icon, bg=COLOR_BG).pack(side=tk.LEFT, padx=8)
        else:
            tk.Label(
                brand_frame,
                text="HARVI",
                fg=COLOR_TEXT,
                bg=COLOR_BG,
                font=("Helvetica", 18, "bold"),
            ).pack(side=tk.LEFT, padx=8)

        content = tk.Frame(self.root, bg=COLOR_BG)
        content.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)

        left_panel = tk.Frame(content, bg=COLOR_BG, width=380)
        left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 20))
        left_panel.pack_propagate(False)

        right_panel = tk.Frame(content, bg=COLOR_BG)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        status_card = self._make_card(left_panel, "Device Status")
        status_inner = tk.Frame(status_card, bg=COLOR_PANEL)
        status_inner.pack(fill=tk.X, padx=12, pady=10)
        tk.Label(
            status_inner,
            textvariable=self.joystick_name,
            bg=COLOR_PANEL,
            fg=COLOR_TEXT,
            font=("Helvetica", 11),
        ).pack(side=tk.LEFT)
        self.btn_refresh = self._make_button(status_inner, "Refresh", self.refresh_controllers, width=10)
        self.btn_refresh.pack(side=tk.RIGHT)

        params_card = self._make_card(left_panel, "Parameters (slider range: 20–100)")
        self.vars = {}
        self.scales = {}

        self.create_slider(
            params_card,
            "Intensity",
            "intensity",
            20,
            100,
            50,
            display_fn=lambda v: f"{map_intensity(v):.2f} (0.20–1.00)",
        )
        self.create_slider(
            params_card,
            "Texture / Balance",
            "texture",
            20,
            100,
            50,
            display_fn=lambda v: f"Left {map_balance_left(v)*100:.0f}% / Right {(1-map_balance_left(v))*100:.0f}%",
        )
        self.create_slider(
            params_card,
            "Rhythm",
            "rhythm",
            20,
            100,
            50,
            display_fn=lambda v: f"{map_rhythm_hz(v):.2f} Hz (0.60–4.00)",
        )
        self.create_slider(
            params_card,
            "Grain",
            "grain",
            20,
            100,
            50,
            display_fn=lambda v: f"{map_grain_duty(v)*100:.0f}% duty (10–70%)",
        )

        ttk.Separator(params_card, orient="horizontal").pack(fill="x", pady=10, padx=8)

        self.create_slider(
            params_card,
            "Duration",
            "duration_ms",
            500,
            5000,
            3000,
            display_fn=lambda v: f"{int(float(v))} ms ({float(v)/1000.0:.2f} s)",
            disabled=True,
        )

        action_card = self._make_card(left_panel, "Actions")
        action_inner = tk.Frame(action_card, bg=COLOR_PANEL)
        action_inner.pack(fill=tk.X, padx=12, pady=10)
        self.btn_play = self._make_button(action_inner, "▶ Play", self.start_vibration, primary=True)
        self.btn_play.pack(fill=tk.X, ipady=6)
        self.btn_stop = self._make_button(action_inner, "■ Stop", self.stop_vibration_ui)
        self.btn_stop.pack(fill=tk.X, pady=6)
        self.btn_stop.config(state="disabled")
        self.btn_record = self._make_button(
            action_inner,
            f"Record Favorite (1/{self.max_records})",
            self.record_params,
        )
        self.btn_record.pack(fill=tk.X)
        self.record_status_label = tk.Label(
            action_inner,
            textvariable=self.record_status_var,
            bg=COLOR_PANEL,
            fg=COLOR_MUTED,
            font=("Helvetica", 10),
            anchor="w",
        )
        self.record_status_label.pack(fill=tk.X, pady=(6, 0))
        self._update_record_ui()

        spacer = tk.Frame(left_panel, bg=COLOR_BG)
        spacer.pack(fill=tk.BOTH, expand=True)

        notes_card = self._make_card(left_panel, "Parameter Notes")
        notes_inner = tk.Frame(notes_card, bg=COLOR_PANEL)
        notes_inner.pack(fill=tk.X, padx=12, pady=10)
        for line in (
            "Intensity: overall vibration strength; higher feels stronger.",
            "Texture/Balance: left-right motor mix; shifts where the vibration feels.",
            "Rhythm: how fast the pulses repeat; higher feels faster.",
            "Grain: how long each pulse stays on; higher feels fuller and smoother.",
        ):
            tk.Label(
                notes_inner,
                text=line,
                bg=COLOR_PANEL,
                fg=COLOR_TEXT,
                font=("Helvetica", 10),
                anchor="w",
                justify="left",
            ).pack(anchor="w")

        self.graph_frame = tk.LabelFrame(
            right_panel,
            text=" Preview (Live) ",
            bg=COLOR_PANEL,
            fg=COLOR_MUTED,
            font=("Helvetica", 10, "bold"),
            bd=1,
            relief="solid",
        )
        self.graph_frame.pack(fill=tk.BOTH, expand=True)

    # -----------------------
    # Live preview
    # -----------------------
    def schedule_preview_update(self, delay_ms=40):
        if self._preview_after_id is not None:
            try:
                self.root.after_cancel(self._preview_after_id)
            except Exception:
                pass
        self._preview_after_id = self.root.after(delay_ms, self.update_preview_now)

    def get_current_params(self):
        intensity = self.vars["intensity"].get()
        texture = self.vars["texture"].get()
        rhythm = self.vars["rhythm"].get()
        grain = self.vars["grain"].get()
        duration_s = self.vars["duration_ms"].get() / 1000.0
        return intensity, texture, rhythm, grain, duration_s

    def update_preview_now(self):
        self._preview_after_id = None
        intensity, texture, rhythm, grain, duration_s = self.get_current_params()
        segments, total_time = generate_xbox_rumble_segments(intensity, texture, rhythm, grain, duration_s)
        self.update_graph(segments, total_time)

    # -----------------------
    # Record helpers
    # -----------------------
    def _build_record_payload(self):
        intensity, texture, rhythm, grain, duration_s = self.get_current_params()
        left_share = map_balance_left(texture)
        payload = {
            "timestamp": datetime.now().isoformat(),
            "slider": {
                "intensity": float(intensity),
                "texture": float(texture),
                "rhythm": float(rhythm),
                "grain": float(grain),
                "duration_ms": int(round(duration_s * 1000)),
            },
            "signal": {
                "intensity": float(map_intensity(intensity)),
                "left_share": float(left_share),
                "right_share": float(1.0 - left_share),
                "speed_hz": float(map_rhythm_hz(rhythm)),
                "duty": float(map_grain_duty(grain)),
                "duration_s": float(duration_s),
            },
        }
        return payload

    def _next_index_path(self, base_dir: Path) -> Path:
        max_idx = 0
        for path in base_dir.glob("*.json"):
            if path.stem.isdigit():
                try:
                    max_idx = max(max_idx, int(path.stem))
                except ValueError:
                    continue
        return base_dir / f"{max_idx + 1:03d}.json"

    def _indexed_filename(self, filename: str, record_index: int) -> str:
        if record_index <= 1:
            return Path(filename).name
        stem = Path(filename).stem
        suffix = Path(filename).suffix or ".json"
        return f"{stem}_{record_index:02d}{suffix}"

    def _unique_named_path(self, base_dir: Path, filename: str) -> Path:
        name = Path(filename).name
        stem = Path(name).stem
        suffix = Path(name).suffix or ".json"
        candidate = base_dir / f"{stem}{suffix}"
        if not candidate.exists():
            return candidate
        for idx in range(1, 1000):
            alt = base_dir / f"{stem}_{idx:02d}{suffix}"
            if not alt.exists():
                return alt
        return base_dir / f"{stem}_{int(time.time())}{suffix}"

    def _resolve_output_path(self, record_index: int) -> Path:
        if self.output_dir is not None:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            name = self._indexed_filename(self.output_filename, record_index)
            candidate = self.output_dir / name
            if candidate.exists():
                return self._unique_named_path(self.output_dir, name)
            return candidate
        base_dir = Path.cwd() / "data" / "bestparam"
        base_dir.mkdir(parents=True, exist_ok=True)
        return self._next_index_path(base_dir)

    def record_params(self):
        try:
            if self.record_count >= self.max_records:
                return
            payload = self._build_record_payload()
            record_index = self.record_count + 1
            payload["favorite_index"] = int(record_index)
            payload["favorite_total"] = int(self.max_records)
            out_path = self._resolve_output_path(record_index)
            out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            self.record_count += 1
            self.record_paths.append(str(out_path))
            self._update_record_ui()
            if self.record_count >= self.max_records:
                if self.complete_on_record:
                    messagebox.showinfo(
                        "Experiment complete",
                        "Experiment complete.",
                    )
                    self._close_window(exit_process=False)
                else:
                    messagebox.showinfo(
                        "Recorded",
                        f"Recorded {self.record_count}/{self.max_records}.",
                    )
            else:
                if not self.complete_on_record:
                    messagebox.showinfo(
                        "Recorded",
                        f"Recorded {self.record_count}/{self.max_records}.",
                    )
        except Exception as exc:
            messagebox.showerror("Error", f"Failed to record:\n{exc}")

    def _update_record_ui(self) -> None:
        next_idx = min(self.record_count + 1, self.max_records)
        self.record_status_var.set(
            f"Favorites recorded: {self.record_count}/{self.max_records}"
        )
        self.btn_record.config(
            text=f"Record Favorite ({next_idx}/{self.max_records})",
            state="disabled" if self.record_count >= self.max_records else "normal",
        )

    # -----------------------
    # UI helpers
    # -----------------------
    def create_slider(self, parent, label, var_name, min_v, max_v, def_v, display_fn, disabled=False):
        f = tk.Frame(parent, bg=COLOR_PANEL)
        f.pack(fill="x", pady=6, padx=8)

        tk.Label(
            f,
            text=label,
            bg=COLOR_PANEL,
            fg=COLOR_TEXT,
            font=("Helvetica", 11, "bold"),
            width=18,
            anchor="w",
        ).pack(side="left")

        self.vars[var_name] = tk.DoubleVar(value=def_v)

        value_label = tk.Label(
            f,
            text=display_fn(def_v),
            bg=COLOR_PANEL,
            fg=COLOR_MUTED,
            font=("Helvetica", 10),
        )
        value_label.pack(side="right")

        def on_change(v):
            try:
                value_label.config(text=display_fn(float(v)))
            except Exception:
                value_label.config(text=str(v))
            # ✅ live preview update while dragging
            self.schedule_preview_update()

        scale = tk.Scale(
            f,
            from_=min_v,
            to=max_v,
            variable=self.vars[var_name],
            command=on_change,
            orient="horizontal",
            showvalue=0,
            bg=COLOR_PANEL,
            troughcolor=COLOR_BORDER,
            activebackground=COLOR_ACCENT_HOVER,
            highlightthickness=0,
            relief="flat",
            sliderlength=18,
        )
        if disabled:
            scale.configure(state="disabled")
        scale.pack(fill="x")
        self.scales[var_name] = scale

        on_change(def_v)

    def setup_matplotlib(self, parent):
        self.fig = Figure(figsize=(6, 4), dpi=100)
        self.fig.patch.set_facecolor(COLOR_PANEL)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_facecolor(COLOR_PANEL)
        self.canvas = FigureCanvasTkAgg(self.fig, master=parent)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

    def update_graph(self, segments, total_duration):
        self.ax.clear()
        self.ax.set_facecolor(COLOR_PANEL)

        # ✅ axis labels
        self.ax.set_xlabel("Time (s)", color=COLOR_TEXT)
        self.ax.set_ylabel("Rumble Intensity (0–1)", color=COLOR_TEXT)
        self.ax.tick_params(colors=COLOR_TEXT)

        if not segments:
            self.ax.set_xlim(0, max(1.0, total_duration))
            self.ax.set_ylim(-0.05, 1.1)
            self.ax.grid(True, alpha=0.3)
            self.canvas.draw()
            return

        times, lefts, rights = [0.0], [0.0], [0.0]
        times.append(segments[0]["start"])
        lefts.append(0.0)
        rights.append(0.0)

        for seg in segments:
            start = seg["start"]
            end = start + seg["duration"]
            l_val = seg["left"]
            r_val = seg["right"]

            if start > times[-1] + 0.0001:
                times.append(start); lefts.append(0.0); rights.append(0.0)

            times.append(start); lefts.append(l_val); rights.append(r_val)
            times.append(end);   lefts.append(l_val); rights.append(r_val)

            if not seg.get("continuous_next", False):
                times.append(end); lefts.append(0.0); rights.append(0.0)

        self.ax.plot(times, lefts, label="Left", linewidth=2, color=COLOR_ACCENT)
        self.ax.plot(times, rights, label="Right", linewidth=2, alpha=0.7, color="#f59e0b")

        self.ax.set_xlim(0, max(total_duration, times[-1] if times else 1.0))
        self.ax.set_ylim(-0.05, 1.1)
        self.ax.legend(loc="upper right", frameon=False)
        self.ax.grid(True, alpha=0.3)
        self.canvas.draw()

    # -----------------------
    # Device / events
    # -----------------------
    def poll_pygame_events(self):
        if pygame.get_init():
            pygame.event.pump()
        if not self.closing:
            self.root.after(50, self.poll_pygame_events)

    def refresh_controllers(self):
        if self.is_playing:
            messagebox.showinfo("Info", "Stop playback before refreshing the device.")
            return
        with self.joy_lock:
            try:
                pygame.joystick.quit()
                pygame.joystick.init()
            except Exception as e:
                print("refresh joystick failed:", repr(e))

            if pygame.joystick.get_count() > 0:
                try:
                    js = pygame.joystick.Joystick(0)
                    js.init()
                    self.joystick = js
                    self.joystick_name.set(f"Controller: {self.joystick.get_name()}")
                except Exception as e:
                    self.joystick = None
                    self.joystick_name.set("Controller: init failed")
                    print("joystick init failed:", repr(e))
            else:
                self.joystick_name.set("Controller: not connected")
                self.joystick = None

    # -----------------------
    # Playback control
    # -----------------------
    def start_vibration(self):
        if not self.joystick:
            self.refresh_controllers()
            if not self.joystick:
                return

        self._stop_current_playback(wait=True)

        intensity, texture, rhythm, grain, duration_s = self.get_current_params()
        segments, total_time = generate_xbox_rumble_segments(intensity, texture, rhythm, grain, duration_s)

        # keep preview consistent at play time
        self.update_graph(segments, total_time)

        self.is_playing = True
        self.btn_play.config(state="disabled")
        self.btn_stop.config(state="normal")
        self.btn_refresh.config(state="disabled")

        self.current_stop_event = threading.Event()
        self.current_session_id += 1

        self.play_thread = threading.Thread(
            target=self._playback_loop,
            args=(segments, self.current_stop_event, self.current_session_id, duration_s),
            daemon=True,
        )
        self.play_thread.start()

    def stop_vibration_ui(self):
        self._stop_current_playback(wait=False)

    def _stop_current_playback(self, wait=False):
        if self.current_stop_event:
            self.current_stop_event.set()
        with self.joy_lock:
            try:
                if self.joystick:
                    self.joystick.stop_rumble()
            except:
                pass
        if wait and self.play_thread and self.play_thread.is_alive():
            self.play_thread.join(0.2)
        self._set_ui_idle()

    def _set_ui_idle(self):
        if self.is_playing:
            self.is_playing = False
            self.btn_play.config(state="normal")
            self.btn_stop.config(state="disabled")
            self.btn_refresh.config(state="normal")

    def _playback_loop(self, segments, stop_event, session_id, duration_s):
        start_global = time.perf_counter()

        for seg in segments:
            if stop_event.is_set():
                break

            t_start = start_global + float(seg["start"])
            t_end = t_start + float(seg["duration"])

            if not precise_wait(t_start, stop_event):
                break
            if stop_event.is_set():
                break

            with self.joy_lock:
                if self.joystick:
                    dur_ms = max(0, int(float(seg["duration"]) * 1000))
                    # continuity padding (kept)
                    if seg.get("continuous_next", False):
                        dur_ms += 20
                    try:
                        self.joystick.rumble(float(seg["left"]), float(seg["right"]), dur_ms)
                    except:
                        break

            if not precise_wait(t_end, stop_event):
                break

        if not stop_event.is_set():
            precise_wait(start_global + duration_s, stop_event)

        with self.joy_lock:
            try:
                if self.joystick:
                    self.joystick.stop_rumble()
            except:
                pass

        if not self.closing:
            try:
                self.root.after(0, lambda: self._on_finish(session_id))
            except:
                pass

    def _on_finish(self, sid):
        if sid == self.current_session_id:
            self._set_ui_idle()

    def _close_window(self, exit_process: bool = True):
        if self.closing:
            return
        self.closing = True
        self._stop_current_playback(wait=True)
        try:
            pygame.quit()
        except:
            pass
        try:
            self.root.destroy()
        except:
            pass
        if exit_process:
            sys.exit()

    def on_close(self):
        self._close_window(exit_process=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Xbox controller haptics tuner.")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save the recorded parameters (optional).",
    )
    parser.add_argument(
        "--filename",
        type=str,
        default="favorite_signal.json",
        help="Filename when --output-dir is provided.",
    )
    parser.add_argument(
        "--complete-dialog",
        action="store_true",
        help="Show completion dialog after recording and close the window.",
    )
    args = parser.parse_args(argv)

    root = tk.Tk()
    app = XboxVibrationApp(
        root,
        output_dir=args.output_dir,
        output_filename=args.filename,
        complete_on_record=args.complete_dialog,
    )
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
