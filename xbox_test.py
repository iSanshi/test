import tkinter as tk
from tkinter import ttk, messagebox
import pygame
import sys
import threading
import time
import math
import os

# --- Matplotlib ---
import matplotlib
matplotlib.use("TkAgg")
matplotlib.rcParams["font.sans-serif"] = ["PingFang SC", "Microsoft YaHei", "SimHei", "Arial Unicode MS"]
matplotlib.rcParams["axes.unicode_minus"] = False
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


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
    def __init__(self, root):
        self.root = root
        self.root.title("Xbox Haptics Lab")
        self.root.geometry("600x900")

        self.closing = False
        self.joystick = None
        self.joystick_name = tk.StringVar(value="Controller: not detected")

        self.is_playing = False
        self.play_thread = None
        self.current_stop_event = None
        self.current_session_id = 0
        self.joy_lock = threading.Lock()

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

        style = ttk.Style()
        style.configure("Param.TLabel", font=("Arial", 10, "bold"))

        status_frame = ttk.LabelFrame(root, text="Device Status", padding=10)
        status_frame.pack(fill="x", padx=15, pady=10)
        ttk.Label(status_frame, textvariable=self.joystick_name).pack(side="left")
        self.btn_refresh = ttk.Button(status_frame, text="Refresh", command=self.refresh_controllers)
        self.btn_refresh.pack(side="right")

        control_frame = ttk.LabelFrame(root, text="Parameters (slider range: 20–100)", padding=15)
        control_frame.pack(fill="x", padx=15, pady=5)

        self.vars = {}

        # ✅ Four parameters: all 20..100, labels show REAL units
        self.create_slider(
            control_frame,
            "Intensity",
            "intensity",
            20, 100, 50,
            display_fn=lambda v: f"{map_intensity(v):.2f} (0.20–1.00)"
        )
        self.create_slider(
            control_frame,
            "Texture / Balance",
            "texture",
            20, 100, 50,
            display_fn=lambda v: f"Left {map_balance_left(v)*100:.0f}% / Right {(1-map_balance_left(v))*100:.0f}%"
        )
        self.create_slider(
            control_frame,
            "Rhythm",
            "rhythm",
            20, 100, 50,
            display_fn=lambda v: f"{map_rhythm_hz(v):.2f} Hz (0.60–4.00)"
        )
        self.create_slider(
            control_frame,
            "Grain",
            "grain",
            20, 100, 50,
            display_fn=lambda v: f"{map_grain_duty(v)*100:.0f}% duty (10–70%)"
        )

        ttk.Separator(control_frame, orient="horizontal").pack(fill="x", pady=10)

        # Duration stays in ms
        self.create_slider(
            control_frame,
            "Duration",
            "duration_ms",
            500, 5000, 3000,
            display_fn=lambda v: f"{int(float(v))} ms ({float(v)/1000.0:.2f} s)"
        )

        action_frame = ttk.Frame(root, padding=15)
        action_frame.pack(fill="x", side="bottom")
        self.btn_play = ttk.Button(action_frame, text="▶ Play", command=self.start_vibration)
        self.btn_play.pack(fill="x", ipady=5)
        self.btn_stop = ttk.Button(action_frame, text="■ Stop", command=self.stop_vibration_ui, state="disabled")
        self.btn_stop.pack(fill="x", pady=5)

        graph_frame = ttk.LabelFrame(root, text="Preview (Live)", padding=5)
        graph_frame.pack(fill="both", expand=True, padx=15, pady=5)
        self.setup_matplotlib(graph_frame)

        self.refresh_controllers()
        self.poll_pygame_events()

        # ✅ initial live preview
        self.update_preview_now()

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
    # UI helpers
    # -----------------------
    def create_slider(self, parent, label, var_name, min_v, max_v, def_v, display_fn):
        f = ttk.Frame(parent)
        f.pack(fill="x", pady=5)

        ttk.Label(f, text=label, style="Param.TLabel", width=18).pack(side="left")

        self.vars[var_name] = tk.DoubleVar(value=def_v)

        value_label = ttk.Label(f, text=display_fn(def_v))
        value_label.pack(side="right")

        def on_change(v):
            try:
                value_label.config(text=display_fn(float(v)))
            except Exception:
                value_label.config(text=str(v))
            # ✅ live preview update while dragging
            self.schedule_preview_update()

        ttk.Scale(
            f,
            from_=min_v, to=max_v,
            variable=self.vars[var_name],
            command=on_change
        ).pack(fill="x")

        on_change(def_v)

    def setup_matplotlib(self, parent):
        self.fig = Figure(figsize=(5, 3), dpi=100)
        self.fig.patch.set_facecolor("#F0F0F0")
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=parent)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

    def update_graph(self, segments, total_duration):
        self.ax.clear()

        # ✅ axis labels
        self.ax.set_xlabel("Time (s)")
        self.ax.set_ylabel("Rumble Intensity (0–1)")

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

        self.ax.plot(times, lefts, label="Left", linewidth=2)
        self.ax.plot(times, rights, label="Right", linewidth=2, alpha=0.7)

        self.ax.set_xlim(0, max(total_duration, times[-1] if times else 1.0))
        self.ax.set_ylim(-0.05, 1.1)
        self.ax.legend(loc="upper right")
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

    def on_close(self):
        self.closing = True
        self._stop_current_playback(wait=True)
        try:
            pygame.quit()
        except:
            pass
        self.root.destroy()
        sys.exit()


if __name__ == "__main__":
    root = tk.Tk()
    app = XboxVibrationApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()
