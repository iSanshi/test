#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Gamepad-first Interface for Haptic Preference Learning.
Redesigned to support full navigation via game controller with optimized grid logic.
"""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk, messagebox
from typing import Optional, Sequence, Callable, List, Tuple
import threading
import time
import numpy as np
import os
import multiprocessing
from multiprocessing import Process, Queue
import queue # For the Empty exception

# --- Matplotlib Imports ---
import matplotlib
import matplotlib.style
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Use default (light) style
matplotlib.style.use('default')

matplotlib.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "axes.unicode_minus": False
})

from .session import PreferenceSession, SessionMode, SessionPhase, GroundTruthKind
from ..audio.generator import (
    DEFAULT_OUTPUT_DEVICE,
    OUTPUT_DEVICE_LABELS,
    _safe_pump_events,
    _map_intensity,
    _map_balance_left,
    _map_rhythm_hz,
    _map_grain_duty,
)

try:
    from PIL import Image, ImageTk
except Exception:
    Image = None
    ImageTk = None

from pathlib import Path
from datetime import datetime
import json

# --- Colors & Styles (Light Theme) ---
COLOR_BG = "#f3f4f8"        # Global Background
COLOR_PANEL = "#ffffff"     # Card/Panel Background
COLOR_ACCENT = "#3b82f6"    # Primary Blue
COLOR_ACCENT_HOVER = "#2563eb"
COLOR_FOCUS = "#fbbf24"     # Focus Highlight (Amber/Yellow)
COLOR_TEXT = "#1f2933"      # Main Text
COLOR_MUTED = "#6b7280"     # Secondary Text
COLOR_BORDER = "#e5e7eb"

UNCERTAINTY_DESCRIPTIONS = {
    1: "Very unsure (非常不确定)",
    2: "Somewhat unsure (有点不确定)",
    3: "Neutral (中立)",
    4: "Somewhat sure (有点确定)",
    5: "Very sure (非常确定)",
}

# -------------------------------------------------------------------------
# MULTIPROCESSING WORKER FOR PYGAME
# This runs in a separate process to avoid crashing Tkinter on macOS
# -------------------------------------------------------------------------
def gamepad_worker(state_queue: Queue, command_queue: Queue):
    """
    Worker process that initializes Pygame and polls the controller.
    It sends button events to the main process via state_queue.
    """
    import pygame
    
    # We DO NOT set dummy driver here, so input works correctly.
    try:
        pygame.init()
        pygame.joystick.init()
    except Exception as e:
        state_queue.put(("ERROR", str(e)))
        return

    joystick = None
    
    prev_buttons = []
    prev_hat = (0, 0)
    prev_axis_nav = (0, 0) 

    running = True
    while running:
        # 1. Check for shutdown commands
        try:
            while not command_queue.empty():
                cmd = command_queue.get_nowait()
                if cmd == "STOP":
                    running = False
        except Exception:
            pass
        
        if not running:
            break

        # 2. Ensure Joystick Connected
        if pygame.joystick.get_count() > 0:
            if joystick is None:
                try:
                    joystick = pygame.joystick.Joystick(0)
                    joystick.init()
                    state_queue.put(("STATUS", f"Connected: {joystick.get_name()}"))
                    prev_buttons = [0] * joystick.get_numbuttons()
                except Exception:
                    joystick = None
        else:
            if joystick is not None:
                state_queue.put(("STATUS", "Gamepad Disconnected"))
                joystick = None

        # 3. Poll Input
        if joystick:
            pygame.event.pump()
            
            # --- BUTTONS ---
            num_buttons = joystick.get_numbuttons()
            if len(prev_buttons) != num_buttons:
                prev_buttons = [0] * num_buttons

            for i in range(num_buttons):
                btn_val = joystick.get_button(i)
                if btn_val and not prev_buttons[i]: # Button Down
                    state_queue.put(("BTN_DOWN", i))
                prev_buttons[i] = btn_val

            # --- D-PAD (HAT) ---
            num_hats = joystick.get_numhats()
            if num_hats > 0:
                hat = joystick.get_hat(0)
                if hat != prev_hat:
                    dx = hat[0]
                    dy = hat[1]
                    # SDL Hat mapping often varies, but usually:
                    # (0, 1) is UP, (0, -1) is DOWN in Pygame (sometimes)
                    # Let's align with standard UI direction logic:
                    # Up should decrease row index, Down increase.
                    if dy == 1: state_queue.put(("NAV", (0, -1))) # Up
                    elif dy == -1: state_queue.put(("NAV", (0, 1))) # Down
                    
                    if dx == 1: state_queue.put(("NAV", (1, 0))) # Right
                    elif dx == -1: state_queue.put(("NAV", (-1, 0))) # Left
                    prev_hat = hat

            # --- ANALOG STICK (For Navigation) ---
            num_axes = joystick.get_numaxes()
            if num_axes >= 2:
                ax0 = joystick.get_axis(0)
                ax1 = joystick.get_axis(1)
                
                thresh = 0.6
                curr_x, curr_y = 0, 0
                
                if ax0 > thresh: curr_x = 1
                elif ax0 < -thresh: curr_x = -1
                
                # Axes 1 (Y) is typically inverted: -1 is Up
                if ax1 > thresh: curr_y = 1   # Down
                elif ax1 < -thresh: curr_y = -1 # Up

                if (curr_x, curr_y) != prev_axis_nav:
                    if curr_x != 0 or curr_y != 0:
                         state_queue.put(("NAV", (curr_x, curr_y)))
                    prev_axis_nav = (curr_x, curr_y)

        time.sleep(0.016)

    pygame.quit()


# -------------------------------------------------------------------------
# UI CLASSES
# -------------------------------------------------------------------------

class GamepadFocusManager:
    """
    Manages grid-based navigation for gamepad with smart skipping.
    Allows sparse buttons or wide buttons to be navigated naturally.
    """
    def __init__(self, root: tk.Tk):
        self.root = root
        self.elements: List[List[Optional['GameButton']]] = []
        self.current_row = 0
        self.current_col = 0
        self.active_element: Optional['GameButton'] = None

    def register_grid(self, grid: List[List[Optional['GameButton']]]):
        self.elements = grid
        self.find_valid_focus()

    def find_valid_focus(self):
        """Locate the first available interactive element to focus on start."""
        for r in range(len(self.elements)):
            for c in range(len(self.elements[r])):
                el = self.elements[r][c]
                if el is not None and el.state != "disabled":
                    self.current_row = r
                    self.current_col = c
                    self._update_focus()
                    return

    def navigate(self, dr: int, dc: int):
        """
        Move focus in direction (dr, dc).
        Skips over None cells and *identical* button references (for wide buttons).
        """
        if not self.elements: return
        
        rows = len(self.elements)
        if rows == 0: return
        cols_len = len(self.elements[0]) # Assuming rectangular grid for safety
        
        curr_r, curr_c = self.current_row, self.current_col
        current_btn = self.elements[curr_r][curr_c]
        
        # Start scanning in the direction
        test_r, test_c = curr_r, curr_c
        
        # Limit scan to prevent infinite loops (e.g. grid size * 2)
        max_steps = max(rows, cols_len) * 2
        
        for _ in range(max_steps):
            test_r += dr
            test_c += dc
            
            # 1. Check Bounds
            if not (0 <= test_r < rows): break
            # Note: Rows might have different lengths if not careful, but we assume normalized grid
            if not (0 <= test_c < len(self.elements[test_r])): break
            
            candidate = self.elements[test_r][test_c]
            
            # 2. Skip Logic
            # Skip if None (gap)
            if candidate is None:
                continue
                
            # Skip if it is the SAME button we are already on (e.g. moving right inside a wide button)
            if candidate is current_button:
                continue
                
            # 3. Validation Logic
            # If we found a DIFFERENT button, check if enabled
            if candidate.state != "disabled":
                # Found valid target!
                self.current_row = test_r
                self.current_col = test_c
                self._update_focus()
                return
            else:
                # If disabled, do we stop or skip? 
                # Usually better to stop at disabled buttons so user knows they are there but blocked,
                # rather than skipping over them confusingly.
                # However, for pure navigation fluidity, let's stop but NOT focus (or focus grayed out).
                # Current design: we skip disabled to find next valid.
                continue

    def activate_current(self):
        if self.active_element: self.active_element.invoke()

    def _update_focus(self):
        if self.active_element: self.active_element.set_focus(False)
        el = self.elements[self.current_row][self.current_col]
        if el:
            el.set_focus(True)
            self.active_element = el

    # Helper for logic above
    @property
    def current_button(self):
        if 0 <= self.current_row < len(self.elements):
            row = self.elements[self.current_row]
            if 0 <= self.current_col < len(row):
                return row[self.current_col]
        return None

# Global alias for skip logic check
current_button = None 


class GameButton(tk.Frame):
    """Custom styled button that supports 'focus' state."""
    def __init__(self, parent, text, command=None, on_focus: Optional[Callable] = None, width=200, height=50, bg=COLOR_PANEL, fg=COLOR_TEXT):
        super().__init__(parent, bg=bg, highlightbackground=COLOR_BORDER, highlightthickness=1, relief="flat")
        self.command = command
        self.on_focus_cb = on_focus
        
        self.bg_normal = bg
        self.bg_focus = "#fef3c7" # Light yellow for focus bg
        self.fg_normal = fg
        self.fg_focus = "#000000" # Strong black
        self.border_normal = COLOR_BORDER
        self.border_focus = COLOR_FOCUS
        self.state = "normal"
        
        self.pack_propagate(False)
        self.configure(width=width, height=height)
        self.label = tk.Label(self, text=text, bg=bg, fg=fg, font=("Helvetica", 11, "bold"))
        self.label.place(relx=0.5, rely=0.5, anchor="center")
        
        self.bind("<Button-1>", lambda e: self.invoke())
        self.label.bind("<Button-1>", lambda e: self.invoke())

    def set_focus(self, focused: bool):
        if self.state == "disabled": return
        if focused:
            self.configure(bg=self.bg_focus, highlightbackground=self.border_focus, highlightthickness=3)
            self.label.configure(bg=self.bg_focus, fg=self.fg_focus)
            if self.on_focus_cb:
                self.on_focus_cb()
        else:
            self.configure(bg=self.bg_normal, highlightbackground=self.border_normal, highlightthickness=1)
            self.label.configure(bg=self.bg_normal, fg=self.fg_normal)

    def invoke(self):
        if self.state != "disabled" and self.command:
            # Click flash
            self.label.configure(fg=COLOR_ACCENT)
            self.after(100, lambda: self.label.configure(fg=self.fg_focus if self['highlightthickness'] > 1 else self.fg_normal))
            self.command()

    def set_state(self, state: str):
        self.state = state
        if state == "disabled":
            self.label.configure(fg="#d1d5db") # Very light gray
            self.configure(cursor="arrow")
        else:
            self.label.configure(fg=self.fg_normal)
            self.configure(cursor="hand2")

    def config(self, **kwargs):
        if "state" in kwargs: self.set_state(kwargs["state"])
        if "text" in kwargs: self.label.configure(text=kwargs["text"])
        if "bg" in kwargs:
            self.bg_normal = kwargs["bg"]
            if self.state != "disabled" and self['highlightthickness'] <= 1:
                self.configure(bg=self.bg_normal)
                self.label.configure(bg=self.bg_normal)


class AudioPreferenceStudyApp:
    def __init__(self, root: tk.Tk, session: Optional[PreferenceSession] = None, fixed_mode: Optional[SessionMode] = None):
        self.root = root
        self.root.title("Haptic Preference Learning — Gamepad UI")
        self.root.geometry("1280x800")
        self.root.configure(bg=COLOR_BG)
        
        self.session = session or PreferenceSession()
        try:
            self.session.audio.set_output_device("xbox_controller")
            self.session.audio.duration = 3
        except Exception:
            pass
        self.fixed_mode = fixed_mode
        self.mode_var = tk.StringVar(value=SessionMode.USER.value)
        if self.fixed_mode:
            self.mode_var.set(self.fixed_mode.value)

        self.current_candidate = None
        self.current_audio_data = {}
        self.selected_choice = None
        self.level_var = tk.IntVar(value=3)
        self.auto_play_var = tk.BooleanVar(value=False)
        self._icons = {}
        self._closing = False
        self._test_poll_job: Optional[str] = None
        self._last_drawn_test_iteration: int = -1
        self._last_logged_test_record: int = 0
        self._exported_data: bool = False
        self.validation_rounds: int = 3

        # UI Layout
        self._setup_layout()
        self._init_plots()
        
        # Navigation
        self.focus_manager = GamepadFocusManager(self.root)
        self._build_focus_grid()
        
        # --- START GAMEPAD PROCESS ---
        self.gp_state_q = Queue()
        self.gp_cmd_q = Queue()
        self.gp_process = Process(target=gamepad_worker, args=(self.gp_state_q, self.gp_cmd_q))
        self.gp_process.daemon = True
        self.gp_process.start()
        
        self._log_raw("System", "Gamepad Service Started (Multiprocess).")
        self._poll_gamepad_queue()
        self._poll_pygame_events()

        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _poll_pygame_events(self):
        if self._closing:
            return
        try:
            _safe_pump_events()
        except Exception:
            pass
        self.root.after(50, self._poll_pygame_events)

    def _load_icon(self, name: str, max_height: int = 64) -> Optional[tk.PhotoImage]:
        if name in self._icons: return self._icons[name]
        try:
            assets_dir = Path(__file__).resolve().parent / "logo"
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
        except: pass
        return None

    def _setup_layout(self):
        # Header
        header = tk.Frame(self.root, bg=COLOR_BG, height=80)
        header.pack(fill=tk.X, padx=20, pady=10)
        
        brand_frame = tk.Frame(header, bg=COLOR_BG)
        brand_frame.pack(side=tk.RIGHT)
        usc_icon = self._load_icon("USC", 60)
        harvi_icon = self._load_icon("harvi", 60)
        if usc_icon: tk.Label(brand_frame, image=usc_icon, bg=COLOR_BG).pack(side=tk.LEFT, padx=10)
        else: tk.Label(brand_frame, text="USC", fg="#000000", bg=COLOR_BG, font=("Arial", 20, "bold")).pack(side=tk.LEFT, padx=10)
        if harvi_icon: tk.Label(brand_frame, image=harvi_icon, bg=COLOR_BG).pack(side=tk.LEFT, padx=10)
        else: tk.Label(brand_frame, text="HARVI", fg="#000000", bg=COLOR_BG, font=("Arial", 20, "bold")).pack(side=tk.LEFT, padx=10)

        title_lbl = tk.Label(header, text="Haptic Preference Learning", fg=COLOR_TEXT, bg=COLOR_BG, font=("Helvetica", 24, "bold"))
        title_lbl.pack(side=tk.LEFT, anchor="center")

        # Main Area
        content = tk.Frame(self.root, bg=COLOR_BG)
        content.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)

        # Left Controls
        left_panel = tk.Frame(content, bg=COLOR_BG, width=440)
        left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 20))
        
        # Session Card
        self.card_session = tk.LabelFrame(left_panel, text=" Session ", bg=COLOR_PANEL, fg=COLOR_MUTED, font=("Helvetica", 10))
        self.card_session.pack(fill=tk.X, pady=10)
        self.iter_label = tk.Label(self.card_session, text="Iteration: 0 / 35", bg=COLOR_PANEL, fg=COLOR_ACCENT, font=("Helvetica", 16, "bold"))
        self.iter_label.pack(pady=(10, 5))
        
        btn_frame_1 = tk.Frame(self.card_session, bg=COLOR_PANEL)
        btn_frame_1.pack(pady=10)
        self.btn_start = GameButton(btn_frame_1, "Start Session", command=self._start_session, width=180, bg=COLOR_PANEL)
        self.btn_start.pack(pady=5)
        self.btn_reset = GameButton(btn_frame_1, "Reset", command=self._reset, width=180, bg=COLOR_PANEL)
        self.btn_reset.pack(pady=5)
        self.btn_reset.set_state("disabled")

        # Playback Card (Layout Fixed: Left/Right)
        self.card_play = tk.LabelFrame(left_panel, text=" Haptic Candidates ", bg=COLOR_PANEL, fg=COLOR_MUTED, font=("Helvetica", 10))
        self.card_play.pack(fill=tk.X, pady=10)
        
        play_row = tk.Frame(self.card_play, bg=COLOR_PANEL)
        play_row.pack(fill=tk.X, pady=15, padx=10)
        
        self.btn_play_a = GameButton(
            play_row,
            "▶ Play A (X)",
            command=lambda: self._play(1),
            on_focus=lambda: self._set_wave_display(1),
            width=180,
            bg=COLOR_PANEL,
        )
        self.btn_play_a.pack(side=tk.LEFT, padx=(0, 10), expand=True)
        
        self.btn_play_b = GameButton(
            play_row,
            "▶ Play B (Y)",
            command=lambda: self._play(2),
            on_focus=lambda: self._set_wave_display(2),
            width=180,
            bg=COLOR_PANEL,
        )
        self.btn_play_b.pack(side=tk.LEFT, padx=(10, 0), expand=True)
        
        self.btn_play_a.set_state("disabled")
        self.btn_play_b.set_state("disabled")

        # Choice Card
        self.card_choice = tk.LabelFrame(left_panel, text=" Make Choice ", bg=COLOR_PANEL, fg=COLOR_MUTED, font=("Helvetica", 10))
        self.card_choice.pack(fill=tk.X, pady=10)
        
        row_ab = tk.Frame(self.card_choice, bg=COLOR_PANEL)
        row_ab.pack(pady=10)
        self.btn_choose_a = GameButton(row_ab, "Prefer A", command=lambda: self._choose("A"), width=150, bg=COLOR_PANEL)
        self.btn_choose_a.pack(side=tk.LEFT, padx=10)
        self.btn_choose_b = GameButton(row_ab, "Prefer B", command=lambda: self._choose("B"), width=150, bg=COLOR_PANEL)
        self.btn_choose_b.pack(side=tk.LEFT, padx=10)
        self.btn_choose_a.set_state("disabled")
        self.btn_choose_b.set_state("disabled")

        lbl_unc = tk.Label(self.card_choice, text="Uncertainty Level", bg=COLOR_PANEL, fg=COLOR_MUTED)
        lbl_unc.pack(pady=(10, 0))
        self.unc_desc_label = tk.Label(self.card_choice, text="Neutral", bg=COLOR_PANEL, fg=COLOR_ACCENT, font=("Helvetica", 11, "bold"))
        self.unc_desc_label.pack(pady=(0, 10))

        unc_row = tk.Frame(self.card_choice, bg=COLOR_PANEL)
        unc_row.pack(pady=(0, 15))
        self.unc_btns = []
        for i in range(1, 6):
            # Pass on_focus callback to update label dynamically
            b = GameButton(unc_row, str(i), command=lambda x=i: self._set_level_selection(x), 
                           on_focus=lambda x=i: self._on_unc_focus(x),
                           width=45, height=45, bg=COLOR_PANEL)
            b.pack(side=tk.LEFT, padx=4)
            self.unc_btns.append(b)

        self.btn_submit = GameButton(left_panel, "SUBMIT CHOICE (RB)", command=self._submit, width=400, height=60, bg=COLOR_ACCENT, fg="#ffffff")
        self.btn_submit.pack(pady=20)
        self.btn_submit.set_state("disabled")

        # Right Panel
        right_panel = tk.Frame(content, bg=COLOR_BG)
        right_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        self.notebook = ttk.Notebook(right_panel)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        self.tab_wave = tk.Frame(self.notebook, bg=COLOR_PANEL)
        self.tab_map = tk.Frame(self.notebook, bg=COLOR_PANEL)
        self.tab_log = tk.Frame(self.notebook, bg=COLOR_PANEL)
        
        self.notebook.add(self.tab_wave, text="Waveforms")
        if self.mode_var.get() == SessionMode.TEST.value: self.notebook.add(self.tab_map, text="GT Map")
        self.notebook.add(self.tab_log, text="Log")
        
        self.log_text = tk.Text(self.tab_log, bg="#f9fafb", fg="#111827", font=("Consolas", 10), relief="flat")
        self.log_text.pack(fill=tk.BOTH, expand=True, padx=1, pady=1)

        self.status_var = tk.StringVar(value="Gamepad: Initializing...")
        footer = tk.Label(self.root, textvariable=self.status_var, bg="#e5e7eb", fg=COLOR_TEXT, font=("Helvetica", 10), anchor="w", padx=10)
        footer.pack(fill=tk.X, side=tk.BOTTOM)

    def _build_focus_grid(self):
        # OPTIMIZED GRID LOGIC:
        # Create a 5-column grid. Buttons that are "centered" or "wide" occupy multiple slots.
        # This ensures vertical navigation always hits the target (e.g., Down from Col 4 hits Submit).
        
        # Row 0: Start (Occupies all 5 cols for easy access)
        r0 = [self.btn_start] * 5
        
        # Row 1: Reset (Occupies all 5 cols)
        r1 = [self.btn_reset] * 5
        
        # Row 2: Play A (Cols 0-2), Play B (Cols 3-4)
        # This split (3 vs 2) generally feels balanced for "Left side vs Right side"
        r2 = [self.btn_play_a, self.btn_play_a, self.btn_play_a, self.btn_play_b, self.btn_play_b]
        
        # Row 3: Choose A (Cols 0-2), Choose B (Cols 3-4)
        r3 = [self.btn_choose_a, self.btn_choose_a, self.btn_choose_a, self.btn_choose_b, self.btn_choose_b]
        
        # Row 4: Uncertainty (Exact 5 items, 1 per col)
        r4 = self.unc_btns
        
        # Row 5: Submit (Occupies all 5 cols)
        r5 = [self.btn_submit] * 5
        
        grid = [r0, r1, r2, r3, r4, r5]
        self.focus_manager.register_grid(grid)

    def _init_plots(self):
        self.fig_wave = Figure(figsize=(5, 4), dpi=100)
        self.fig_wave.patch.set_facecolor(COLOR_PANEL)
        self.canvas_wave = FigureCanvasTkAgg(self.fig_wave, master=self.tab_wave)
        self.canvas_wave.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self._draw_empty_wave()
        
        self.fig_map = Figure(figsize=(5, 4), dpi=100)
        self.fig_map.patch.set_facecolor(COLOR_PANEL)
        self.canvas_map = FigureCanvasTkAgg(self.fig_map, master=self.tab_map)
        self.canvas_map.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def _draw_empty_wave(self):
        self.fig_wave.clear()
        ax = self.fig_wave.add_subplot(111)
        ax.set_facecolor(COLOR_PANEL)
        ax.text(1.5, 0.5, "Press Start to Begin", ha="center", va="center", color=COLOR_MUTED)
        ax.set_xlim(0, 3.0)
        ax.set_ylim(-0.05, 1.1)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Rumble Intensity (0–1)")
        ax.grid(True, alpha=0.3)
        self.canvas_wave.draw()

    def _update_iteration_label(self):
        if self.session.state.phase is SessionPhase.VALIDATION:
            idx = self.session.state.validation_index + 1
            total = max(1, self.session.state.validation_rounds)
            self.iter_label.config(text=f"Validation Round {idx} / {total}")
        else:
            self.iter_label.config(
                text=f"Iteration: {self.session.state.current_iteration} / {self.session.state.max_iterations}"
            )

    # --- Session Logic ---
    def _start_session(self):
        try:
            mode = SessionMode.USER if self.mode_var.get() == SessionMode.USER.value else SessionMode.TEST
            self.session.start(
                mode,
                35,
                GroundTruthKind.GAUSSIAN_CENTER.value,
                validation_rounds=self.validation_rounds,
            )
            self._exported_data = False
            self._last_drawn_test_iteration = -1
            self._last_logged_test_record = 0
            self._cancel_test_poll()
            self._log_raw("System", "Session Started.")
            self._log_raw("System", "Rumble params: I/T/R/G in [20,100], duration=3.0s to Xbox controller.")
            self.btn_start.set_state("disabled")
            self.btn_reset.set_state("normal")
            if mode is SessionMode.TEST:
                self.btn_play_a.set_state("disabled")
                self.btn_play_b.set_state("disabled")
                self.btn_choose_a.set_state("disabled")
                self.btn_choose_b.set_state("disabled")
                self.btn_submit.set_state("disabled")
                self._draw_empty_wave()
                self._log_raw("System", "Automatic test started (no manual A/B selection).")
                self.session.run_test_loop()
                self._schedule_test_poll()
                self._draw_map()
            else:
                self._prepare_new_candidate()
                self.focus_manager.current_row = 2
                self.focus_manager.current_col = 0
                self.focus_manager._update_focus()
        except Exception as e:
            self._log_raw("Error", f"Starting: {e}")

    def _reset(self):
        self._cancel_test_poll()
        self.session.reset()
        self.current_candidate = None
        self.selected_choice = None
        self._draw_empty_wave()
        self.btn_start.set_state("normal")
        self.btn_reset.set_state("disabled")
        self.btn_play_a.set_state("disabled")
        self.btn_play_b.set_state("disabled")
        self.btn_choose_a.set_state("disabled")
        self.btn_choose_b.set_state("disabled")
        self.btn_submit.set_state("disabled")
        self.iter_label.config(text="Iteration: 0 / 35")
        self._log_raw("System", "Session Reset.")
        self._exported_data = False
        self._last_drawn_test_iteration = -1
        self._last_logged_test_record = 0

    def _prepare_new_candidate(self):
        try:
            if self.session.state.phase is SessionPhase.VALIDATION:
                self.current_candidate = self.session.generate_validation_query()
            else:
                self.current_candidate = self.session.generate_user_query()
            if not self.current_candidate:
                self._finish_session()
                return
            self.current_audio_data = self.current_candidate.audio_data
            self._wave_display = None
            self._draw_waveforms()
            self.selected_choice = None
            self.btn_choose_a.config(text="Prefer A", bg=COLOR_PANEL, state="normal")
            self.btn_choose_b.config(text="Prefer B", bg=COLOR_PANEL, state="normal")
            self.btn_submit.set_state("disabled")
            self.btn_play_a.set_state("normal")
            self.btn_play_b.set_state("normal")
            self._update_iteration_label()
        except Exception as e:
            self._log_raw("Error", f"Generating query: {e}")

    def _play(self, which):
        if not self.current_candidate: return
        try:
            self.session.audio.stop_audio()
            entry = self.current_audio_data.get(which, {})
            if not entry:
                return
            self.session.audio.play_audio(entry.get("x"), metadata=entry.get("meta"), blocking=False)
        except Exception as e:
            self._log_raw("Error", f"Audio: {e}")

    def _choose(self, choice):
        if not self.current_candidate: return
        self.selected_choice = choice
        if choice == "A":
            self.btn_choose_a.config(bg=COLOR_ACCENT, text="Preferred A ✓")
            self.btn_choose_b.config(bg=COLOR_PANEL, text="Prefer B")
        else:
            self.btn_choose_b.config(bg=COLOR_ACCENT, text="Preferred B ✓")
            self.btn_choose_a.config(bg=COLOR_PANEL, text="Prefer A")
        
        # Make sure Submit button is enabled
        self.btn_submit.set_state("normal")
        self.btn_submit.config(bg=COLOR_ACCENT) 

    def _on_unc_focus(self, level):
        """Called when gamepad focus moves to an uncertainty button."""
        self.unc_desc_label.config(text=UNCERTAINTY_DESCRIPTIONS.get(level, ""))

    def _set_level_selection(self, level):
        """Called when a level is actually selected (clicked/pressed)."""
        self.level_var.set(level)
        self.unc_desc_label.config(text=UNCERTAINTY_DESCRIPTIONS.get(level, ""))
        for i, btn in enumerate(self.unc_btns):
            if (i + 1) == level: btn.config(bg=COLOR_ACCENT, fg="#ffffff")
            else: btn.config(bg=COLOR_PANEL, fg=COLOR_TEXT)

    def _submit(self):
        if not self.selected_choice: return
        try:
            level = self.level_var.get()
            if self.session.state.phase is SessionPhase.VALIDATION:
                round_idx = self.session.state.validation_index + 1
                self.session.record_validation_choice(self.selected_choice, level)
                self._log_pair("Validation", level, self.selected_choice, iter_label=f"Val {round_idx:02d}")

                if self.session.validation_complete():
                    self._finish_session()
                else:
                    self._prepare_new_candidate()
                    self.focus_manager.current_row = 2
                    self.focus_manager.current_col = 0
                    self.focus_manager._update_focus()
                return

            self.session.record_user_choice(self.selected_choice, level)

            # DETAILED LOGGING (Restored)
            self._log_pair("User", level, self.selected_choice)

            if self.session.training_complete():
                if self.session.state.validation_rounds > 0:
                    self.session.start_validation()
                    self._log_raw(
                        "System",
                        f"Validation rounds started ({self.session.state.validation_rounds}).",
                    )
                    self._prepare_new_candidate()
                else:
                    self._finish_session()
            else:
                self._prepare_new_candidate()
                # Reset focus to Play A
                self.focus_manager.current_row = 2
                self.focus_manager.current_col = 0
                self.focus_manager._update_focus()
        except Exception as e:
            self._log_raw("Error", f"Submit error: {e}")

    def _finish_session(self):
        self._log_raw("System", "Session Complete!")
        self._show_recommendation_dialog(
            title="User Study Complete",
            header="User Study Complete. Thank you!",
            on_close=self._finalize_user_session,
        )

    def _finalize_user_session(self):
        self._persist_study_data(status="complete")
        self._reset()

    # --- Logging Helpers (Restored) ---
    def _log_raw(self, tag, msg):
        ts = datetime.now().strftime("%H:%M:%S")
        self.log_text.insert(tk.END, f"[{ts}] [{tag}] {msg}\n")
        self.log_text.see(tk.END)

    def _fmt_phys(self, phys: Sequence[float]) -> str:
        return f"[I={phys[0]:.2f}, T={phys[1]:.2f}, R={phys[2]:.2f}, G={phys[3]:.2f}]"

    def _fmt_rumble(self, meta: Optional[dict]) -> str:
        if not meta:
            return "[I=?, T=?, R=?, G=?]"
        sliders = meta.get("sliders", {})
        try:
            intensity_slider = float(sliders.get("intensity", 50.0))
            texture_slider = float(sliders.get("texture", 50.0))
            rhythm_slider = float(sliders.get("rhythm", 50.0))
            grain_slider = float(sliders.get("grain", 50.0))

            inten = _map_intensity(intensity_slider)
            tex = _map_balance_left(texture_slider)
            rhythm = _map_rhythm_hz(rhythm_slider)
            grain = _map_grain_duty(grain_slider)
            return (
                f"[I={inten:.2f}, "
                f"T=L{tex*100:.0f}%/R{(1-tex)*100:.0f}%, "
                f"R={rhythm:.2f}Hz, "
                f"G={grain*100:.0f}% | "
                f"s={intensity_slider:.1f},{texture_slider:.1f},{rhythm_slider:.1f},{grain_slider:.1f}]"
            )
        except Exception:
            return "[I=?, T=?, R=?, G=?]"

    def _log_pair(self, mode_tag: str, level: int, choice: str, iter_label: Optional[str] = None) -> None:
        if not self.current_candidate: return
        p1_phys, p2_phys = self.current_candidate.physical
        meta1 = self.current_audio_data.get(1, {}).get("meta", {})
        meta2 = self.current_audio_data.get(2, {}).get("meta", {})
        extra = []
        if not np.isnan(self.current_candidate.info_gain):
            extra.append(f"EIG={self.current_candidate.info_gain:.3f}")
        extra.append(f"Dist={self.current_candidate.query_distance:.3f}")
        extra_str = (" | " + ", ".join(extra)) if extra else ""

        iter_tag = iter_label if iter_label is not None else f"Iter {self.session.state.current_iteration:02d}"
        line = (
            f"{iter_tag} | "
            f"A {self._fmt_rumble(meta1)} vs B {self._fmt_rumble(meta2)} | "
            f"pick={choice} | level={level}{extra_str}"
        )
        self._log_raw(mode_tag, line)

    def _persist_study_data(self, status: str = "complete"):
        if self._exported_data:
            return
        try:
            snapshot = self.session.build_snapshot(status=status)
            snapshot["timestamp"] = datetime.now().isoformat()

            base_dir = Path.cwd() / "data"
            base_dir.mkdir(exist_ok=True)
            date_prefix = datetime.now().strftime("%Y%m%d")
            counter = 1
            while True:
                exp_dir = base_dir / f"{date_prefix}_{counter:02d}"
                if not exp_dir.exists():
                    exp_dir.mkdir()
                    break
                counter += 1

            (exp_dir / "session.json").write_text(json.dumps(snapshot, indent=2), encoding="utf-8")
            log_text = self.log_text.get("1.0", tk.END).strip() if self.log_text is not None else ""
            (exp_dir / "log.txt").write_text(log_text, encoding="utf-8")
            self._exported_data = True
        except Exception as exc:
            self._log_raw("Error", f"Export failed: {exc}")

    def _play_recommended(self):
        if self.session.rec_best.parameters is None:
            return
        try:
            params = np.asarray(self.session.rec_best.parameters, dtype=float)
            self.session.audio.stop_audio()
            _, data, meta = self.session.audio.generate_signal(*params)
            self.session.audio.play_audio(data, metadata=meta, blocking=False)
        except Exception as exc:
            self._log_raw("Error", f"Play recommended failed: {exc}")

    def _show_recommendation_dialog(
        self,
        title: str,
        header: str,
        detail: Optional[str] = None,
        on_close: Optional[Callable[[], None]] = None,
    ) -> None:
        params = None
        if self.session.rec_best.parameters is not None:
            params = np.asarray(self.session.rec_best.parameters, dtype=float)

        dialog = tk.Toplevel(self.root)
        dialog.title(title)
        dialog.configure(bg=COLOR_PANEL)
        dialog.resizable(False, False)
        dialog.transient(self.root)
        dialog.grab_set()

        frame = tk.Frame(dialog, bg=COLOR_PANEL, padx=24, pady=20)
        frame.pack(fill=tk.BOTH, expand=True)

        tk.Label(
            frame,
            text=header,
            bg=COLOR_PANEL,
            fg=COLOR_TEXT,
            font=("Helvetica", 14, "bold"),
        ).pack(pady=(0, 8))

        if detail:
            tk.Label(
                frame,
                text=detail,
                bg=COLOR_PANEL,
                fg=COLOR_MUTED,
                font=("Helvetica", 11),
            ).pack(pady=(0, 10))

        if params is not None:
            tk.Label(
                frame,
                text=f"Recommended: {self._fmt_phys(params)}",
                bg=COLOR_PANEL,
                fg=COLOR_ACCENT,
                font=("Helvetica", 12, "bold"),
            ).pack(pady=(0, 18))
        else:
            tk.Label(
                frame,
                text="No recommendation available.",
                bg=COLOR_PANEL,
                fg=COLOR_MUTED,
                font=("Helvetica", 12),
            ).pack(pady=(0, 18))

        btn_row = tk.Frame(frame, bg=COLOR_PANEL)
        btn_row.pack()

        play_btn = tk.Button(
            btn_row,
            text="Play Recommended",
            command=self._play_recommended,
            bg=COLOR_ACCENT,
            fg="#ffffff",
            relief="flat",
            padx=14,
            pady=6,
        )
        if params is None:
            play_btn.configure(state="disabled")
        play_btn.pack(side=tk.LEFT, padx=6)

        def handle_close():
            try:
                dialog.grab_release()
            except Exception:
                pass
            dialog.destroy()
            if on_close:
                on_close()

        close_btn = tk.Button(
            btn_row,
            text="Close",
            command=handle_close,
            bg=COLOR_PANEL,
            fg=COLOR_TEXT,
            relief="groove",
            padx=14,
            pady=6,
        )
        close_btn.pack(side=tk.LEFT, padx=6)
        dialog.protocol("WM_DELETE_WINDOW", handle_close)

    # --- Auto-test polling (TEST mode) ---
    def _schedule_test_poll(self) -> None:
        self._test_poll_job = self.root.after(250, self._poll_test_session)

    def _cancel_test_poll(self) -> None:
        if self._test_poll_job is None:
            return
        try:
            self.root.after_cancel(self._test_poll_job)
        except Exception:
            pass
        self._test_poll_job = None

    def _poll_test_session(self) -> None:
        if self.session.state.mode is not SessionMode.TEST:
            self._cancel_test_poll()
            return

        iteration = int(self.session.state.current_iteration)
        self.iter_label.config(text=f"Iteration: {iteration} / {self.session.state.max_iterations}")

        try:
            records = getattr(self.session, "test_query_history", [])
            while self._last_logged_test_record < len(records):
                rec = records[self._last_logged_test_record]
                p1_phys, p2_phys = rec.physical
                meta1 = {
                    "sliders": {
                        "intensity": float(p1_phys[0]),
                        "texture": float(p1_phys[1]),
                        "rhythm": float(p1_phys[2]),
                        "grain": float(p1_phys[3]),
                    }
                }
                meta2 = {
                    "sliders": {
                        "intensity": float(p2_phys[0]),
                        "texture": float(p2_phys[1]),
                        "rhythm": float(p2_phys[2]),
                        "grain": float(p2_phys[3]),
                    }
                }
                extra_str = f" | EIG={float(rec.info_gain):.3f}, Dist={float(rec.query_distance):.3f}"
                line = (
                    f"Iter {int(rec.iteration):02d} | "
                    f"A {self._fmt_rumble(meta1)} vs B {self._fmt_rumble(meta2)} | "
                    f"pick={rec.choice} | level={int(rec.level)}{extra_str}"
                )
                self._log_raw("Test", line)
                self._last_logged_test_record += 1
        except Exception:
            pass

        if iteration != self._last_drawn_test_iteration:
            self._draw_map()
            self._last_drawn_test_iteration = iteration

        if not self.session.state.running:
            self.btn_start.set_state("normal")
            self.btn_reset.set_state("normal")
            self._cancel_test_poll()

            if self.session.rec_best.parameters is not None:
                params = np.asarray(self.session.rec_best.parameters, dtype=float)
                dist = float(np.linalg.norm(params - np.asarray(self.session.ideal_phys, dtype=float)))
                self._log_raw(
                    "Final-Test",
                    f"Recommendation: [{params[0]:.2f}, {params[1]:.2f}, {params[2]:.2f}, {params[3]:.2f}] | "
                    f"DistToGT={dist:.3f}",
                )
                self._show_recommendation_dialog(
                    title="Automatic Test Complete",
                    header="Automatic test complete.",
                    detail=f"Distance to GT: {dist:.3f}",
                    on_close=lambda: self._persist_study_data(status="complete"),
                )
            else:
                messagebox.showinfo("Test complete", f"Ran {iteration} iterations.")
                self._persist_study_data(status="complete")

            return

        self._test_poll_job = self.root.after(250, self._poll_test_session)

    # --- GT/GP map (TEST mode) ---
    def _draw_map(self) -> None:
        if self.session.state.mode is not SessionMode.TEST:
            return
        if self.fig_map is None or self.canvas_map is None or self.session.gp is None:
            return

        gp = self.session.gp
        param_ranges = self.session.audio.param_ranges

        def midpoint(key: str) -> float:
            low, high = param_ranges[key]
            return float((float(low) + float(high)) / 2.0)

        def gp_mean(phys: Sequence[float]) -> float:
            try:
                mu = gp.mean1pt(gp.normalize_parameters(phys))
                return float(mu[0] if isinstance(mu, (list, tuple, np.ndarray)) else mu)
            except Exception:
                return 0.0

        self.fig_map.clear()
        ax11 = self.fig_map.add_subplot(221, projection="3d")
        ax12 = self.fig_map.add_subplot(222, projection="3d")
        ax21 = self.fig_map.add_subplot(223, projection="3d")
        ax22 = self.fig_map.add_subplot(224, projection="3d")

        fix_r = midpoint("density")
        fix_g = midpoint("gradient")
        xs = np.linspace(*param_ranges["amplitude"], 31)
        ys = np.linspace(*param_ranges["frequency"], 31)
        X, Y = np.meshgrid(xs, ys)
        Z_gt = np.zeros_like(X)
        Z_gp = np.zeros_like(X)
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                phys = [X[i, j], Y[i, j], fix_r, fix_g]
                Z_gt[i, j] = self.session.gt_value(phys)
                Z_gp[i, j] = gp_mean(phys)

        ax11.plot_surface(X, Y, Z_gt, cmap="coolwarm", alpha=0.9)
        ax11.set_title(f"GT: Intensity×Texture @ (R={fix_r:.0f}, G={fix_g:.0f})")
        ax11.set_xlabel("Intensity")
        ax11.set_ylabel("Texture")
        ax11.set_zlabel("GT")
        ax12.plot_surface(X, Y, Z_gp, cmap="viridis", alpha=0.9)
        ax12.set_title(f"GP: Intensity×Texture @ (R={fix_r:.0f}, G={fix_g:.0f})")
        ax12.set_xlabel("Intensity")
        ax12.set_ylabel("Texture")
        ax12.set_zlabel("GP mean")

        ideal = np.asarray(self.session.ideal_phys, dtype=float)
        ax11.scatter(ideal[0], ideal[1], self.session.gt_value([ideal[0], ideal[1], fix_r, fix_g]), marker="*", s=120)
        ax12.scatter(ideal[0], ideal[1], gp_mean([ideal[0], ideal[1], fix_r, fix_g]), marker="*", s=120)
        if self.session.rec_best.parameters is not None:
            rec = np.asarray(self.session.rec_best.parameters, dtype=float)
            ax11.scatter(rec[0], rec[1], self.session.gt_value([rec[0], rec[1], fix_r, fix_g]), marker="x", s=80)
            ax12.scatter(rec[0], rec[1], gp_mean([rec[0], rec[1], fix_r, fix_g]), marker="x", s=80)

        fix_i = midpoint("amplitude")
        fix_t = midpoint("frequency")
        xs = np.linspace(*param_ranges["density"], 31)
        ys = np.linspace(*param_ranges["gradient"], 31)
        X, Y = np.meshgrid(xs, ys)
        Z_gt = np.zeros_like(X)
        Z_gp = np.zeros_like(X)
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                phys = [fix_i, fix_t, X[i, j], Y[i, j]]
                Z_gt[i, j] = self.session.gt_value(phys)
                Z_gp[i, j] = gp_mean(phys)

        ax21.plot_surface(X, Y, Z_gt, cmap="coolwarm", alpha=0.9)
        ax21.set_title(f"GT: Rhythm×Grain @ (I={fix_i:.0f}, T={fix_t:.0f})")
        ax21.set_xlabel("Rhythm")
        ax21.set_ylabel("Grain")
        ax21.set_zlabel("GT")
        ax22.plot_surface(X, Y, Z_gp, cmap="viridis", alpha=0.9)
        ax22.set_title(f"GP: Rhythm×Grain @ (I={fix_i:.0f}, T={fix_t:.0f})")
        ax22.set_xlabel("Rhythm")
        ax22.set_ylabel("Grain")
        ax22.set_zlabel("GP mean")

        ax21.scatter(ideal[2], ideal[3], self.session.gt_value([fix_i, fix_t, ideal[2], ideal[3]]), marker="*", s=120)
        ax22.scatter(ideal[2], ideal[3], gp_mean([fix_i, fix_t, ideal[2], ideal[3]]), marker="*", s=120)
        if self.session.rec_best.parameters is not None:
            rec = np.asarray(self.session.rec_best.parameters, dtype=float)
            ax21.scatter(rec[2], rec[3], self.session.gt_value([fix_i, fix_t, rec[2], rec[3]]), marker="x", s=80)
            ax22.scatter(rec[2], rec[3], gp_mean([fix_i, fix_t, rec[2], rec[3]]), marker="x", s=80)

        self.fig_map.tight_layout()
        self.canvas_map.draw_idle()

    def _segments_to_plot(self, segments, total_duration: float):
        times, lefts, rights = [0.0], [0.0], [0.0]
        if not segments:
            end_t = max(3.0, float(total_duration))
            return [0.0, end_t], [0.0, 0.0], [0.0, 0.0], end_t

        times.append(segments[0]["start"])
        lefts.append(0.0)
        rights.append(0.0)

        for seg in segments:
            start = float(seg.get("start", 0.0))
            end = start + float(seg.get("duration", 0.0))
            l_val = float(seg.get("left", 0.0))
            r_val = float(seg.get("right", 0.0))

            if start > times[-1] + 0.0001:
                times.append(start); lefts.append(0.0); rights.append(0.0)

            times.append(start); lefts.append(l_val); rights.append(r_val)
            times.append(end);   lefts.append(l_val); rights.append(r_val)

            if not seg.get("continuous_next", False):
                times.append(end); lefts.append(0.0); rights.append(0.0)

        max_t = max(total_duration, times[-1] if times else 3.0)
        return times, lefts, rights, max_t

    def _draw_waveforms(self):
        self.fig_wave.clear()
        ax = self.fig_wave.add_subplot(111)
        ax.set_facecolor(COLOR_PANEL)
        max_t = 3.0
        plotted = False
        show_target = getattr(self, "_wave_display", None)
        has_a = 1 in self.current_audio_data
        has_b = 2 in self.current_audio_data
        pairs = []
        if show_target == 1 and has_a:
            pairs = [(1, "A", "-")]
        elif show_target == 2 and has_b:
            pairs = [(2, "B", "--")]
        elif has_a and has_b:
            pairs = [(1, "A", "-"), (2, "B", "--")]
        elif has_a:
            pairs = [(1, "A", "-")]
        elif has_b:
            pairs = [(2, "B", "--")]

        for idx, label_prefix, style in pairs:
            entry = self.current_audio_data.get(idx, {})
            meta = entry.get("meta", {})
            segments = meta.get("segments", [])
            total_dur = float(meta.get("duration", 3.0))
            times, lefts, rights, local_max = self._segments_to_plot(segments, total_dur)
            max_t = max(max_t, local_max)
            ax.plot(times, lefts, linestyle=style, label=f"{label_prefix} Left", linewidth=2)
            ax.plot(times, rights, linestyle=style, label=f"{label_prefix} Right", linewidth=2, alpha=0.7)
            plotted = True

        ax.set_xlim(0, max(3.0, max_t))
        ax.set_ylim(-0.05, 1.1)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Rumble Intensity (0–1)")
        if plotted:
            ax.legend(facecolor=COLOR_PANEL)
        ax.grid(True, alpha=0.3)
        ax.tick_params(colors=COLOR_TEXT)
        for spine in ax.spines.values(): spine.set_edgecolor(COLOR_BORDER)
        self.canvas_wave.draw_idle()

    def _set_wave_display(self, which: Optional[int]):
        self._wave_display = which
        self._draw_waveforms()

    # --- GAMEPAD POLLING ---
    def _poll_gamepad_queue(self):
        try:
            while not self.gp_state_q.empty():
                evt_type, data = self.gp_state_q.get_nowait()
                
                if evt_type == "STATUS":
                    self.status_var.set(f"Gamepad: {data}")
                    
                elif evt_type == "ERROR":
                    self._log_raw("GP-Err", data)
                    
                elif evt_type == "BTN_DOWN":
                    idx = data
                    # Standard Xbox: A=0, B=1, X=2, Y=3, LB=4, RB=5, Back=6, Start=7
                    if idx == 0: self.focus_manager.activate_current() # A
                    elif idx == 2: # X
                        if self.btn_play_a.state == "normal": self._play(1)
                    elif idx == 3: # Y
                        if self.btn_play_b.state == "normal": self._play(2)
                    elif idx == 5: # RB
                        if self.btn_submit.state == "normal": self._submit()
                    elif idx == 7: # Start
                        if self.btn_start.state == "normal": self._start_session()
                
                elif evt_type == "NAV":
                    dx, dy = data
                    self.focus_manager.navigate(dy, dx)
        except queue.Empty:
            pass
        finally:
            self.root.after(20, self._poll_gamepad_queue)

    def _on_close(self):
        self._closing = True
        try:
            self._cancel_test_poll()
        except Exception:
            pass
        try:
            self.session.stop()
        except Exception:
            pass
        if not self._exported_data:
            status = "complete" if self.session.is_complete() else "incomplete"
            self._persist_study_data(status=status)
        try: self.session.audio.stop_audio()
        except: pass
        if self.gp_process.is_alive():
            self.gp_cmd_q.put("STOP")
            self.gp_process.join(timeout=0.5)
            if self.gp_process.is_alive():
                self.gp_process.terminate()
        self.root.destroy()

def user_main() -> int:
    try:
        root = tk.Tk()
        app = AudioPreferenceStudyApp(root, fixed_mode=SessionMode.USER)
        root.mainloop()
    except Exception as exc:
        print(f"Failed to launch: {exc}")
        return 1
    return 0

def test_main() -> int:
    try:
        root = tk.Tk()
        app = AudioPreferenceStudyApp(root, fixed_mode=SessionMode.TEST)
        root.mainloop()
    except Exception as exc:
        print(f"Failed to launch auto test UI: {exc}")
        return 1
    return 0

def main() -> int:
    return user_main()

if __name__ == "__main__":
    main()
