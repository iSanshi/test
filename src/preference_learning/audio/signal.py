"""
Audio signal synthesis utilities.
"""

from __future__ import annotations

import numpy as np


def generate_tone_signal(
    filler_amplitude: float,
    filler_frequency: float,
    filler_density: float,
    filler_env_gradient: float,
    duration: float,
    cycles: int,
    fs: float,
):
    """
    Generate a tone signal with the specified parameters.

    This function is a direct refactor of the original ``generateSignal6params`` module.
    Behaviour and numerical operations remain unchanged.
    """
    pattern = {
        "filler": {
            "amplitude": float(filler_amplitude),
            "frequency": float(filler_frequency),
            "density": float(filler_density),
            "envelope": {"relative_gradient": float(filler_env_gradient)},
        },
        "duration": float(duration),
        "cycle": float(cycles),
    }
    fs = float(fs)

    absolute_frequency = (250 - 50) / 100 * pattern["filler"]["frequency"] + 50

    if 110 < absolute_frequency <= 130:
        scale = (0.2 - 1) / (130 - 110) * absolute_frequency - (0.2 - 1) / (130 - 110) * 130 + 0.2
    elif 130 < absolute_frequency < 150:
        scale = (1 - 0.2) / (150 - 130) * absolute_frequency - (1 - 0.2) / (150 - 130) * 130 + 0.2
    else:
        scale = 1

    absolute_amplitude = pattern["filler"]["amplitude"] / 100 * scale

    filler_time = np.arange(0, pattern["duration"], 1 / fs)
    len_signal = len(filler_time)

    pic = 100e-3
    granularity = pic / 2 * 5
    nu = round(pattern["duration"] / granularity)
    granularity = round(pattern["duration"] / nu, 3)
    pic = granularity / 5 * 2

    if pattern["filler"]["density"] <= 50:
        boundary = -pattern["filler"]["density"] / 50 + 1
        fade_number = round(pic * fs)
        upper = np.linspace(boundary, 1, fade_number)
        downer = np.linspace(1, boundary, fade_number)
    else:
        fade_number = round(
            ((0.1 - 1) / 50 * pattern["filler"]["density"] - (0.1 - 1) / 50 * 50 + 1) * pic * fs
        )
        upper = np.linspace(0, 1, fade_number)
        before_upper = np.zeros(round(pic * fs - len(upper)))
        upper = np.concatenate([before_upper, upper])
        downer = np.linspace(1, 0, fade_number)
        after_downer = np.zeros(round(pic * fs - len(downer)))
        downer = np.concatenate([downer, after_downer])

    keeper = np.ones(round(granularity * fs) - len(upper) - len(downer))
    env = np.concatenate([upper, keeper, downer])
    num = round(granularity * fs)

    envelope = np.array([])
    j = 1
    idx = j * num
    while idx <= len_signal:
        envelope = np.concatenate([envelope, env])
        j += 1
        idx = j * num

    if len(envelope) < len_signal:
        envelope = np.concatenate([envelope, env[: len_signal - len(envelope)]])
    elif len(envelope) > len_signal:
        raise ValueError("Envelope construction exceeded signal length.")

    if pattern["filler"]["envelope"]["relative_gradient"] <= 0:
        p1 = (100 + pattern["filler"]["envelope"]["relative_gradient"]) / 100
        p2 = 1
    else:
        p1 = 1
        p2 = (100 - pattern["filler"]["envelope"]["relative_gradient"]) / 100

    filler_env = np.linspace(p1, p2, len_signal)

    fm1 = np.linspace(0, 1, int(10e-3 * fs))
    fm2 = np.linspace(1, 0, int(10e-3 * fs))
    fade_mask = np.ones(len_signal)
    fade_mask[: len(fm1)] = fm1
    fade_mask[-len(fm2) :] = fm2

    base_filler = absolute_amplitude * np.sin(2 * np.pi * absolute_frequency * filler_time)
    conv_density = base_filler * envelope
    conv_envelope = conv_density * filler_env
    conv_fade_mask = conv_envelope * fade_mask

    data = np.tile(conv_fade_mask, int(pattern["cycle"]))
    time = np.arange(len(data)) / fs

    data = np.round(data, 5)
    time = np.round(time, 6)

    for_plot = {
        "for_density": envelope * absolute_amplitude,
        "for_envelope": filler_env * absolute_amplitude,
        "filler_time": filler_time,
    }

    return time, data, for_plot


def generate_xbox_signal(
    amplitude: float,
    frequency: float,
    density: float,
    gradient: float,
    duration: float,
    sample_rate: int = 100,
):
    """
    Dual-channel haptic control signal for Xbox controllers.

    Returns a stereo array where column 0 targets the heavy/low-frequency motor
    and column 1 targets the light/high-frequency motor.
    """
    amp_norm = float(np.clip(amplitude / 100.0, 0.0, 1.0))
    freq_norm = float(np.clip(frequency / 100.0, 0.0, 1.0))
    dens_norm = float(np.clip(density / 100.0, 0.0, 1.0))

    n_samples = max(int(duration * sample_rate), 1)
    t = np.linspace(0.0, float(duration), n_samples, endpoint=False)

    # Balance left/right using constant-power panning.
    left_gain = np.cos(freq_norm * np.pi / 2.0)
    right_gain = np.sin(freq_norm * np.pi / 2.0)

    # Texture modulation from density with explicit duty cycle to create rhythm.
    if dens_norm < 0.08:
        texture_mod = np.ones_like(t)
    else:
        mod_freq = 3.0 + dens_norm * 9.0  # 3–12 Hz pulses
        duty = 0.75 - dens_norm * 0.45  # 0.30–0.75 duty cycle
        phase = 0.5 * (1.0 + np.sin(2 * np.pi * mod_freq * t))
        texture_mod = np.where(phase >= 1.0 - duty, 1.0, 0.15)
        if dens_norm > 0.75:
            jitter = np.random.uniform(-0.12, 0.12, size=len(t))
            texture_mod = texture_mod + jitter
    texture_mod = np.clip(texture_mod, 0.0, 1.0)

    # Envelope from gradient.
    if gradient == 0:
        envelope = np.ones_like(t)
    else:
        grad_norm = float(np.clip(gradient / 100.0, -1.0, 1.0))
        slope = np.linspace(0.0, 1.0, n_samples)
        if grad_norm > 0:
            start_val = 1.0 - grad_norm
            envelope = start_val + (1.0 - start_val) * slope
        else:
            end_val = 1.0 + grad_norm
            envelope = 1.0 - (1.0 - end_val) * slope
    envelope = np.clip(envelope, 0.0, 1.0)

    master = np.clip(amp_norm * 0.6 * envelope * texture_mod, 0.0, 1.0)
    data_left = master * left_gain
    data_right = master * right_gain
    data = np.column_stack((data_left, data_right))

    # Provide a smoothed plot waveform for nicer visualization.
    window = np.ones(max(int(sample_rate * 0.08), 3))
    window /= window.sum()
    plot_waveform = np.convolve(master * max(left_gain, right_gain, 1e-6), window, mode="same")
    meta = {"fs": sample_rate, "plot_waveform": plot_waveform}
    return t, data, meta
