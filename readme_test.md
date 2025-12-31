# Method: Physically-Adaptive Discrete Haptic Signal Generation
# 方法：物理自适应离散触觉信号生成

## 1. System Overview (系统概述)
To investigate user preferences for discrete haptic feedback, we developed a custom haptic generation system using Python 3 and the Pygame library. The system interfaces with a standard Xbox controller equipped with Eccentric Rotating Mass (ERM) motors. To ensure high temporal precision and signal fidelity on a non-real-time operating system (Windows), the system employs a multi-threaded architecture:
1.  **UI Thread:** Handles user interaction and parameter visualization using Tkinter, decoupled from signal processing.
2.  **Haptic Thread:** A high-priority daemon thread dedicated to waveform synthesis and hardware instruction dispatch, ensuring a timing accuracy of $<1\text{ms}$.

## 2. Parameter Mapping & Physical Linearization (参数映射与物理线性化)
Raw inputs from the user interface are normalized to a range of $[20, 100]$. To address the physical non-linearity and deadzones inherent in ERM motors, these inputs are mapped to an **Effective Operating Range** before signal generation.

### 2.1 Intensity (Voltage Control)
The normalized intensity parameter $I_{norm}$ is mapped to the motor voltage amplitude $V$. We established a physical deadzone floor of 0.2 (20%) to prevent non-response at low signal levels due to static friction (stiction).
$$V = 0.2 + I_{norm} \times 0.8$$
Where $V \in [0.2, 1.0]$. This ensures that even the minimum intensity setting produces a perceptible haptic actuation.

### 2.2 Texture (Motor Interpolation)
Texture is simulated by varying the power distribution between the controller's two distinct motors: the left "heavy" weight (low frequency, high amplitude) and the right "light" weight (high frequency, low amplitude). A linear interpolation factor $\alpha$ is applied:
$$P_{left} = V \times \alpha$$
$$P_{right} = V \times (1 - \alpha)$$

### 2.3 Rhythm & Grain (Temporal Constraints)
* **Rhythm (Frequency):** Mapped to a range of $0.3 \text{Hz} - 4.0 \text{Hz}$. The upper limit of 4.0Hz is imposed to prevent signal aliasing caused by the mechanical inertia of the rotors.
* **Grain (Duty Cycle):** Mapped to a duty cycle $D \in [10\%, 70\%]$. The maximum cap of 70% is enforced to guarantee a sufficient "spin-down" period between pulses, ensuring discrete signals remain distinct rather than merging into a continuous "buzz."

## 3. Waveform Synthesis Strategy (波形合成策略)
The system utilizes a **Segment-based Synthesis** approach. A single vibration cycle is decomposed into active segments and passive gaps, subject to rigorous physical constraints.

### 3.1 Dynamic Gap Enforcement (动态间隙保护)
To maintain haptic clarity, the system enforces a **Physical Minimum Gap** ($\delta_{min} = 45\text{ms}$). For any given cycle period $T_{cycle} = 1/f$, the maximum allowable pulse width $T_{pulse}$ is dynamically clamped:
$$T_{pulse} = \min(T_{cycle} \times D, \quad T_{cycle} - \delta_{min})$$
This ensures that regardless of the user's duty cycle setting, the motors are always afforded at least 45ms to decelerate, preserving the "granularity" of the feedback.

### 3.2 Duration Clamping & Tail Dropping (时长截断与末端丢弃)
To ensure strict adherence to the total experimental duration, the generator calculates the remaining time $T_{remain}$ before each pulse.
* **Truncation:** If a pulse length exceeds $T_{remain}$, it is truncated to fit.
* **Dropping:** If $T_{remain}$ is insufficient to render a meaningful pulse, the final segment is dropped entirely.
This logic prevents the "run-over" effect common in loop-based haptic scripts, ensuring consistent stimulus duration across trials.

## 4. Hardware-Timed Temporal Control (硬件定时时序控制)
A critical challenge in PC-based haptics is the latency introduced by USB communication and OS scheduling (typically 10-20ms jitter). We implemented a **"Fire-and-Forget"** protocol to mitigate this.

Instead of the traditional software loop (*Send Start $\to$ Wait $\to$ Send Stop*), our system pre-calculates the exact millisecond duration of the pulse ($T_{pulse}$) and transmits it directly to the controller firmware:
$$\text{Command: } \texttt{Rumble}(L, R, T_{pulse})$$
The controller's onboard firmware handles the termination of the voltage automatically upon timeout. This eliminates the dependency on the host CPU for stopping the motor, resulting in sharper, more precise haptic transients (clicks/bumps) compared to software-stopped signals.

## 5. Implementation Details (实现细节)
* **Hybrid Precision Wait:** The playback thread utilizes a hybrid timing mechanism combining `time.sleep()` for coarse waiting and a busy-wait spin loop for the final millisecond, achieving sub-millisecond precision without excessive CPU overhead.
* **Thread Safety:** A mutex lock (`threading.Lock`) protects the device handle, ensuring robust operation during rapid parameter adjustments or device re-connection events.