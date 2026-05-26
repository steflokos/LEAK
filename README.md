# LEAK — Leakage Evaluation Analysis Kit

**LEAK** is a modular, high-performance Python Qt side-channel analysis framework designed for hardware security testing with the **ChipWhisperer Nano**. The application automates target firmware compilation, handles real-time power trace acquisition, and features a fully parallelized, vectorized **Correlation Power Analysis (CPA)** engine alongside an advanced Digital Signal Processing (DSP) toolbox to break cryptographic implementations like AES-128.

---

## Key Performance Enhancements

* **Vectorized CPA Core:** The attack algorithm is fully vectorized using BLAS-optimized NumPy matrix operations. By eliminating nested Python loops, the entire 256 key-hypothesis search space is evaluated simultaneously, shifting computing bottlenecks to highly optimized C-level code.
* **Non-Blocking Main-Thread Capture:** Bypasses thread signal conflicts inherent to hardware USB drivers (`libusb`/`pyusb`) by implementing a cooperative multitasking `QTimer` queue. This keeps the user interface responsive and allows for real-time trace graphing without risking application crashes.
* **Modular DSP Pipeline:** Equipped with interactive pre-processing toolboxes including Jitter Peak Alignment, zero-phase Butterworth Bandpass Filtering, and 1D Gaussian Smoothing to isolate clock harmonics and eliminate high-frequency thermal noise.

---

## Architecture & Data Flow

The platform separates tasks into a three-step pipeline mapped across dedicated user interface tabs:

```
[1. Upload & Compile] ---> [2. Capture Traces] ---> [3. Analysis & Cracking]
   (ARM GCC Toolchain)         (CW Nano Hardware)         (Multithreaded CPA Engine)

```

1. **Compilation Staging:** Raw `.c` and `.h` target source files are mirrored to an isolated `.build/` scratchpad where a portable toolchain cross-compiles native target firmware architectures.
2. **Automated Run Isolation:** Power traces captured from the scope are instantly written into permanent, timestamp-mapped run folders (`captured_runs/run_YYYYMMDD_HHMMSS_{N}traces/`) containing contiguous NumPy array configurations (`traces.npy`, `textins.npy`, `keys.npy`). This prevents compilation tasks from overwriting active physical datasets.
3.  GIL-Free Thread Pools: The math logic releases Python's Global Interpreter Lock (GIL) through native NumPy arrays, allowing the system to scale calculations concurrently across all available hardware CPU threads.

---

## Step-by-Step Execution Guide

### Step 1: Upload & Compile Firmware

1. Navigate to the **1. Upload & Compile** tab.
2. Direct the file path fields to your target firmware directories using the **Browse...** buttons (default options populate from `assets/` if available).
3. Click **Compile Firmware**.
* *Note: If running the application for the first time, the background manager will automatically download and extract a portable version of the ARM GCC Toolchain into a hidden directory (`.toolchain/`). Progress can be observed via the dedicated download progress status tracker.*



### Step 2: Connect & Flash the Target Hardware

1. Move to the **2. Capture Traces** tab.
2. Connect your physical ChipWhisperer Nano to your machine's USB port and click **Connect to CW Nano**. The indicator text will shift to a green **Status: Connected** message.
3. Under the **Target Firmware** group, browse and select the freshly compiled `.hex` asset located inside the local `.build/` workspace directory.
4. Click **Flash Chip** to trigger the background programmer, which resets the STM32 target and loads the new binaries.

### Step 3: Capture a Dataset Run

1. In the **Capture Settings** group, modify the **Number of Traces** selection box to your desired transaction size (e.g., 50 to 1,000 traces depending on noise factors).
2. Click **Start Capture**.
3. The platform will dynamically update the progress tracking widgets. If an error occurs or a sequence needs to be aborted, click the **Cancel** button to safely halt hardware transactions and cleanly log the run data captured up to that fraction of a second.

### Step 4: Analyze and Execute the CPA Attack

1. Navigate to the **3. Analysis & Cracking** tab and click **Load Captured Run Folder**.
2. Select your targeted timestamped run directory from the `captured_runs/` storage tree.
3. Configure your filtering parameters inside the **DSP Signal Preprocessing Toolbox**:
* **Jitter Alignment:** Aligns clock drift anomalies by correlating waves against a golden sample template across a designated range.
* **Frequency Bandpass:** Slices away out-of-band low-frequency noise and parasitic radio frequencies.
* **Gaussian Blur:** Dampens ambient thermal noise by smoothing out sample peaks.


4. Set your **CPU Engine Threads** budget and click **Execute CPA Attack**.
5. Once complete, inspect the subkey leakage properties on your visualization dashboards, and use the **Inspect Key Byte Index** modifier widget to rapidly scan and step through correlation profiles across all 16 target positions of the recovered key.