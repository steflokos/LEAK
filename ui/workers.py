import shutil
from pathlib import Path
import concurrent.futures
import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal

from core.compiler import CompilerManager
from core.analysis import (
    AES128_SBox_Out, apply_gaussian_filter, apply_bandpass_filter, 
    apply_max_pooling, apply_sum_pooling, apply_peak_alignment, apply_poc_alignment, apply_dtw_alignment,
    apply_segment_alignment, apply_peak_slicing, apply_wavelet_denoising,
    apply_pca_filtering, apply_fft_magnitude,
    analyze_byte, compute_pge, compute_tvla, LEAKAGE_MODELS
)
from core.analysis import (
    AES128_SBox_Out, apply_dsp_pipeline, analyze_byte, compute_pge, compute_tvla, LEAKAGE_MODELS
)

class CompileWorker(QThread):
    """
    Asynchronously manages portable toolchain verification and native firmware 
    compilation using generic structural path inputs.
    """
    log_signal = pyqtSignal(str)
    download_progress_signal = pyqtSignal(int, int)  # bytes_downloaded, total_bytes
    finished_signal = pyqtSignal(bool, str)

    def __init__(self, source_files: list):
        super().__init__()
        self.source_files = [Path(f) for f in source_files if f]
        self._is_cancelled = False

    def cancel(self):
        self._is_cancelled = True

    def _status_callback(self, message: str):
        self.log_signal.emit(message)

    def _progress_callback(self, current: int, total: int) -> bool:
        self.download_progress_signal.emit(current, total)
        return not self._is_cancelled

    def run(self):
        try:
            manager = CompilerManager()
            toolchain_ready = manager.ensure_toolchain(
                status_callback=self._status_callback, progress_callback=self._progress_callback
            )

            if not toolchain_ready:
                self.log_signal.emit("Process cancelled: Compiler toolchain setup aborted.")
                self.finished_signal.emit(False, "")
                return

            build_dir = manager.root_dir / ".build"
            build_dir.mkdir(exist_ok=True)
            self.log_signal.emit(f"Preparing workspace in {build_dir.name}/")

            # Dynamically copy files to local build cache space
            copied_paths = []
            for src in self.source_files:
                dest = build_dir / src.name
                shutil.copy(src, dest)
                copied_paths.append(dest)

            output_hex = build_dir / "firmware.hex"
            cw_firmware_dir = manager.root_dir / "assets" / "cw_firmware"

            # Dynamically split entry point files away from peripheral dependencies
            main_c = next((p for p in copied_paths if p.name == "main.c"), copied_paths[0])
            dep_files = [p for p in copied_paths if p.suffix == ".c" and p != main_c]

            self.log_signal.emit("Executing gcc compilation flags...")
            success = manager.compile_firmware(main_c, dep_files, output_hex, cw_firmware_dir)

            if success:
                self.log_signal.emit(f"SUCCESS: Firmware generated at {output_hex.name}")
                self.finished_signal.emit(True, str(output_hex))
            else:
                self.log_signal.emit("ERROR: Compilation failed.")
                self.finished_signal.emit(False, "")

        except Exception as e:
            self.log_signal.emit(f"CRITICAL ERROR: {str(e)}")
            self.finished_signal.emit(False, str(e))


class FlashWorker(QThread):
    log_signal = pyqtSignal(str)
    finished_signal = pyqtSignal(bool)

    def __init__(self, hw_manager, hex_path: str):
        super().__init__()
        self.hw = hw_manager
        self.hex_path = hex_path

    def run(self):
        try:
            self.log_signal.emit(f"Flashing firmware: {Path(self.hex_path).name}...")
            self.hw.flash_firmware(self.hex_path)
            self.log_signal.emit("SUCCESS: Firmware flashed and chip rebooted.")
            self.finished_signal.emit(True)
        except Exception as e:
            self.log_signal.emit(f"ERROR: Flashing failed - {str(e)}")
            self.finished_signal.emit(False)


class CaptureWorker(QThread):
    log_signal = pyqtSignal(str)
    progress_signal = pyqtSignal(int, int)
    finished_signal = pyqtSignal(bool, object, object, object)

    def __init__(self, hw_manager, num_traces: int):
        super().__init__()
        self.hw = hw_manager
        self.num_traces = num_traces
        self._is_cancelled = False

    def cancel(self):
        self._is_cancelled = True

    def _progress_callback(self, current: int, total: int) -> bool:
        self.progress_signal.emit(current, total)
        return not self._is_cancelled

    def run(self):
        try:
            self.log_signal.emit(f"Starting capture of {self.num_traces} traces...")
            traces, textins, keys = self.hw.capture_traces(
                self.num_traces, progress_callback=self._progress_callback
            )
            if self._is_cancelled:
                self.log_signal.emit("Acquisition safely cancelled by user.")
                self.finished_signal.emit(False, traces, textins, keys)
            else:
                self.log_signal.emit("SUCCESS: Trace capture complete.")
                self.finished_signal.emit(True, traces, textins, keys)
        except Exception as e:
            self.log_signal.emit(f"ERROR: Capture failed - {str(e)}")
            self.finished_signal.emit(False, None, None, None)


class AnalysisWorker(QThread):
    progress_signal = pyqtSignal(int, int)      
    finished_signal = pyqtSignal(list, object, list, object)  

    def __init__(self, traces, textins, dsp_settings, num_threads=None, true_keys=None):
        super().__init__()
        self.traces = traces
        self.textins = textins
        self.dsp = dsp_settings
        self.num_threads = num_threads
        self.true_keys = true_keys

    def run(self):
        # 1. Pipeline Execution
        # This guarantees the Worker uses the EXACT same math as the UI Preview
        working_traces = apply_dsp_pipeline(self.traces, self.dsp)

        # 2. Dynamic Multi-Cipher Leakage Engagement
        model_name = self.dsp.get('leakage_model', "AES-128 S-Box Output (Round 1)")
        leakage_model_class = next((m for m in LEAKAGE_MODELS if m.name == model_name), AES128_SBox_Out)
        
        num_targets = leakage_model_class.num_targets
        guess_space = leakage_model_class.guess_space
        num_samples = working_traces.shape[1]
        
        recovered_key = [0] * num_targets
        all_correlations = np.empty((num_targets, guess_space, num_samples), dtype=np.float64)
        pge_list = [guess_space - 1] * num_targets
        tvla_matrix = np.zeros((num_targets, num_samples), dtype=np.float64)
        completed = 0
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.num_threads) as executor:
            futures = {
                executor.submit(analyze_byte, b, working_traces, self.textins[:, b], leakage_model_class): b 
                for b in range(num_targets)
            }
            
            for future in concurrent.futures.as_completed(futures):
                bnum, best_guess, byte_corr = future.result()
                recovered_key[bnum] = best_guess
                all_correlations[bnum] = byte_corr
                
                if self.true_keys is not None:
                    t_key_byte = int(self.true_keys[bnum] if self.true_keys.ndim == 1 else self.true_keys[0, bnum])
                    pge_list[bnum] = compute_pge(byte_corr, t_key_byte)
                    tvla_matrix[bnum] = compute_tvla(working_traces, self.textins[:, bnum], t_key_byte, leakage_model_class)
                
                completed += 1
                self.progress_signal.emit(completed, num_targets)

        self.finished_signal.emit(recovered_key, all_correlations, pge_list, tvla_matrix)


class AutoSniperWorker(QThread):
    progress_signal = pyqtSignal(int, int, str)
    finished_signal = pyqtSignal(list, list)

    def __init__(self, traces, textins, base_dsp, true_keys=None):
        super().__init__()
        self.traces = traces
        self.textins = textins
        self.base_dsp = base_dsp
        self.true_keys = true_keys

    def run(self):
        # We need the unified pipeline to execute the filter combinations
        from core.analysis import apply_dsp_pipeline
        
        model_name = self.base_dsp.get('leakage_model', "AES-128 S-Box Output (Round 1)")
        leakage_model_class = next((m for m in LEAKAGE_MODELS if m.name == model_name), AES128_SBox_Out)
        num_targets = leakage_model_class.num_targets
        
        recovered_key = [0] * num_targets
        best_pges = [255] * num_targets

        # THE SNIPER GRID: The combinations of filters to test per-byte
        prom_values = [0.05, 0.08, 0.10, 0.12, 0.15]
        sigma_values = [None, 1.0, 1.5, 2.0]
        probability_matrix = np.zeros((16, 12), dtype=np.uint8)
        for b in range(num_targets):
            found_zero = False
            best_pge = 255
            best_guess = 0
            t_key_byte = int(self.true_keys[b] if self.true_keys.ndim == 1 else self.true_keys[0, b])

            for prom in prom_values:
                for sigma in sigma_values:
                    self.progress_signal.emit(b, num_targets, f"Byte {b}: Testing Prom={prom}, Sigma={sigma}")
                    
                    # 1. Modify the DSP settings for this specific test
                    test_dsp = self.base_dsp.copy()
                    test_dsp['slice_prom'] = prom
                    if sigma is None:
                        test_dsp['gauss_enabled'] = False
                    else:
                        test_dsp['gauss_enabled'] = True
                        test_dsp['gauss_sigma'] = sigma

                    # 2. Run the modified pipeline
                    working_traces = apply_dsp_pipeline(self.traces, test_dsp)
                    
                    # 3. Analyze ONLY the current target byte
                    _, guess, byte_corr = analyze_byte(b, working_traces, self.textins[:, b], leakage_model_class)
                    pge = compute_pge(byte_corr, t_key_byte)
                    
                    # Keep track of the best result in case we never hit 0
                    if pge < best_pge:
                        best_pge = pge
                        best_guess = guess

                    # 4. If we cracked it perfectly, abort the loop and move to next byte!
                    if pge == 0:
                        found_zero = True
                        break
                        
                if found_zero:
                    break

            # Lock in the best result for this byte
            recovered_key[b] = best_guess
            best_pges[b] = best_pge
            self.progress_signal.emit(b + 1, num_targets, f"Locked Byte {b}: {best_guess:02x} (PGE {best_pge})")
            byte_corr_top12 = np.argsort(byte_corr)[-12:][::-1] 
            probability_matrix[b] = byte_corr_top12

        self.finished_signal.emit(recovered_key, best_pges)