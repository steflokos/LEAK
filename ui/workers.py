import shutil
from pathlib import Path
import concurrent.futures
import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal

from core.compiler import CompilerManager
from core.analysis import (
    AES128_SBox_Out, apply_dsp_pipeline, analyze_byte, compute_pge, compute_tvla,
    aes128_forward_key_schedule, LEAKAGE_MODELS,
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

    # FIX 1: Add 'ciphers' to the arguments
    def __init__(self, traces, textins, ciphers, dsp_settings, num_threads=None, true_keys=None, batch_count=1):
        super().__init__()
        self.traces = traces
        self.textins = textins
        self.ciphers = ciphers       # <-- NEW
        self.dsp = dsp_settings
        self.num_threads = num_threads
        self.true_keys = true_keys
        self.batch_count = max(1, int(batch_count))

    def run(self):
        n = len(self.traces)
        if self.batch_count > 1:
            batch_size = (n + self.batch_count - 1) // self.batch_count
            processed = []
            for i in range(self.batch_count):
                start = i * batch_size
                end = min(start + batch_size, n)
                if start >= n:
                    break
                processed.append(apply_dsp_pipeline(self.traces[start:end], self.dsp))
                self.progress_signal.emit(i + 1, self.batch_count)
            working_traces = np.concatenate(processed, axis=0)
        else:
            working_traces = apply_dsp_pipeline(self.traces, self.dsp)

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
        
        # Route input data: last-round model uses ciphertexts, all others use plaintexts
        use_ciphers = "Ciphertext" in model_name
        target_data = self.ciphers if (use_ciphers and self.ciphers is not None) else self.textins

        # For PGE/TVLA comparison, last-round model needs k10 (not k0) as the reference.
        comparison_keys = None
        if self.true_keys is not None:
            k0 = self.true_keys if self.true_keys.ndim == 1 else self.true_keys[0]
            if use_ciphers:
                comparison_keys = np.array(aes128_forward_key_schedule(k0.tolist()), dtype=np.uint8)
            else:
                comparison_keys = k0

        # Cap concurrent threads so simultaneous t_dev allocations don't OOM.
        # Each thread allocates t_dev ≈ working_traces.nbytes; budget ~20 GB headroom.
        trace_gb = working_traces.nbytes / (1024 ** 3)
        safe_max = max(1, int(20 / max(trace_gb, 0.001)))
        effective_threads = min(self.num_threads or 16, safe_max)

        with concurrent.futures.ThreadPoolExecutor(max_workers=effective_threads) as executor:
            futures = {
                executor.submit(analyze_byte, b, working_traces, target_data[:, b], leakage_model_class): b
                for b in range(num_targets)
            }

            for future in concurrent.futures.as_completed(futures):
                bnum, best_guess, byte_corr = future.result()
                recovered_key[bnum] = best_guess
                all_correlations[bnum] = byte_corr

                if comparison_keys is not None:
                    t_key_byte = int(comparison_keys[bnum])
                    pge_list[bnum] = compute_pge(byte_corr, t_key_byte)
                    tvla_matrix[bnum] = compute_tvla(working_traces, target_data[:, bnum], t_key_byte, leakage_model_class)

                completed += 1
                self.progress_signal.emit(completed, num_targets)

        self.finished_signal.emit(recovered_key, all_correlations, pge_list, tvla_matrix)


class AutoSniperWorker(QThread):
    progress_signal = pyqtSignal(int, int, str)
    # FIX: Added a 4th object to pass back the best correlation weights
    finished_signal = pyqtSignal(list, list, object, object)

    def __init__(self, traces, textins, ciphers, base_dsp, sniper_config, true_keys=None):
        super().__init__()
        self.traces = traces
        self.textins = textins
        self.ciphers = ciphers     # <-- NEW
        self.base_dsp = base_dsp
        self.sniper_config = sniper_config
        self.true_keys = true_keys

    def run(self):
        import itertools
        from core.analysis import apply_dsp_pipeline, analyze_byte, LEAKAGE_MODELS
        
        model_name = self.base_dsp.get('leakage_model', "AES-128 S-Box Output (Round 1)")
        leakage_model_class = next((m for m in LEAKAGE_MODELS if m.name == model_name), AES128_SBox_Out)
        num_targets = leakage_model_class.num_targets
        
        recovered_key = [0] * num_targets
        best_pges = [255] * num_targets
        
        probability_matrix = np.zeros((num_targets, 256), dtype=np.uint8)
        # FIX: Create a matrix to store the max correlations for the beam search scoring
        best_corrs = np.zeros((num_targets, 256), dtype=np.float64) 
        
        best_metrics = [float('inf')] * num_targets 

        keys_cfg, values_cfg = zip(*self.sniper_config.items())
        combinations = [dict(zip(keys_cfg, v)) for v in itertools.product(*values_cfg)]
        total_combos = len(combinations)
        use_ciphers = "Ciphertext" in model_name
        target_data = self.ciphers if (use_ciphers and self.ciphers is not None) else self.textins

        # Derive comparison keys: last-round model needs k10 for PGE evaluation
        comparison_keys = None
        if self.true_keys is not None:
            k0 = self.true_keys if self.true_keys.ndim == 1 else self.true_keys[0]
            if use_ciphers:
                comparison_keys = np.array(aes128_forward_key_schedule(k0.tolist()), dtype=np.uint8)
            else:
                comparison_keys = k0

        for i, combo in enumerate(combinations):
            combo_str = ", ".join([f"{k}={v}" for k, v in combo.items() if len(self.sniper_config[k]) > 1])
            if not combo_str: combo_str = "Fixed Baseline parameters"
            
            self.progress_signal.emit(0, num_targets, f"Evaluating Config ({i+1}/{total_combos}): [ {combo_str} ]")
            
            test_dsp = self.base_dsp.copy()
            test_dsp.update(combo)
            if test_dsp.get('fft_end') == 0: test_dsp['fft_end'] = None
            
            working_traces = apply_dsp_pipeline(self.traces, test_dsp)
            
            for b in range(num_targets):
                _, guess, byte_corr = analyze_byte(b, working_traces, target_data[:, b], leakage_model_class)
                
                max_corr_per_guess = np.max(byte_corr, axis=1)
                sorted_guesses = np.argsort(max_corr_per_guess)[::-1]
                
                if comparison_keys is not None:
                    t_key = int(comparison_keys[b])
                    metric = int(np.where(sorted_guesses == t_key)[0][0])
                else:
                    metric = -float(max_corr_per_guess[sorted_guesses[0]])

                if metric < best_metrics[b]:
                    best_metrics[b] = metric
                    recovered_key[b] = sorted_guesses[0]
                    best_pges[b] = metric if self.true_keys is not None else 0
                    
                    probability_matrix[b] = sorted_guesses
                    # FIX: Save the raw correlation scores for the beam search
                    best_corrs[b] = max_corr_per_guess 
                    
                    self.progress_signal.emit(b + 1, num_targets, f"Byte {b} Improved: Best Guess {recovered_key[b]:02x}")

        # FIX: Emit best_corrs alongside the probability matrix
        self.finished_signal.emit(recovered_key, best_pges, probability_matrix, best_corrs)