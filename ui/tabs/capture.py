import os
import time
from pathlib import Path
import numpy as np
from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QLabel,
    QLineEdit,
    QFileDialog,
    QGroupBox,
    QTextEdit,
    QSpinBox,
    QProgressBar,
    QMessageBox,
    QCheckBox,      # <-- NEW
    QFormLayout,    # <-- NEW
)
from PyQt6.QtCore import Qt, QTimer
from core.hardware import HardwareManager
import chipwhisperer as cw
import logging                                  # <-- NEW
from PyQt6.QtCore import Qt, QTimer, QObject, pyqtSignal  # <-- ADDED QObject and pyqtSignal
from PyQt6.QtWidgets import QApplication
import json                                     # <-- NEW
from datetime import datetime, timezone         # <-- NEW
import sys


class OutputInterceptor(QObject):
    """Intercepts all raw print() statements and system terminal output."""
    log_signal = pyqtSignal(str)

    def write(self, text):
        # Strip trailing newlines because text_log.append() adds its own automatically
        clean_text = text.strip()
        if clean_text:
            self.log_signal.emit(clean_text)

    def flush(self):
        pass


class CaptureTab(QWidget):
    def __init__(self):
        super().__init__()
        self.hw = HardwareManager()

        self.capture_timer = QTimer()
        self.capture_timer.timeout.connect(self._capture_single_trace_step)

        self.current_trace_idx = 0
        self.total_traces_requested = 0
        self.ktp = None
        self.fixed_key = None

        self.last_traces = None
        self.last_textins = None
        self.last_keys = None

        self._setup_ui()

        self.terminal_interceptor = OutputInterceptor()
        self.terminal_interceptor.log_signal.connect(self.log_message)
        
        # Reroute standard output and standard errors directly to our UI
        sys.stdout = self.terminal_interceptor
        sys.stderr = self.terminal_interceptor

    def _setup_ui(self):
        layout = QVBoxLayout()
        layout.setSpacing(15)

        # --- 1. Connection Group ---
        conn_group = QGroupBox("Hardware Connection")
        conn_layout = QHBoxLayout()
        self.lbl_status = QLabel("Status: Disconnected")
        self.lbl_status.setStyleSheet("color: red; font-weight: bold;")
        self.btn_connect = QPushButton("Connect to CW Nano")
        self.btn_connect.clicked.connect(self.toggle_connection)
        conn_layout.addWidget(self.lbl_status)
        conn_layout.addWidget(self.btn_connect)
        conn_group.setLayout(conn_layout)
        layout.addWidget(conn_group)

        # --- 2. Firmware Flashing Group ---
        flash_group = QGroupBox("Target Firmware")
        flash_layout = QHBoxLayout()
        self.path_hex = QLineEdit()
        self.path_hex.setReadOnly(True)
        self.path_hex.setPlaceholderText("Select .hex file (or compile in tab 1)...")
        btn_browse = QPushButton("Browse...")
        btn_browse.clicked.connect(self.browse_hex)
        self.btn_flash = QPushButton("Flash Chip")
        self.btn_flash.clicked.connect(self.start_flash)
        self.btn_flash.setEnabled(False)
        flash_layout.addWidget(self.path_hex)
        flash_layout.addWidget(btn_browse)
        flash_layout.addWidget(self.btn_flash)
        flash_group.setLayout(flash_layout)
        layout.addWidget(flash_group)

        # --- 3. Trace Capture Group ---
        cap_group = QGroupBox("Capture Settings")
        cap_layout = QVBoxLayout()

        settings_layout = QHBoxLayout()
        settings_layout.addWidget(QLabel("Number of Traces:"))
        self.spin_traces = QSpinBox()
        self.spin_traces.setRange(10, 500000)
        self.spin_traces.setValue(50)
        settings_layout.addWidget(self.spin_traces)
        settings_layout.addStretch()

        btn_action_layout = QHBoxLayout()
        self.btn_capture = QPushButton("Start Capture")
        self.btn_capture.setMinimumHeight(40)
        self.btn_capture.setStyleSheet("font-weight: bold;")
        self.btn_capture.clicked.connect(self.start_capture)
        self.btn_capture.setEnabled(False)

        self.btn_cancel = QPushButton("Cancel")
        self.btn_cancel.setMinimumHeight(40)
        self.btn_cancel.setEnabled(False)
        self.btn_cancel.clicked.connect(self.cancel_capture)

        btn_action_layout.addWidget(self.btn_capture, stretch=3)
        btn_action_layout.addWidget(self.btn_cancel, stretch=1)

        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)

        self.lbl_validation = QLabel("Validation Status: Waiting for capture...")
        self.lbl_validation.setStyleSheet("font-weight: bold; color: #aaaaaa;")
        cap_layout.addLayout(settings_layout)
        cap_layout.addLayout(btn_action_layout)
        cap_layout.addWidget(self.progress_bar)
        cap_layout.addWidget(self.lbl_validation)
        cap_group.setLayout(cap_layout)
        layout.addWidget(cap_group)

        # --- 4. Log Output ---
        log_group = QGroupBox("Hardware Log")
        log_layout = QVBoxLayout()
        self.text_log = QTextEdit()
        self.text_log.setReadOnly(True)
        self.text_log.setStyleSheet(
            "background-color: #1e1e1e; color: #00ff00; font-family: monospace;"
        )
        log_layout.addWidget(self.text_log)
        log_group.setLayout(log_layout)
        layout.addWidget(log_group)

        # --- NEW: Scope & Target Hardware Settings Group ---
        hw_settings_group = QGroupBox("Scope & Target Configuration")
        hw_settings_layout = QFormLayout()
        hw_settings_layout.setSpacing(10)

        # 1. Samples Box (100 to 50k, default 5000)
        self.spin_samples = QSpinBox()
        self.spin_samples.setRange(100, 50000)
        self.spin_samples.setValue(5000)
        hw_settings_layout.addRow("ADC Samples:", self.spin_samples)

        # 2. Random Plaintext and Key Checkboxes (Pre-checked)
        self.chk_random_plaintext = QCheckBox("Enable Random Plaintexts")
        self.chk_random_plaintext.setChecked(True)
        hw_settings_layout.addRow(self.chk_random_plaintext)

        self.chk_random_key = QCheckBox("Enable Random Keys")
        self.chk_random_key.setChecked(False)
        hw_settings_layout.addRow(self.chk_random_key)

        # 3. Greyed out Box stating MCU frequency
        self.txt_mcu_speed = QLineEdit("7.5 MHz")
        self.txt_mcu_speed.setReadOnly(True)
        self.txt_mcu_speed.setEnabled(False)  # Makes it look visually greyed out
        hw_settings_layout.addRow("MCU Clock Speed:", self.txt_mcu_speed)

        hw_settings_group.setLayout(hw_settings_layout)
        layout.addWidget(hw_settings_group) # Add it directly into your main layout hierarchy

        self.setLayout(layout)

    def log_message(self, message):
        self.text_log.append(message)
        # --- NEW: Force the UI to refresh instantly ---
        QApplication.processEvents()

    def toggle_connection(self):
        if not self.hw.is_connected:
            try:
                self.log_message("Attempting to connect to CW Nano...")
                self.hw.connect()
                self.lbl_status.setText("Status: Connected")
                self.lbl_status.setStyleSheet("color: green; font-weight: bold;")
                self.btn_connect.setText("Disconnect")
                self.btn_flash.setEnabled(True)
                self.btn_capture.setEnabled(True)
                self.log_message("SUCCESS: Connected to ChipWhisperer Nano.")
            except Exception as e:
                QMessageBox.critical(self, "Connection Error", str(e))
                self.log_message(f"ERROR: {str(e)}")
        else:
            self.hw.disconnect()
            self.lbl_status.setText("Status: Disconnected")
            self.lbl_status.setStyleSheet("color: red; font-weight: bold;")
            self.btn_connect.setText("Connect to CW Nano")
            self.btn_flash.setEnabled(False)
            self.btn_capture.setEnabled(False)
            self.log_message("Disconnected from hardware.")

    def browse_hex(self):
        start_dir = str(Path(__file__).resolve().parent.parent.parent / ".build")
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Firmware", start_dir, "HEX Files (*.hex)"
        )
        if file_path:
            self.path_hex.setText(file_path)

    def start_flash(self):
        hex_path = self.path_hex.text()
        if not hex_path:
            QMessageBox.warning(self, "No File", "Please select a .hex file to flash.")
            return
        self.btn_flash.setEnabled(False)
        self.btn_capture.setEnabled(False)
        self.text_log.clear()
        try:
            self.log_message(f"Flashing firmware: {Path(hex_path).name}...")
            self.hw.flash_firmware(hex_path)
            self.log_message("SUCCESS: Firmware flashed and chip rebooted.")
        except Exception as e:
            self.log_message(f"ERROR: Flashing failed - {str(e)}")
        finally:
            self.btn_flash.setEnabled(True)
            self.btn_capture.setEnabled(True)

    def start_capture(self):
        if not self.hw.is_connected:
            return
            
        self.total_traces_requested = self.spin_traces.value()
        self.current_trace_idx = 0

        self.btn_flash.setEnabled(False)
        self.btn_capture.setEnabled(False)
        self.btn_connect.setEnabled(False)
        self.btn_cancel.setEnabled(True)

        self.progress_bar.setValue(0)
        self.progress_bar.setMaximum(self.total_traces_requested)

        # --- NEW: Reset validation label ---
        self.lbl_validation.setStyleSheet("font-weight: bold; color: #aaaaaa;")
        self.lbl_validation.setText("Validation Status: Capturing...")
        self.text_log.clear()

        # --- NEW: Read UI Settings ---
        self.use_random_pt = self.chk_random_plaintext.isChecked()
        self.use_random_key = self.chk_random_key.isChecked()
        
        # Override the hardware.py default with our UI Spinbox!
        self.hw.scope.adc.samples = self.spin_samples.value()

        self.log_message(
            f"Initializing hardware array spaces for {self.total_traces_requested} traces..."
        )

        try:
            num_samples = self.hw.scope.adc.samples
            self.last_traces = np.empty(
                (self.total_traces_requested, num_samples), dtype=np.float32
            )
            self.last_textins = np.empty(
                (self.total_traces_requested, 16), dtype=np.uint8
            )
            self.last_keys = np.empty((self.total_traces_requested, 16), dtype=np.uint8)
            self.last_ciphers = np.empty((self.total_traces_requested, 16), dtype=np.uint8)
            
            self.hw.scope.io.nrst = False  # Assert reset (Low)
            time.sleep(0.05)
            self.hw.scope.io.nrst = True   # Release reset (High)
            time.sleep(0.05)
            self.hw.target.flush()

            # --- NEW: Setup Baseline Cryptography ---
            # Generate static fallbacks in case the user disabled randomization
            self.fixed_key = bytearray(os.urandom(16))
            self.fixed_pt = bytearray(os.urandom(16))

            # Send the base key to the target
            self.hw.target.simpleserial_write("k", self.fixed_key)
            time.sleep(0.05)
            self.hw.target.flush()

            self.log_message(
                "Hardware initialized. Starting main-thread loop transaction..."
            )
            self.capture_timer.start(0)
        except Exception as e:
            self.log_message(f"CRITICAL INIT ERROR: {str(e)}")
            self._end_capture_sequence(success=False)

    def _capture_single_trace_step(self):
        if self.current_trace_idx >= self.total_traces_requested:
            self._end_capture_sequence(success=True)
            return
        try:
            import os
            
            # 1. FIX: Clear out any lingering garbage from the previous trace
            self.hw.target.flush()
            
            # --- Dynamic Key Injection ---
            if self.use_random_key:
                current_key = bytearray(os.urandom(16))
                self.hw.target.simpleserial_write("k", current_key)
                
                # 2. FIX: Wait a fraction of a second and flush the Key ACK byte!
                time.sleep(0.01) 
                self.hw.target.flush()
            else:
                current_key = self.fixed_key

            # --- Dynamic Plaintext Injection ---
            if self.use_random_pt:
                current_pt = bytearray(os.urandom(16))
            else:
                current_pt = self.fixed_pt

            self.hw.scope.arm()
            self.hw.target.simpleserial_write("p", current_pt)

            if self.hw.scope.capture():
                self.log_message(f"ERROR: Timeout on trace {self.current_trace_idx}. Target failed to trigger.")
                self.hw.target.flush()
                self._end_capture_sequence(success=False) 
                return

            wave = self.hw.scope.get_last_trace()
            ciphertext_raw = self.hw.target.simpleserial_read("r", 16, timeout=250)
            
            # 3. FIX: Catch 'None' explicitly to prevent hard crashes if the target misfires
            if ciphertext_raw is None:
                self.log_message(f"ERROR: Serial timeout on trace {self.current_trace_idx}. No ciphertext received.")
                self._end_capture_sequence(success=False)
                return

            self.hw.target.flush()

            idx = self.current_trace_idx
            
            # Record it into the pre-allocated array space
            self.last_traces[idx] = wave
            self.last_textins[idx] = current_pt   
            self.last_keys[idx] = current_key     
            self.last_ciphers[idx] = list(ciphertext_raw)

            self.current_trace_idx += 1
            self.progress_bar.setValue(self.current_trace_idx)
        except Exception as e:
            self.capture_timer.stop()
            self.log_message(f"CRITICAL ERROR mid-capture: {str(e)}")
            self._end_capture_sequence(success=False)

    def cancel_capture(self):
        self.log_message("Capture sequence aborted by user.")
        self._end_capture_sequence(success=False)

    def _end_capture_sequence(self, success):
        self.capture_timer.stop()
        self.btn_flash.setEnabled(True)
        self.btn_capture.setEnabled(True)
        self.btn_connect.setEnabled(True)
        self.btn_cancel.setEnabled(False)

        if self.current_trace_idx > 0:
            self.last_traces = self.last_traces[: self.current_trace_idx]
            self.last_textins = self.last_textins[: self.current_trace_idx]
            self.last_keys = self.last_keys[: self.current_trace_idx]

            # ADD THIS LINE:
            self.last_ciphers = self.last_ciphers[: self.current_trace_idx]

            # UPDATE THIS CALL to pass the array:
            self._auto_save_run(self.last_traces, self.last_textins, self.last_keys, self.last_ciphers)
        else:
            QMessageBox.critical(self, "Capture Ended", "No traces gathered.")
            self.progress_bar.setValue(0)

    def _auto_save_run(self, traces, textins, keys, ciphers):
        """Creates a dedicated timestamped folder inside captured_runs/"""
        root_dir = Path(__file__).resolve().parent.parent.parent
        runs_base_dir = root_dir / "captured_runs"
        runs_base_dir.mkdir(exist_ok=True)

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        run_folder_name = f"run_{timestamp}_{len(traces)}_traces"
        run_dir = runs_base_dir / run_folder_name
        run_dir.mkdir(exist_ok=True)

        try:
            self.log_message(
                f"Saving dataset directly into permanent storage folder: {run_folder_name}..."
            )
            
            np.save(run_dir / "traces.npy", traces)
            np.save(run_dir / "plaintexts.npy", textins)
            
            # --- NEW: Smart Key Export for Cross-Tool Compatibility ---
            if self.chk_random_key.isChecked():
                # Advanced Mode: Save the full 2D matrix of shifting keys
                np.save(run_dir / "key.npy", keys)
            else:
                # Standard Mode: Extract the first row and save as a 1D array
                np.save(run_dir / "key.npy", keys[0] if len(keys) > 0 else keys)
                
            np.save(run_dir / "ciphertexts.npy", ciphers)

            # --- RUN VALIDATION LOGIC ---
            validation_results = self._validate_ciphertexts(textins, keys, ciphers)
            # --- NEW: Update UI Label ---
            percent = validation_results["percent"]
            if percent == 100.0:
                self.lbl_validation.setStyleSheet("font-weight: bold; color: #00ff00;") # Green
                self.lbl_validation.setText(f"Validation Status: SUCCESS ({percent}% Matched)")
            else:
                self.lbl_validation.setStyleSheet("font-weight: bold; color: #ff5555;") # Red
                self.lbl_validation.setText(f"Validation Status: FAILED ({percent}% Matched)")
            num_samples = traces.shape[1] if len(traces) > 0 else 0
            key_hex_str = bytes(self.fixed_key).hex() if self.fixed_key is not None else ""
            
            device_serial = "unknown"
            if self.hw.is_connected and hasattr(self.hw.scope, 'sn'):
                device_serial = self.hw.scope.sn

            metadata = {
                "device_serial": device_serial,
                "user_id": 36,  
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "num_traces": len(traces),
                "num_samples": self.spin_samples.value(),            # <-- CHANGED: Uses your spin box value
                "key_hex": key_hex_str,
                "random_key": self.chk_random_key.isChecked(),       # <-- CHANGED: Uses checkbox state
                "random_plaintext": self.chk_random_plaintext.isChecked(), # <-- CHANGED: Uses checkbox state
                "capture_name": run_folder_name,
                "traces_per_file": None,
                "firmware_hash": "N/A",     
                "firmware_name": Path(self.path_hex.text()).stem if self.path_hex.text() else "test",
                "firmware_profile": {
                    "key_bytes": 16,
                    "block_size": 16,
                    "algorithm": "aes-128",
                    "sbox": "aes"
                },
                "target_settings": {
                    "baud": 38400,
                    "protocol_version": "auto"
                },
                "io_settings": {
                    "tio1": "None",
                    "tio2": "None",
                    "tio3": "None",
                    "tio4": "high_z",
                    "nrst": "True",
                    "target_pwr": None
                },
                "scope_settings": {},
                "read_timeout_ms": 10,
                
                # --- INJECT VALIDATION RESULTS HERE ---
                "validation": validation_results
            }

            with open(run_dir / "metadata.json", "w") as f:
                json.dump(metadata, f, indent=2)

            self.log_message(
                f"SUCCESS: Dataset saved completely inside {run_dir.name}/"
            )
            
            # Update success message to include validation status
            QMessageBox.information(
                self,
                "Capture Complete",
                f"Traces captured and safely archived into permanent storage folder:\n\n{run_folder_name}\n\nValidation: {validation_results['percent']}% matched.",
            )
        except Exception as e:
            self.log_message(f"ERROR saving automated run metadata: {str(e)}")


    def _validate_ciphertexts(self, textins, keys, ciphers):
        self.log_message("Validating hardware ciphertexts against AES-128 software reference...")
        try:
            from Crypto.Cipher import AES
        except ImportError:
            self.log_message("WARNING: 'pycryptodome' library not installed. Skipping validation.")
            return {
                "algorithm": "aes-128", "total": len(textins), "matched": 0, 
                "mismatched": 0, "percent": 0.0, "skipped": True, "mismatches": []
            }

        total = len(textins)
        matched = 0
        mismatches = []

        for i in range(total):
            pt = bytes(textins[i])
            k = bytes(keys[i])
            captured_ct = bytes(ciphers[i])

            # Software AES-128 ECB encryption using PyCryptodome
            cipher = AES.new(k, AES.MODE_ECB)
            expected_ct = cipher.encrypt(pt)

            if expected_ct == captured_ct:
                matched += 1
            else:
                mismatches.append(i)

        mismatched = total - matched
        percent = (matched / total) * 100.0 if total > 0 else 0.0

        if mismatched > 0:
            self.log_message(f"WARNING: {mismatched} traces failed validation!")
        else:
            self.log_message("SUCCESS: 100% of ciphertexts matched the expected AES output.")

        return {
            "algorithm": "aes-128",
            "total": total,
            "matched": matched,
            "mismatched": mismatched,
            "percent": round(percent, 2),
            "skipped": False,
            "mismatches": mismatches
        }
