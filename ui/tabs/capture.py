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
)
from PyQt6.QtCore import Qt, QTimer
from core.hardware import HardwareManager
import chipwhisperer as cw


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
        self.spin_traces.setRange(10, 100000)
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

        cap_layout.addLayout(settings_layout)
        cap_layout.addLayout(btn_action_layout)
        cap_layout.addWidget(self.progress_bar)
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

        self.setLayout(layout)

    def log_message(self, message):
        self.text_log.append(message)

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
        self.text_log.clear()

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

            self.ktp = cw.ktp.Basic()
            self.hw.scope.io.nrst = False  # Assert reset (Low)
            time.sleep(0.05)
            self.hw.scope.io.nrst = True   # Release reset (High)
            time.sleep(0.05)
            self.hw.target.flush()

            self.fixed_key, _ = self.ktp.next()
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
            _, text = self.ktp.next()
            self.hw.scope.arm()
            self.hw.target.simpleserial_write("p", text)

            if self.hw.scope.capture():
                self.log_message(f"ERROR: Timeout on trace {self.current_trace_idx}. Target failed to trigger.")
                self.hw.target.flush()
                self._end_capture_sequence(success=False) # Halt the timer and reset UI
                return

            wave = self.hw.scope.get_last_trace()
            self.hw.target.simpleserial_read("r", 16, timeout=250)
            self.hw.target.flush()

            idx = self.current_trace_idx
            self.last_traces[idx] = wave
            self.last_textins[idx] = text
            self.last_keys[idx] = self.fixed_key

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

            # Save data automatically to a unique, permanent directory
            self._auto_save_run(self.last_traces, self.last_textins, self.last_keys)
        else:
            QMessageBox.critical(self, "Capture Ended", "No traces gathered.")
            self.progress_bar.setValue(0)

    def _auto_save_run(self, traces, textins, keys):
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
            np.save(run_dir / "textins.npy", textins)
            np.save(run_dir / "keys.npy", keys)

            self.log_message(
                f"SUCCESS: Dataset saved completely inside {run_dir.name}/"
            )
            QMessageBox.information(
                self,
                "Capture Complete",
                f"Traces captured and safely archived into permanent storage folder:\n\n{run_folder_name}",
            )
        except Exception as e:
            self.log_message(f"ERROR saving automated run metadata: {str(e)}")
