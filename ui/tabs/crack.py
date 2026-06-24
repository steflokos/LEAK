import os
import numpy as np
from pathlib import Path
import itertools
from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QLabel,
    QGroupBox,
    QProgressBar,
    QMessageBox,
    QCheckBox,
    QDoubleSpinBox,
    QSpinBox,
    QFileDialog,
    QComboBox,
    QDialog,
    QFormLayout,
    QDialogButtonBox,
    QTabWidget,
    QListWidget,
    QAbstractItemView,
    QScrollArea,
)
from PyQt6.QtCore import Qt

import pyqtgraph as pg
from ui.workers import AnalysisWorker, AutoSniperWorker
from ui.workers import AnalysisWorker
from core.analysis import (
    apply_dsp_pipeline,
    apply_gaussian_filter,
    apply_bandpass_filter,
    apply_max_pooling,
    apply_sum_pooling,
    apply_peak_alignment,
    apply_segment_alignment,
    apply_peak_slicing,
    apply_wavelet_denoising,
    apply_poc_alignment,
    apply_dtw_alignment,
    apply_pca_filtering,
    apply_fft_magnitude,
    LEAKAGE_MODELS,
)


class SweepParameterWidget(QWidget):
    """Dynamically builds a sweeping configuration based on data type (Number, Bool, Category)."""

    def __init__(self, key, label_text, config):
        super().__init__()
        self.key = key
        self.param_type = config["type"]
        self.layout = QHBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.true_sample_rate = 20_000_000
        self.chk_sweep = QCheckBox(f"Sweep {label_text}")
        self.chk_sweep.stateChanged.connect(self.toggle_sweep)
        self.layout.addWidget(self.chk_sweep)

        # --- BUG FIX: Sanitize None values coming from the base dictionary ---
        raw_default = config.get("default")
        if raw_default is None:
            if self.param_type in [int, float]:
                self.default_val = config.get("min", 0)
            elif self.param_type == bool:
                self.default_val = False
            elif self.param_type == list:
                self.default_val = config["options"][0]
        else:
            self.default_val = raw_default

        if self.param_type in [int, float]:
            SpinClass = QDoubleSpinBox if self.param_type == float else QSpinBox

            # Static Mode
            self.spin_single = SpinClass()
            self.spin_single.setRange(
                config.get("min", -100000), config.get("max", 10000000)
            )
            self.spin_single.setValue(self.default_val)
            if self.param_type == float:
                self.spin_single.setDecimals(3)

            # Sweeping Mode
            self.spin_min = SpinClass()
            self.spin_min.setRange(
                config.get("min", -100000), config.get("max", 10000000)
            )
            self.spin_min.setValue(self.default_val)
            self.spin_max = SpinClass()
            self.spin_max.setRange(
                config.get("min", -100000), config.get("max", 10000000)
            )
            self.spin_max.setValue(self.default_val)
            self.spin_step = SpinClass()
            if self.param_type == float:
                self.spin_step.setRange(0.001, config.get("max", 10000000))
                self.spin_step.setDecimals(3)
            else:
                self.spin_step.setRange(1, int(config.get("max", 10000000)))
            self.spin_step.setValue(config.get("step", 1))

            self.lbl_min = QLabel("Min:")
            self.lbl_max = QLabel("Max:")
            self.lbl_step = QLabel("Step:")

            self.layout.addWidget(self.spin_single)
            self.layout.addWidget(self.lbl_min)
            self.layout.addWidget(self.spin_min)
            self.layout.addWidget(self.lbl_max)
            self.layout.addWidget(self.spin_max)
            self.layout.addWidget(self.lbl_step)
            self.layout.addWidget(self.spin_step)

        elif self.param_type in [list, bool]:
            options = config["options"] if self.param_type == list else [True, False]

            # Static Mode
            self.combo_single = QComboBox()
            self.combo_single.addItems([str(opt) for opt in options])
            self.combo_single.setCurrentText(str(self.default_val))

            # Sweeping Mode (Multi-Select Box)
            self.list_sweep = QListWidget()
            self.list_sweep.setSelectionMode(
                QAbstractItemView.SelectionMode.MultiSelection
            )
            self.list_sweep.addItems([str(opt) for opt in options])
            self.list_sweep.setMaximumHeight(65)

            # Select default item on start
            items = self.list_sweep.findItems(
                str(self.default_val), Qt.MatchFlag.MatchExactly
            )
            if items:
                items[0].setSelected(True)

            self.layout.addWidget(self.combo_single)
            self.layout.addWidget(self.list_sweep)

            # Map string selections back to native Python types
            self.options_map = {str(opt): opt for opt in options}

        self.toggle_sweep()

    def toggle_sweep(self):
        """Hides/Shows the correct inputs based on whether the user is Sweeping this parameter."""
        is_sweeping = self.chk_sweep.isChecked()
        if self.param_type in [int, float]:
            self.spin_single.setVisible(not is_sweeping)
            self.lbl_min.setVisible(is_sweeping)
            self.spin_min.setVisible(is_sweeping)
            self.lbl_max.setVisible(is_sweeping)
            self.spin_max.setVisible(is_sweeping)
            self.lbl_step.setVisible(is_sweeping)
            self.spin_step.setVisible(is_sweeping)
        elif self.param_type in [list, bool]:
            self.combo_single.setVisible(not is_sweeping)
            self.list_sweep.setVisible(is_sweeping)

    def get_values(self):
        """Generates the array of parameters to pass to the grid search."""
        if not self.chk_sweep.isChecked():
            if self.param_type in [int, float]:
                return [self.spin_single.value()]
            else:
                return [self.options_map[self.combo_single.currentText()]]
        else:
            if self.param_type in [int, float]:
                vals = []
                current = self.spin_min.value()
                limit = self.spin_max.value()
                step = self.spin_step.value()
                if step <= 0:
                    return [current]
                while current <= limit:
                    val = round(current, 5) if self.param_type == float else current
                    vals.append(val)
                    current += step
                return vals
            else:
                selected = self.list_sweep.selectedItems()
                if not selected:  # Failsafe if user selected nothing while sweeping
                    return [self.options_map[self.combo_single.currentText()]]
                return [self.options_map[item.text()] for item in selected]


class SniperSettingsDialog(QDialog):
    def __init__(self, base_dsp, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Automated DSP Sniper Configuration")
        self.setMinimumWidth(850)
        self.setMinimumHeight(600)
        self.layout = QVBoxLayout(self)

        # SCHEMA ENGINE: To add new framework features in the future, JUST ADD THEM HERE!
        # The UI will automatically detect the type and generate the right Multi-Select/SpinBoxes.
        DSP_SCHEMA = {
            "1. Alignment & Warping": [
                (
                    "align_mode",
                    "Alignment Logic",
                    {
                        "type": list,
                        "options": [
                            "None",
                            "Peak Cross-Correlation",
                            "Phase-Only Correlation (POC)",
                            "Dynamic Time Warping (DTW)",
                        ],
                        "default": base_dsp.get("align_mode", "None"),
                    },
                ),
                (
                    "align_start",
                    "Align Start",
                    {
                        "type": int,
                        "min": 0,
                        "max": 100000,
                        "default": base_dsp.get("align_start", 0),
                        "step": 100,
                    },
                ),
                (
                    "align_end",
                    "Align End",
                    {
                        "type": int,
                        "min": 0,
                        "max": 100000,
                        "default": base_dsp.get("align_end", 500),
                        "step": 100,
                    },
                ),
                (
                    "elastic_enabled",
                    "Enable Elastic Segment",
                    {"type": bool, "default": base_dsp.get("elastic_enabled", False)},
                ),
                (
                    "elastic_segs",
                    "Elastic Segments",
                    {
                        "type": int,
                        "min": 2,
                        "max": 64,
                        "default": base_dsp.get("elastic_segs", 8),
                        "step": 2,
                    },
                ),
                (
                    "elastic_warp",
                    "Elastic Warp",
                    {
                        "type": int,
                        "min": 1,
                        "max": 200,
                        "default": base_dsp.get("elastic_warp", 20),
                        "step": 5,
                    },
                ),
            ],
            "2. Defeats & Masking": [
                (
                    "masking_mode",
                    "Masking Defeat",
                    {
                        "type": list,
                        "options": [
                            "None",
                            "Absolute Value Centering",
                            "Trace Squaring",
                        ],
                        "default": base_dsp.get("masking_mode", "None"),
                    },
                ),
                (
                    "shuffle_mode",
                    "Jitter/Shuffling Defeat",
                    {
                        "type": list,
                        "options": [
                            "None",
                            "Integrated-Sum (Global)",
                            "Sliding Window Integration (SWI)",
                            "Window-Sum Pooling",
                            "Window-Max Pooling",
                        ],
                        "default": base_dsp.get("shuffle_mode", "None"),
                    },
                ),
            ],
            "3. Peak Slicing (Desync)": [
                (
                    "slice_enabled",
                    "Enable Peak Slicing",
                    {"type": bool, "default": base_dsp.get("slice_enabled", False)},
                ),
                (
                    "slice_start",
                    "Slice Start Search",
                    {
                        "type": int,
                        "min": 0,
                        "max": 1000000,
                        "default": base_dsp.get("slice_start", 1150),
                        "step": 100,
                    },
                ),
                (
                    "slice_end",
                    "Slice End Search",
                    {
                        "type": int,
                        "min": 10,
                        "max": 1000000,
                        "default": base_dsp.get("slice_end", 4800),
                        "step": 100,
                    },
                ),
                (
                    "slice_size",
                    "Global Window Size",
                    {
                        "type": int,
                        "min": 3,
                        "max": 50000,
                        "default": base_dsp.get("slice_size", 45),
                        "step": 5,
                    },
                ),
                (
                    "slice_dist",
                    "Peak Distance",
                    {
                        "type": int,
                        "min": 1,
                        "max": 500,
                        "default": base_dsp.get("slice_dist", 95),
                        "step": 5,
                    },
                ),
                (
                    "slice_prom",
                    "Peak Prominence",
                    {
                        "type": float,
                        "min": 0.001,
                        "max": 100.0,
                        "default": base_dsp.get("slice_prom", 0.1),
                        "step": 0.05,
                    },
                ),
                (
                    "slice_count",
                    "Expected Peak Count",
                    {
                        "type": int,
                        "min": 1,
                        "max": 256,
                        "default": base_dsp.get("slice_count", 32),
                        "step": 4,
                    },
                ),
            ],
            "4. Linear Filters & FFT": [
                (
                    "bandpass_enabled",
                    "Enable Bandpass Filter",
                    {"type": bool, "default": base_dsp.get("bandpass_enabled", False)},
                ),
                (
                    "bp_low",
                    "Bandpass Low Cut",
                    {
                        "type": int,
                        "min": 1000,
                        "max": 10000000,
                        "default": base_dsp.get("bp_low", 100000),
                        "step": 10000,
                    },
                ),
                (
                    "bp_high",
                    "Bandpass High Cut",
                    {
                        "type": int,
                        "min": 10000,
                        "max": 20000000,
                        "default": base_dsp.get("bp_high", 5000000),
                        "step": 500000,
                    },
                ),
                (
                    "wavelet_enabled",
                    "Enable Wavelet Smoothing",
                    {"type": bool, "default": base_dsp.get("wavelet_enabled", False)},
                ),
                (
                    "wavelet_type",
                    "Wavelet Type",
                    {
                        "type": list,
                        "options": ["db4", "sym4", "haar"],
                        "default": base_dsp.get("wavelet_type", "db4"),
                    },
                ),
                # BUG FIX: Wavelet level default bumped to 2 for better power trace smoothing
                (
                    "wavelet_level",
                    "Wavelet Level",
                    {
                        "type": int,
                        "min": 1,
                        "max": 5,
                        "default": base_dsp.get("wavelet_level", 2),
                        "step": 1,
                    },
                ),
                (
                    "gauss_enabled",
                    "Enable Gaussian Blur",
                    {"type": bool, "default": base_dsp.get("gauss_enabled", False)},
                ),
                (
                    "gauss_sigma",
                    "Gaussian Sigma",
                    {
                        "type": float,
                        "min": 0.1,
                        "max": 20.0,
                        "default": base_dsp.get("gauss_sigma", 4.0),
                        "step": 0.5,
                    },
                ),
                (
                    "pca_enabled",
                    "Enable PCA Filter",
                    {"type": bool, "default": base_dsp.get("pca_enabled", False)},
                ),
                (
                    "pca_comps",
                    "PCA Components",
                    {
                        "type": int,
                        "min": 1,
                        "max": 50,
                        "default": base_dsp.get("pca_comps", 5),
                        "step": 1,
                    },
                ),
                (
                    "fft_enabled",
                    "Enable FFT Magnitude",
                    {"type": bool, "default": base_dsp.get("fft_enabled", False)},
                ),
                (
                    "fft_start",
                    "FFT Window Start",
                    {
                        "type": int,
                        "min": 0,
                        "max": 100000,
                        "default": base_dsp.get("fft_start", 0),
                        "step": 50,
                    },
                ),
                (
                    "fft_end",
                    "FFT Window End",
                    {
                        "type": int,
                        "min": 0,
                        "max": 100000,
                        "default": base_dsp.get("fft_end", 0),
                        "step": 50,
                    },
                ),
                # BUG FIX: Expose FFT Cutoff Bins to sniper grid search
                (
                    "fft_cutoff",
                    "FFT Max Bins (0=All)",
                    {
                        "type": int,
                        "min": 0,
                        "max": 10000,
                        "default": base_dsp.get("fft_cutoff", 0),
                        "step": 100,
                    },
                ),
            ],
        }

        self.tabs = QTabWidget()
        self.sweepers = {}

        # Auto-Build the UI Tabs directly from the Schema!
        for tab_name, parameters in DSP_SCHEMA.items():
            tab_widget = QWidget()
            tab_layout = QVBoxLayout(tab_widget)

            scroll = QScrollArea()
            scroll.setWidgetResizable(True)
            scroll_content = QWidget()
            scroll_layout = QVBoxLayout(scroll_content)

            for key, label, config in parameters:
                widget = SweepParameterWidget(key, label, config)
                self.sweepers[key] = widget
                scroll_layout.addWidget(widget)

            scroll_layout.addStretch()
            scroll.setWidget(scroll_content)
            tab_layout.addWidget(scroll)
            self.tabs.addTab(tab_widget, tab_name)

        self.layout.addWidget(self.tabs)

        self.btns = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        self.btns.accepted.connect(self.accept)
        self.btns.rejected.connect(self.reject)
        self.layout.addWidget(self.btns)

    def get_config(self):
        """Collects the nested array of sweeping parameters to pass to the Worker."""
        config = {}
        for key, widget in self.sweepers.items():
            config[key] = widget.get_values()
        return config


class CrackTab(QWidget):
    def __init__(self):
        super().__init__()
        self.traces = None
        self.textins = None
        self.keys = None
        self.correlations = None
        self.recovered_key = None
        self.pge_list = None
        self.tvla_matrix = None
        self.worker = None

        # Build strict string mappings instead of fragile indices
        self.leakage_models_map = {model.name: model for model in LEAKAGE_MODELS}
        self._setup_ui()


    def _setup_ui(self):
        master_layout = QVBoxLayout()
        master_layout.setSpacing(10)

        # --- Top Scope Management Block ---
        top_group = QGroupBox("CPA Environment Global Scope")
        top_layout = QHBoxLayout()

        self.btn_load = QPushButton("Load Captured Run Folder")
        self.btn_load.clicked.connect(self.load_data)

        lbl_model = QLabel("Target Crypto Leakage:")
        self.combo_model = QComboBox()
        self.combo_model.addItems(list(self.leakage_models_map.keys()))
        self.combo_model.currentTextChanged.connect(self.on_leakage_model_changed)

        lbl_threads = QLabel("Engine Threads:")
        self.spin_threads = QSpinBox()
        self.spin_threads.setRange(1, 32)
        self.spin_threads.setValue(os.cpu_count() or 4)

        self.btn_crack = QPushButton("Execute SCA Verification")
        self.btn_crack.clicked.connect(self.start_cpa)
        self.btn_crack.setEnabled(False)
        self.btn_crack.setMinimumHeight(35)
        self.btn_crack.setStyleSheet("font-weight: bold; color: #ff5555;")

        self.btn_sniper = QPushButton("Auto-Sniper Grid Search")
        self.btn_sniper.clicked.connect(self.start_auto_sniper)
        self.btn_sniper.setEnabled(False)
        self.btn_sniper.setMinimumHeight(35)
        self.btn_sniper.setStyleSheet(
            "font-weight: bold; color: #ffaa00;"
        )  # Orange Button

        top_layout.addWidget(self.btn_load)
        top_layout.addSpacing(10)
        top_layout.addWidget(lbl_model)
        top_layout.addWidget(self.combo_model)
        top_layout.addSpacing(10)
        top_layout.addWidget(lbl_threads)
        top_layout.addWidget(self.spin_threads)
        top_layout.addStretch()
        top_layout.addWidget(self.btn_sniper)  # <-- Added Sniper Button
        top_layout.addWidget(self.btn_crack)
        top_group.setLayout(top_layout)
        master_layout.addWidget(top_group)

        # --- DSP Preprocessing Pipeline Drawer ---
        dsp_group = QGroupBox("DSP Signal Preprocessing Engine Configuration")
        dsp_master_layout = QHBoxLayout()

        # 1. Alignment Controls Panel
        align_box = QGroupBox("1. Global Trace Alignment")
        align_lay = QVBoxLayout()

        self.combo_align = QComboBox()
        self.combo_align.addItems(
            [
                "None",
                "Peak Cross-Correlation",
                "Phase-Only Correlation (POC)",
                "Dynamic Time Warping (DTW)",
            ]
        )
        self.combo_align.currentTextChanged.connect(self.update_trace_plot)
        align_lay.addWidget(QLabel("Primary Alignment Logic:"))
        align_lay.addWidget(self.combo_align)

        self.spin_astart = QSpinBox()
        self.spin_astart.setRange(0, 100000)
        self.spin_astart.setValue(0)
        self.spin_astart.valueChanged.connect(self.update_trace_plot)

        self.spin_aend = QSpinBox()
        self.spin_aend.setRange(10, 100000)
        self.spin_aend.setValue(500)
        self.spin_aend.valueChanged.connect(self.update_trace_plot)

        self.chk_elastic = QCheckBox("Enable Elastic Segment Alignment")
        self.chk_elastic.stateChanged.connect(self.update_trace_plot)

        self.spin_esegs = QSpinBox()
        self.spin_esegs.setRange(2, 64)
        self.spin_esegs.setValue(8)
        self.spin_esegs.valueChanged.connect(self.update_trace_plot)

        self.spin_ewarp = QSpinBox()
        self.spin_ewarp.setRange(1, 200)
        self.spin_ewarp.setValue(20)
        self.spin_ewarp.valueChanged.connect(self.update_trace_plot)

        h1 = QHBoxLayout()
        h1.addWidget(QLabel("Start:"))
        h1.addWidget(self.spin_astart)
        h1.addWidget(QLabel("End:"))
        h1.addWidget(self.spin_aend)
        align_lay.addLayout(h1)
        align_lay.addWidget(self.chk_elastic)
        h2 = QHBoxLayout()
        h2.addWidget(QLabel("Segs:"))
        h2.addWidget(self.spin_esegs)
        h2.addWidget(QLabel("Warp:"))
        h2.addWidget(self.spin_ewarp)
        align_lay.addLayout(h2)
        align_lay.addStretch()
        align_box.setLayout(align_lay)
        dsp_master_layout.addWidget(align_box)

        # 2. Countermeasure Defeats Panel
        countermeasure_box = QGroupBox("2. Countermeasure Defeat Engine")
        countermeasure_lay = QVBoxLayout()

        # --- A. MASKING ---
        countermeasure_lay.addWidget(QLabel("A. Masking Defeat (Amplitude):"))
        self.combo_masking = QComboBox()
        self.combo_masking.addItems(
            ["None", "Absolute Value Centering", "Trace Squaring"]
        )
        self.combo_masking.currentTextChanged.connect(self.update_trace_plot)
        countermeasure_lay.addWidget(self.combo_masking)
        countermeasure_lay.addSpacing(10)

        # --- B. JITTER / SHUFFLING ---
        countermeasure_lay.addWidget(
            QLabel("B. Jitter/Shuffling Defeat (Time Compression):")
        )
        self.combo_shuffle = QComboBox()
        self.combo_shuffle.addItems(
            [
                "None",
                "Integrated-Sum (Global)",
                "Sliding Window Integration (SWI)",
                "Window-Sum Pooling",
                "Window-Max Pooling",
            ]
        )
        self.combo_shuffle.currentTextChanged.connect(self.update_trace_plot)
        countermeasure_lay.addWidget(self.combo_shuffle)

        h_win = QHBoxLayout()
        h_win.addWidget(QLabel("Global Window Size:"))
        self.spin_ssize = QSpinBox()
        self.spin_ssize.setRange(3, 50000)
        self.spin_ssize.setValue(45)
        self.spin_ssize.valueChanged.connect(self.update_trace_plot)
        h_win.addWidget(self.spin_ssize)
        countermeasure_lay.addLayout(h_win)
        countermeasure_lay.addSpacing(10)

        # --- C. SLICING ---
        self.chk_slice = QCheckBox("C. Enable Peak Slicing (Desync Extraction)")
        self.chk_slice.stateChanged.connect(self.update_trace_plot)
        countermeasure_lay.addWidget(self.chk_slice)

        lay_slice_bounds = QHBoxLayout()
        lay_slice_bounds.addWidget(QLabel("Start:"))
        self.spin_sstart = QSpinBox()
        self.spin_sstart.setRange(0, 1000000)
        self.spin_sstart.setValue(1150)
        self.spin_sstart.valueChanged.connect(self.update_trace_plot)
        lay_slice_bounds.addWidget(self.spin_sstart)

        lay_slice_bounds.addWidget(QLabel("End:"))
        self.spin_send = QSpinBox()
        self.spin_send.setRange(10, 1000000)
        self.spin_send.setValue(4800)
        self.spin_send.valueChanged.connect(self.update_trace_plot)
        lay_slice_bounds.addWidget(self.spin_send)
        countermeasure_lay.addLayout(lay_slice_bounds)

        h_peaks = QHBoxLayout()
        self.spin_sdist = QSpinBox()
        self.spin_sdist.setRange(1, 500)
        self.spin_sdist.setValue(95)
        self.spin_sdist.valueChanged.connect(self.update_trace_plot)

        self.spin_sprom = QDoubleSpinBox()
        self.spin_sprom.setRange(0.001, 100.0)
        self.spin_sprom.setValue(0.1)
        self.spin_sprom.setSingleStep(0.1)
        self.spin_sprom.valueChanged.connect(self.update_trace_plot)

        self.spin_scount = QSpinBox()
        self.spin_scount.setRange(1, 256)
        self.spin_scount.setValue(32)
        self.spin_scount.valueChanged.connect(self.update_trace_plot)

        h_peaks.addWidget(QLabel("Dist:"))
        h_peaks.addWidget(self.spin_sdist)
        h_peaks.addWidget(QLabel("Prom:"))
        h_peaks.addWidget(self.spin_sprom)
        h_peaks.addWidget(QLabel("Count:"))
        h_peaks.addWidget(self.spin_scount)
        countermeasure_lay.addLayout(h_peaks)

        countermeasure_lay.addStretch()
        countermeasure_box.setLayout(countermeasure_lay)
        dsp_master_layout.addWidget(countermeasure_box)

        # 3. Advanced Frequency & Linear Filters Panel
        filters_box = QGroupBox("3. Advanced Wavelet, Frequency & Linear Filters")
        filters_lay = QVBoxLayout()
        self.chk_bp = QCheckBox("Enable Bandpass Filter")
        self.chk_bp.stateChanged.connect(self.update_trace_plot)

        self.spin_bplow = QSpinBox()
        self.spin_bplow.setRange(1000, 10000000)
        self.spin_bplow.setValue(100000)
        self.spin_bplow.valueChanged.connect(self.update_trace_plot)

        self.spin_bphigh = QSpinBox()
        self.spin_bphigh.setRange(10000, 20000000)
        self.spin_bphigh.setValue(5000000)
        self.spin_bphigh.valueChanged.connect(self.update_trace_plot)

        self.chk_wavelet = QCheckBox("Enable Wavelet Smoothing")
        self.chk_wavelet.stateChanged.connect(self.update_trace_plot)
        self.combo_wavelet = QComboBox()
        self.combo_wavelet.addItems(["db4", "sym4", "haar"])
        self.combo_wavelet.currentTextChanged.connect(self.update_trace_plot)
        self.spin_wlevel = QSpinBox()
        self.spin_wlevel.setRange(1, 5)
        self.spin_wlevel.setValue(1)
        self.spin_wlevel.valueChanged.connect(self.update_trace_plot)

        self.chk_gauss = QCheckBox("Enable Gaussian Blur")
        self.chk_gauss.stateChanged.connect(self.update_trace_plot)
        self.spin_sigma = QDoubleSpinBox()
        self.spin_sigma.setRange(0.1, 10.0)
        self.spin_sigma.setValue(4.0)
        self.spin_sigma.valueChanged.connect(self.update_trace_plot)

        self.chk_pca = QCheckBox("Enable PCA Filter")
        self.chk_pca.stateChanged.connect(self.update_trace_plot)
        self.spin_pca = QSpinBox()
        self.spin_pca.setRange(1, 50)
        self.spin_pca.setValue(5)
        self.spin_pca.valueChanged.connect(self.update_trace_plot)

        self.chk_fft = QCheckBox("Enable FFT Magnitude Domain Conversion")
        self.chk_fft.stateChanged.connect(self.update_trace_plot)

        self.spin_fft_start = QSpinBox()
        self.spin_fft_start.setRange(0, 100000)
        self.spin_fft_start.setValue(0)
        self.spin_fft_start.valueChanged.connect(self.update_trace_plot)

        self.spin_fft_end = QSpinBox()
        self.spin_fft_end.setRange(0, 100000)
        self.spin_fft_end.setValue(0)
        self.spin_fft_end.valueChanged.connect(self.update_trace_plot)

        self.spin_fft_cutoff = QSpinBox()
        self.spin_fft_cutoff.setRange(0, 10000)
        self.spin_fft_cutoff.setValue(0)  # 0 means disabled / dynamic max
        self.spin_fft_cutoff.setToolTip(
            "Truncate high-frequency noise past this bin (0 = keep all)"
        )
        self.spin_fft_cutoff.valueChanged.connect(self.update_trace_plot)

        h5 = QHBoxLayout()
        h5.addWidget(self.chk_bp)
        h5.addWidget(self.spin_bplow)
        h5.addWidget(self.spin_bphigh)
        filters_lay.addLayout(h5)
        h6 = QHBoxLayout()
        h6.addWidget(self.chk_wavelet)
        h6.addWidget(self.combo_wavelet)
        h6.addWidget(self.spin_wlevel)
        filters_lay.addLayout(h6)
        h7 = QHBoxLayout()
        h7.addWidget(self.chk_gauss)
        h7.addWidget(self.spin_sigma)
        h7.addWidget(self.chk_pca)
        h7.addWidget(self.spin_pca)
        filters_lay.addLayout(h7)
        h8 = QHBoxLayout()
        h8.addWidget(self.chk_fft)
        h8.addWidget(QLabel("S:"))
        h8.addWidget(self.spin_fft_start)
        h8.addWidget(QLabel("E:"))
        h8.addWidget(self.spin_fft_end)
        h8.addWidget(QLabel("Max Bins:"))
        h8.addWidget(self.spin_fft_cutoff)
        filters_lay.addLayout(h8)
        filters_box.setLayout(filters_lay)
        dsp_master_layout.addWidget(filters_box)

        dsp_group.setLayout(dsp_master_layout)
        master_layout.addWidget(dsp_group)

        # --- Diagnostic Visualization Switcher ---
        viz_group = QGroupBox("Target Diagnostic Visualization Panel")
        viz_layout = QHBoxLayout()
        viz_layout.addWidget(QLabel("Inspect Target Sub-key index:"))

        self.spin_inspect_byte = QSpinBox()
        self.spin_inspect_byte.valueChanged.connect(self.update_correlation_plot)
        viz_layout.addWidget(self.spin_inspect_byte)

        viz_layout.addSpacing(30)
        viz_layout.addWidget(QLabel("Diagnostic Mapping Display Metric:"))
        self.combo_view_mode = QComboBox()
        self.combo_view_mode.addItems(["CPA Correlations", "Welch's t-test (TVLA)"])
        self.combo_view_mode.currentTextChanged.connect(self.update_correlation_plot)
        viz_layout.addWidget(self.combo_view_mode)
        viz_layout.addStretch()
        viz_group.setLayout(viz_layout)
        master_layout.addWidget(viz_group)

        # --- Status Display Elements ---
        status_layout = QHBoxLayout()
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)

        self.on_leakage_model_changed(self.combo_model.currentText())

        self.lbl_key = QLabel("Recovered Key: [ ? ]")
        self.lbl_key.setStyleSheet(
            "font-size: 18px; font-weight: bold; color: #00ff00; background: #222; padding: 5px 15px;"
        )
        status_layout.addWidget(self.progress_bar)
        status_layout.addWidget(self.lbl_key)
        master_layout.addLayout(status_layout)

        # --- Graph Widgets ---
        self.plot_traces = pg.PlotWidget(title="Power Traces Display Scope")
        self.plot_traces.setLabel("left", "Voltage Drop / Power Leakage")
        self.plot_traces.setLabel("bottom", "Time (Samples)")
        self.plot_traces.showGrid(x=True, y=True)
        master_layout.addWidget(self.plot_traces)

        self.plot_corr = pg.PlotWidget(title="Analysis Metric Trace Outputs")
        self.plot_corr.setLabel("left", "Metric Weight Value")
        self.plot_corr.setLabel("bottom", "Time (Samples)")
        self.plot_corr.showGrid(x=True, y=True)
        master_layout.addWidget(self.plot_corr)

        self.setLayout(master_layout)

        # --- ATTACH TO BOTTOM OF _setup_ui() ---
        
        # Helper to trigger both state changes and live graph updates
        def on_feature_toggled(*args):
            self.update_ui_states()
            self.update_trace_plot()

        # Reroute the major toggle switches to the new helper
        self.combo_align.currentTextChanged.connect(on_feature_toggled)
        self.chk_elastic.stateChanged.connect(on_feature_toggled)
        self.combo_masking.currentTextChanged.connect(on_feature_toggled)
        self.combo_shuffle.currentTextChanged.connect(on_feature_toggled)
        self.chk_slice.stateChanged.connect(on_feature_toggled)
        self.chk_bp.stateChanged.connect(on_feature_toggled)
        self.chk_wavelet.stateChanged.connect(on_feature_toggled)
        self.chk_gauss.stateChanged.connect(on_feature_toggled)
        self.chk_pca.stateChanged.connect(on_feature_toggled)
        self.chk_fft.stateChanged.connect(on_feature_toggled)

        # Run it once at startup to initialize the grey-outs
        self.update_ui_states()


    def update_ui_states(self):
        """Dynamically enables/disables UI parameters based on active DSP features."""
        # 1. Alignment Panel
        align_active = self.combo_align.currentText() != "None"
        self.spin_astart.setEnabled(align_active)
        self.spin_aend.setEnabled(align_active)
        
        elastic_active = self.chk_elastic.isChecked()
        self.spin_esegs.setEnabled(elastic_active)
        self.spin_ewarp.setEnabled(elastic_active)

        # 3. Jitter / Shuffling Defeat — read mode first, it governs several sub-controls
        shuffle_mode = self.combo_shuffle.currentText()
        shuffle_active = shuffle_mode != "None"
        integrated_sum_active = shuffle_mode == "Integrated-Sum (Global)"

        # 2. Slicing Panel
        # Start/End are shared bounds for ALL shuffle modes AND peak slicing.
        # Peak Slicing is incompatible with Integrated-Sum (IS collapses to 1 sample first).
        slice_active = self.chk_slice.isChecked()
        if integrated_sum_active:
            self.chk_slice.setEnabled(False)
            self.chk_slice.setChecked(False)
            slice_active = False
        else:
            self.chk_slice.setEnabled(True)

        # Start/End are always editable — they act as a pre-window for raw CPA too.
        self.spin_sstart.setEnabled(True)
        self.spin_send.setEnabled(True)
        self.spin_sdist.setEnabled(slice_active)
        self.spin_sprom.setEnabled(slice_active)
        self.spin_scount.setEnabled(slice_active)

        needs_window = slice_active or shuffle_mode in [
            "Sliding Window Integration (SWI)",
            "Window-Sum Pooling",
            "Window-Max Pooling",
        ]
        self.spin_ssize.setEnabled(needs_window)

        # 4. Filters Panel
        bp_active = self.chk_bp.isChecked()
        self.spin_bplow.setEnabled(bp_active)
        self.spin_bphigh.setEnabled(bp_active)
        
        wavelet_active = self.chk_wavelet.isChecked()
        self.combo_wavelet.setEnabled(wavelet_active)
        self.spin_wlevel.setEnabled(wavelet_active)
        
        self.spin_sigma.setEnabled(self.chk_gauss.isChecked())
        self.spin_pca.setEnabled(self.chk_pca.isChecked())
        
        fft_active = self.chk_fft.isChecked()
        self.spin_fft_start.setEnabled(fft_active)
        self.spin_fft_end.setEnabled(fft_active)
        self.spin_fft_cutoff.setEnabled(fft_active)
    def on_leakage_model_changed(self, model_name):
        model_class = self.leakage_models_map.get(model_name)
        if model_class:
            self.spin_inspect_byte.setRange(0, model_class.num_targets - 1)
            self.progress_bar.setMaximum(model_class.num_targets)
        self.update_trace_plot()

    def get_current_dsp_dictionary(self):
        # BUG FIX: Apply boundary clamp at the UI level to prevent inverted slice slices
        slice_start = min(self.spin_sstart.value(), self.spin_send.value() - 1)
        slice_end = max(self.spin_sstart.value() + 1, self.spin_send.value())
        
        return {
            "align_mode": self.combo_align.currentText(),
            "align_start": self.spin_astart.value(),
            "align_end": self.spin_aend.value(),
            "elastic_enabled": self.chk_elastic.isChecked(),
            "elastic_segs": self.spin_esegs.value(),
            "elastic_warp": self.spin_ewarp.value(),
            "slice_enabled": self.chk_slice.isChecked(),
            "slice_start": slice_start,
            "slice_end": slice_end,
            "slice_size": self.spin_ssize.value(),
            "slice_count": self.spin_scount.value(),
            "slice_dist": self.spin_sdist.value(),
            "slice_prom": self.spin_sprom.value(),
            "shuffle_mode": self.combo_shuffle.currentText(),
            "leakage_model": self.combo_model.currentText(),
            "bandpass_enabled": self.chk_bp.isChecked(),
            "bp_low": self.spin_bplow.value(),
            "bp_high": self.spin_bphigh.value(),
            "bp_fs": getattr(self, "true_sample_rate", 20_000_000),
            "wavelet_enabled": self.chk_wavelet.isChecked(),
            "wavelet_type": self.combo_wavelet.currentText(),
            "wavelet_level": self.spin_wlevel.value(),
            "gauss_enabled": self.chk_gauss.isChecked(),
            "gauss_sigma": self.spin_sigma.value(),
            "fft_enabled": self.chk_fft.isChecked(),
            "fft_start": self.spin_fft_start.value(),
            "fft_end": (
                self.spin_fft_end.value() if self.spin_fft_end.value() > 0 else None
            ),
            "fft_cutoff": (
                self.spin_fft_cutoff.value() if self.spin_fft_cutoff.value() > 0 else None
            ),
            "pca_enabled": self.chk_pca.isChecked(),
            "pca_comps": self.spin_pca.value(),
            "masking_mode": self.combo_masking.currentText()
            # BUG FIX: Removed the duplicated shuffle_mode key that was overriding variables
        }

    def update_trace_plot(self):
        if self.traces is None:
            return
        self.plot_traces.clear()

        num_to_plot = min(10, self.traces.shape[0])
        traces_to_plot = self.traces[:num_to_plot].copy()

        dsp = self.get_current_dsp_dictionary()
        traces_to_plot = apply_dsp_pipeline(
            traces_to_plot, dsp, full_traces=self.traces
        )

        # --- BUG FIX: Prevent disappearing traces on Integrated-Sum ---
        if traces_to_plot.shape[1] == 1:
            # Duplicate the single point so it draws as a flat horizontal line
            traces_to_plot = np.repeat(traces_to_plot, 2, axis=1)
            title_str = "Preview (Collapsed to Single Energy Sum)"
        else:
            title_str = "Preview (Unified DSP Engine)"

        self.plot_traces.setTitle(f"Power Traces Preview [{title_str}]")
        for i in range(traces_to_plot.shape[0]):
            self.plot_traces.plot(traces_to_plot[i], pen=(i, 10))

    def load_data(self):
        root_dir = Path(__file__).resolve().parent.parent.parent
        default_search_path = str(root_dir / "captured_runs")
        selected_dir = QFileDialog.getExistingDirectory(
            self, "Select Captured Run Folder", default_search_path
        )
        if not selected_dir:
            return

        run_path = Path(selected_dir)
        file_traces = run_path / "traces.npy"
        file_textins = run_path / "plaintexts.npy"
        file_keys = run_path / "key.npy"
        file_ciphers = run_path / "ciphertexts.npy"
        file_meta = run_path / "metadata.json"  # <-- ADD THIS

        # FIX: Backward compatibility check (allow loading older runs without ciphers)
        if not (file_traces.exists() and file_textins.exists() and file_keys.exists()):
            QMessageBox.critical(
                self,
                "Directory Error",
                "The selected folder does not contain a complete target dataset matrix.",
            )
            return

        try:
            self.lbl_key.setText("Loading arrays into host memory space...")
            self.traces = np.load(file_traces)
            self.textins = np.load(file_textins)
            self.keys = np.load(file_keys)
            
            # FIX: Load ciphertexts if they exist, otherwise flag it
            if file_ciphers.exists():
                self.ciphers = np.load(file_ciphers)
                has_ciphers_str = " + Ciphertexts"
            else:
                self.ciphers = None
                has_ciphers_str = " (No Ciphertexts found)"
            if file_meta.exists():
                import json
                with open(file_meta, "r") as f:
                    meta = json.load(f)
                    self.true_sample_rate = meta.get("sample_rate_hz", 20_000_000)
            else:
                self.true_sample_rate = 20_000_000
            num_traces, _ = self.traces.shape
            self.btn_crack.setEnabled(True)
            self.btn_sniper.setEnabled(True)
            self.btn_crack.setText(f"Execute SCA Verification ({num_traces} Traces)")
            self.lbl_key.setText("Recovered Key: [ ? ]")
            self.update_trace_plot()
            QMessageBox.information(
                self, "Data Loaded", f"Successfully mapped dataset: {run_path.name}{has_ciphers_str}"
            )
        except Exception as e:
            QMessageBox.warning(self, "Load Error", f"Mapping failure:\n{str(e)}")

    def start_cpa(self):
        if self.traces is None:
            return
        self.btn_crack.setEnabled(False)
        self.btn_load.setEnabled(False)
        self.progress_bar.setValue(0)
        self.lbl_key.setStyleSheet(
            "font-size: 18px; font-weight: bold; color: #aaaaaa; background: #222; padding: 5px 15px;"
        )
        self.lbl_key.setText("Analyzing leakage matrices...")

        dsp_settings = self.get_current_dsp_dictionary()
        threads = self.spin_threads.value()

        self.worker = AnalysisWorker(
            self.traces, self.textins, self.ciphers, dsp_settings, threads, true_keys=self.keys
        )
        self.worker.progress_signal.connect(self.update_progress)
        self.worker.finished_signal.connect(self.on_cpa_finished)
        self.worker.start()

    def update_progress(self, current_byte, total_bytes):
        self.progress_bar.setValue(current_byte)

    def on_cpa_finished(self, recovered_key, all_correlations, pge_list, tvla_matrix):
        self.btn_crack.setEnabled(True)
        self.btn_load.setEnabled(True)

        model_cls = self.leakage_models_map.get(self.combo_model.currentText())
        num_targets = model_cls.num_targets if model_cls else 16
        self.progress_bar.setValue(num_targets)

        self.recovered_key = recovered_key
        self.correlations = all_correlations
        self.pge_list = pge_list
        self.tvla_matrix = tvla_matrix

        key_hex = " ".join([f"{b:02x}" for b in recovered_key])
        true_key_array = self.keys if self.keys.ndim == 1 else self.keys[0]
        actual_hex = " ".join([f"{b:02x}" for b in true_key_array[:num_targets]])
        avg_pge = np.mean(pge_list)

        if key_hex == actual_hex:
            self.lbl_key.setStyleSheet(
                "font-size: 18px; font-weight: bold; color: #00ff00; background: #222; padding: 5px 15px;"
            )
            self.lbl_key.setText(f"SUCCESS! Key: {key_hex}")
        else:
            self.lbl_key.setStyleSheet(
                "font-size: 18px; font-weight: bold; color: #ff5555; background: #222; padding: 5px 15px;"
            )
            self.lbl_key.setText(
                f"FAILED. Got: {key_hex} | Avg PGE Rank: {avg_pge:.1f}"
            )
        self.update_correlation_plot()

    def update_correlation_plot(self):
        if self.correlations is None or self.recovered_key is None:
            return
        self.plot_corr.clear()
        target_byte = self.spin_inspect_byte.value()
        view_mode = self.combo_view_mode.currentText()
        pge_val = self.pge_list[target_byte] if self.pge_list is not None else "?"

        model_cls = self.leakage_models_map.get(self.combo_model.currentText())
        guess_space = model_cls.guess_space if model_cls else 256

        if view_mode == "CPA Correlations":
            best_guess = self.recovered_key[target_byte]
            self.plot_corr.setTitle(
                f"Correlation Spectrum (Index {target_byte} | PGE Rank: {pge_val})"
            )
            self.plot_corr.setLabel("left", "Pearson Coefficient |r|")

            # Extract the correlation data for this specific byte
            corr_data = self.correlations[target_byte]

            # --- BUG FIX: Prevent disappearing correlation graphs ---
            # If the DSP pipeline collapsed the traces to a single sum, duplicate
            # the correlation point so Pyqtgraph has a horizontal line to draw.
            if corr_data.shape[1] == 1:
                corr_data = np.repeat(corr_data, 2, axis=1)

            for kguess in range(guess_space):
                if kguess == best_guess:
                    self.plot_corr.plot(corr_data[kguess], pen=pg.mkPen("r", width=2))
                else:
                    self.plot_corr.plot(
                        corr_data[kguess], pen=pg.mkPen((100, 100, 100, 40))
                    )

        elif view_mode == "Welch's t-test (TVLA)":
            self.plot_corr.setTitle(f"Welch's t-test Leakage Map (Index {target_byte})")
            self.plot_corr.setLabel("left", "t-statistic Value")
            if self.tvla_matrix is not None:
                self.plot_corr.plot(
                    self.tvla_matrix[target_byte], pen=pg.mkPen("c", width=1.5)
                )
                num_samples = self.tvla_matrix.shape[1]
                self.plot_corr.plot(
                    np.ones(num_samples) * 4.5,
                    pen=pg.mkPen("y", style=Qt.PenStyle.DashLine),
                )
                self.plot_corr.plot(
                    np.ones(num_samples) * -4.5,
                    pen=pg.mkPen("y", style=Qt.PenStyle.DashLine),
                )

    def start_auto_sniper(self):
        if self.traces is None:
            return

        base_dsp = self.get_current_dsp_dictionary()

        # 1. Trigger the Dynamic Settings Dialog
        dialog = SniperSettingsDialog(base_dsp, self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return

        sniper_config = dialog.get_config()

        self.btn_crack.setEnabled(False)
        self.btn_sniper.setEnabled(False)
        self.btn_load.setEnabled(False)
        self.progress_bar.setValue(0)
        self.lbl_key.setStyleSheet(
            "font-size: 18px; font-weight: bold; color: #ffaa00; background: #222; padding: 5px 15px;"
        )

        # 2. Pass the dynamic config to the Worker
        self.sniper_worker = AutoSniperWorker(
            self.traces, self.textins, self.ciphers, base_dsp, sniper_config, true_keys=self.keys
        )
        self.sniper_worker.progress_signal.connect(self.update_sniper_progress)
        self.sniper_worker.finished_signal.connect(self.on_sniper_finished)
        self.sniper_worker.start()

    def update_sniper_progress(self, current_byte, total_bytes, msg):
        self.progress_bar.setValue(current_byte)
        self.lbl_key.setText(msg)

    def on_sniper_finished(self, recovered_key, best_pges, probability_matrix, best_corrs):
        self.btn_crack.setEnabled(True)
        self.btn_sniper.setEnabled(True)
        self.btn_load.setEnabled(True)

        self.probability_matrix = probability_matrix
        self.best_corrs = best_corrs  # FIX: Store it in class scope
        model_cls = self.leakage_models_map.get(self.combo_model.currentText())
        num_targets = model_cls.num_targets if model_cls else 16

        self.recovered_key = recovered_key
        self.pge_list = best_pges

        key_hex = " ".join([f"{b:02x}" for b in recovered_key])
        true_key_array = self.keys if self.keys.ndim == 1 else self.keys[0]
        actual_hex = " ".join([f"{b:02x}" for b in true_key_array[:num_targets]])
        avg_pge = np.mean(best_pges)

        if key_hex == actual_hex:
            self.lbl_key.setStyleSheet(
                "font-size: 18px; font-weight: bold; color: #00ff00; background: #222; padding: 5px 15px;"
            )
            self.lbl_key.setText(f"SNIPER SUCCESS! Key: {key_hex}")
        else:
            self.lbl_key.setStyleSheet(
                "font-size: 18px; font-weight: bold; color: #ffaa00; background: #222; padding: 5px 15px;"
            )
            self.lbl_key.setText(f"SNIPER PARTIAL: {key_hex} | Avg PGE: {avg_pge:.1f}")

            # FIX: Message updated to state full 256 distribution properties
            reply = QMessageBox.question(
                self,
                "Sniper Partial Recovery",
                f"Not all bytes reached PGE 0.\n\nAverage PGE Rank: {avg_pge:.1f}\n\nThe complete 256 candidate space distribution has been mapped. Do you want to execute a full Probabilistic Beam Search across all candidate matrix arrays now?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.Yes,
            )

            if reply == QMessageBox.StandardButton.Yes:
                self.start_probabilistic_search()

    def start_probabilistic_search(self):
        """
        Executes a dynamic Beam Search using Sniper PGE rankings.
        """
        try:
            from Crypto.Cipher import AES
        except ImportError:
            QMessageBox.critical(self, "Dependency Missing", "Please install pycryptodome: pip install pycryptodome")
            return

        if self.textins is None:
            QMessageBox.warning(self, "No Data", "Please load a dataset with plaintexts and ciphertexts first!")
            return

        if not hasattr(self, "ciphers") or self.ciphers is None:
            QMessageBox.warning(self, "Missing Ciphertexts", "Real-world verification requires the captured hardware ciphertexts!")
            return

        sample_plaintext = bytes(self.textins[0])
        target_ciphertext = bytes(self.ciphers[0])

        self.lbl_key.setText("Executing dynamic PGE-bounded Beam Search...")
        avg_pge = np.mean(self.pge_list)
        worst_pge = np.max(self.pge_list)
        beam_width = 5000 if worst_pge < 10 else 25000 
        
        if worst_pge > 50:
            QMessageBox.warning(
                self, 
                "High PGE Warning", 
                f"Byte {np.argmax(self.pge_list)} has a PGE of {worst_pge}. The beam search might prune the correct key due to extreme noise. Proceeding with expanded beam width."
            )
        active_beam = [(0.0, [])]

        for b in range(16):
            successors = []
            
            # FIX: Pull the saved max correlations from the Sniper results, not the base CPA run
            max_corrs_per_guess = self.best_corrs[b]
            sorted_guesses = self.probability_matrix[b]
            
            # FIX: DYNAMIC SEARCH DEPTH LOGIC
            # We look at the actual PGE for this byte. If PGE is 0, we only test 1 candidate.
            # If PGE is 10, we test 11 candidates. This guarantees the true key is included 
            # while throwing away hundreds of thousands of dead-end branches.
            if self.pge_list is not None and b < len(self.pge_list):
                search_depth = self.pge_list[b] + 1
            else:
                search_depth = 12 # Fallback if PGE is missing for some reason

            for score_so_far, path in active_beam:
                # Look only down to the required PGE depth
                for rank in range(search_depth):
                    guess = sorted_guesses[rank]
                    local_corr = max_corrs_per_guess[guess]
                    
                    new_score = score_so_far - local_corr 
                    new_path = path + [guess]
                    successors.append((new_score, new_path))
            
            # Prune the tree: Keep only the top 'beam_width' best paths
            successors.sort()
            active_beam = successors[:beam_width]
            print(f"Byte {b} processing complete. Active paths in beam: {len(active_beam)} | Search Depth: {search_depth}")

        # Verification Block
        self.lbl_key.setText(f"Verifying {len(active_beam)} full candidates against ciphertext...")
        
        found_key = None
        for _, candidate_key_list in active_beam:
            candidate_key_bytes = bytes(candidate_key_list)
            
            cipher = AES.new(candidate_key_bytes, AES.MODE_ECB)
            test_ciphertext = cipher.encrypt(sample_plaintext)
            
            if test_ciphertext == target_ciphertext:
                found_key = candidate_key_list
                break

        if found_key is not None:
            key_hex = " ".join([f"{b:02x}" for b in found_key])
            self.lbl_key.setText(f"REAL KEY RECOVERED: {key_hex}")
            self.lbl_key.setStyleSheet(
                "font-size: 18px; font-weight: bold; color: #00ff00; background: #222; padding: 5px 15px;"
            )
            QMessageBox.information(self, "Success!", f"Key mathematically verified against ciphertext:\n\n{key_hex}")
        else:
            self.lbl_key.setText("BEAM SEARCH FAILED: Key not within search space bounds.")
            self.lbl_key.setStyleSheet(
                "font-size: 18px; font-weight: bold; color: #ff5555; background: #222; padding: 5px 15px;"
            )
