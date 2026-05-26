import os
import numpy as np
from pathlib import Path
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, 
    QLabel, QGroupBox, QProgressBar, QMessageBox,
    QCheckBox, QDoubleSpinBox, QSpinBox, QFileDialog, QComboBox
)
from PyQt6.QtCore import Qt
import pyqtgraph as pg
from ui.workers import AnalysisWorker, AutoSniperWorker
from ui.workers import AnalysisWorker
from core.analysis import (
    apply_dsp_pipeline, apply_gaussian_filter, apply_bandpass_filter, apply_max_pooling, apply_sum_pooling, apply_peak_alignment,
    apply_segment_alignment, apply_peak_slicing, apply_wavelet_denoising, apply_poc_alignment, apply_dtw_alignment,
    apply_pca_filtering, apply_fft_magnitude, LEAKAGE_MODELS
)

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
        self.btn_sniper.setStyleSheet("font-weight: bold; color: #ffaa00;") # Orange Button

        # Add this to your existing _setup_ui block
        self.btn_beam_search = QPushButton("Execute Beam Search")
        self.btn_beam_search.clicked.connect(self.start_probabilistic_search)
        self.btn_beam_search.setEnabled(False) # Disable until Auto-Sniper runs
        self.btn_beam_search.setMinimumHeight(35)
        self.btn_beam_search.setStyleSheet("font-weight: bold; color: #00ffff;") # Cyan button
        
        top_layout.addWidget(self.btn_beam_search)
        
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
        self.combo_align.addItems(["None", "Peak Cross-Correlation", "Phase-Only Correlation (POC)", "Dynamic Time Warping (DTW)"])
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
        h1.addWidget(QLabel("Start:")); h1.addWidget(self.spin_astart)
        h1.addWidget(QLabel("End:")); h1.addWidget(self.spin_aend)
        align_lay.addLayout(h1)
        align_lay.addWidget(self.chk_elastic)
        h2 = QHBoxLayout()
        h2.addWidget(QLabel("Segs:")); h2.addWidget(self.spin_esegs)
        h2.addWidget(QLabel("Warp:")); h2.addWidget(self.spin_ewarp)
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
        self.combo_masking.addItems(["None", "Absolute Value Centering", "Trace Squaring"])
        self.combo_masking.currentTextChanged.connect(self.update_trace_plot)
        countermeasure_lay.addWidget(self.combo_masking)
        countermeasure_lay.addSpacing(10)

        # --- B. JITTER / SHUFFLING ---
        countermeasure_lay.addWidget(QLabel("B. Jitter/Shuffling Defeat (Time Compression):"))
        self.combo_shuffle = QComboBox()
        self.combo_shuffle.addItems(["None", "Integrated-Sum (Global)", "Sliding Window Integration (SWI)", "Window-Sum Pooling", "Window-Max Pooling"])
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
        
        h_peaks.addWidget(QLabel("Dist:")); h_peaks.addWidget(self.spin_sdist)
        h_peaks.addWidget(QLabel("Prom:")); h_peaks.addWidget(self.spin_sprom)
        h_peaks.addWidget(QLabel("Count:")); h_peaks.addWidget(self.spin_scount)
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
        self.combo_wavelet.addItems(['db4', 'sym4', 'haar'])
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

        h5 = QHBoxLayout(); h5.addWidget(self.chk_bp); h5.addWidget(self.spin_bplow); h5.addWidget(self.spin_bphigh)
        filters_lay.addLayout(h5)
        h6 = QHBoxLayout(); h6.addWidget(self.chk_wavelet); h6.addWidget(self.combo_wavelet); h6.addWidget(self.spin_wlevel)
        filters_lay.addLayout(h6)
        h7 = QHBoxLayout(); h7.addWidget(self.chk_gauss); h7.addWidget(self.spin_sigma); h7.addWidget(self.chk_pca); h7.addWidget(self.spin_pca)
        filters_lay.addLayout(h7)
        h8 = QHBoxLayout(); h8.addWidget(self.chk_fft); h8.addWidget(QLabel("S:")); h8.addWidget(self.spin_fft_start); h8.addWidget(QLabel("E:")); h8.addWidget(self.spin_fft_end)
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
        self.lbl_key.setStyleSheet("font-size: 18px; font-weight: bold; color: #00ff00; background: #222; padding: 5px 15px;")
        status_layout.addWidget(self.progress_bar)
        status_layout.addWidget(self.lbl_key)
        master_layout.addLayout(status_layout)

        # --- Graph Widgets ---
        self.plot_traces = pg.PlotWidget(title="Power Traces Display Scope")
        self.plot_traces.setLabel('left', 'Voltage Drop / Power Leakage')
        self.plot_traces.setLabel('bottom', 'Time (Samples)')
        self.plot_traces.showGrid(x=True, y=True)
        master_layout.addWidget(self.plot_traces)

        self.plot_corr = pg.PlotWidget(title="Analysis Metric Trace Outputs")
        self.plot_corr.setLabel('left', 'Metric Weight Value')
        self.plot_corr.setLabel('bottom', 'Time (Samples)')
        self.plot_corr.showGrid(x=True, y=True)
        master_layout.addWidget(self.plot_corr)

        self.setLayout(master_layout)

    def on_leakage_model_changed(self, model_name):
        model_class = self.leakage_models_map.get(model_name)
        if model_class:
            self.spin_inspect_byte.setRange(0, model_class.num_targets - 1)
            self.progress_bar.setMaximum(model_class.num_targets)
        self.update_trace_plot()

    def get_current_dsp_dictionary(self):
        return {
            'align_mode': self.combo_align.currentText(),
            'align_start': self.spin_astart.value(),
            'align_end': self.spin_aend.value(),
            'elastic_enabled': self.chk_elastic.isChecked(),
            'elastic_segs': self.spin_esegs.value(),
            'elastic_warp': self.spin_ewarp.value(),
            'slice_enabled': self.chk_slice.isChecked(),
            'slice_start': self.spin_sstart.value(),
            'slice_end': self.spin_send.value(),
            'slice_size': self.spin_ssize.value(),
            'slice_count': self.spin_scount.value(),
            'slice_dist': self.spin_sdist.value(),          
            'slice_prom': self.spin_sprom.value(),          
            'shuffle_mode': self.combo_shuffle.currentText(),      
            'leakage_model': self.combo_model.currentText(),       
            'bandpass_enabled': self.chk_bp.isChecked(),
            'bp_low': self.spin_bplow.value(),
            'bp_high': self.spin_bphigh.value(),
            'bp_fs': 20_000_000, 
            'wavelet_enabled': self.chk_wavelet.isChecked(),
            'wavelet_type': self.combo_wavelet.currentText(),
            'wavelet_level': self.spin_wlevel.value(),
            'gauss_enabled': self.chk_gauss.isChecked(),
            'gauss_sigma': self.spin_sigma.value(),
            'fft_enabled': self.chk_fft.isChecked(),
            'fft_start': self.spin_fft_start.value(),
            'fft_end': self.spin_fft_end.value() if self.spin_fft_end.value() > 0 else None,
            'pca_enabled': self.chk_pca.isChecked(),
            'pca_comps': self.spin_pca.value(),
            'masking_mode': self.combo_masking.currentText(),
            'shuffle_mode': self.combo_shuffle.currentText(),
        }

    def update_trace_plot(self):
        if self.traces is None: return
        self.plot_traces.clear()
        
        num_to_plot = min(10, self.traces.shape[0])
        traces_to_plot = self.traces[:num_to_plot].copy()
        
        dsp = self.get_current_dsp_dictionary()
        traces_to_plot = apply_dsp_pipeline(traces_to_plot, dsp, full_traces=self.traces)

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
        selected_dir = QFileDialog.getExistingDirectory(self, "Select Captured Run Folder", default_search_path)
        if not selected_dir: return

        run_path = Path(selected_dir)
        file_traces, file_textins, file_keys = run_path / "traces.npy", run_path / "textins.npy", run_path / "keys.npy"

        if not (file_traces.exists() and file_textins.exists() and file_keys.exists()):
            QMessageBox.critical(self, "Directory Error", "The selected folder does not contain a complete target dataset matrix.")
            return

        try:
            self.lbl_key.setText("Loading arrays into host memory space...")
            self.traces = np.load(file_traces)
            self.textins = np.load(file_textins)
            self.keys = np.load(file_keys)
            
            num_traces, _ = self.traces.shape
            self.btn_crack.setEnabled(True)
            self.btn_sniper.setEnabled(True)
            self.btn_crack.setText(f"Execute SCA Verification ({num_traces} Traces)")
            self.lbl_key.setText("Recovered Key: [ ? ]")
            self.update_trace_plot()
            QMessageBox.information(self, "Data Loaded", f"Successfully mapped dataset: {run_path.name}")
        except Exception as e:
            QMessageBox.warning(self, "Load Error", f"Mapping failure:\n{str(e)}")

    def start_cpa(self):
        if self.traces is None: return
        self.btn_crack.setEnabled(False); self.btn_load.setEnabled(False)
        self.progress_bar.setValue(0)
        self.lbl_key.setStyleSheet("font-size: 18px; font-weight: bold; color: #aaaaaa; background: #222; padding: 5px 15px;")
        self.lbl_key.setText("Analyzing leakage matrices...")

        dsp_settings = self.get_current_dsp_dictionary()
        threads = self.spin_threads.value()

        self.worker = AnalysisWorker(self.traces, self.textins, dsp_settings, threads, true_keys=self.keys)
        self.worker.progress_signal.connect(self.update_progress)
        self.worker.finished_signal.connect(self.on_cpa_finished)
        self.worker.start()

    def update_progress(self, current_byte, total_bytes):
        self.progress_bar.setValue(current_byte)

    def on_cpa_finished(self, recovered_key, all_correlations, pge_list, tvla_matrix):
        self.btn_crack.setEnabled(True); self.btn_load.setEnabled(True)
        
        model_cls = self.leakage_models_map.get(self.combo_model.currentText())
        num_targets = model_cls.num_targets if model_cls else 16
        self.progress_bar.setValue(num_targets)
        
        self.recovered_key = recovered_key
        self.correlations = all_correlations
        self.pge_list = pge_list
        self.tvla_matrix = tvla_matrix
        
        key_hex = " ".join([f"{b:02x}" for b in recovered_key])
        actual_hex = " ".join([f"{b:02x}" for b in self.keys[0][:num_targets]])
        avg_pge = np.mean(pge_list)
        
        if key_hex == actual_hex:
            self.lbl_key.setStyleSheet("font-size: 18px; font-weight: bold; color: #00ff00; background: #222; padding: 5px 15px;")
            self.lbl_key.setText(f"SUCCESS! Key: {key_hex}")
        else:
            self.lbl_key.setStyleSheet("font-size: 18px; font-weight: bold; color: #ff5555; background: #222; padding: 5px 15px;")
            self.lbl_key.setText(f"FAILED. Got: {key_hex} | Avg PGE Rank: {avg_pge:.1f}")
        self.update_correlation_plot()

    def update_correlation_plot(self):
        if self.correlations is None or self.recovered_key is None: return
        self.plot_corr.clear()
        target_byte = self.spin_inspect_byte.value()
        view_mode = self.combo_view_mode.currentText()
        pge_val = self.pge_list[target_byte] if self.pge_list is not None else "?"
        
        model_cls = self.leakage_models_map.get(self.combo_model.currentText())
        guess_space = model_cls.guess_space if model_cls else 256
        
        if view_mode == "CPA Correlations":
            best_guess = self.recovered_key[target_byte]
            self.plot_corr.setTitle(f"Correlation Spectrum (Index {target_byte} | PGE Rank: {pge_val})")
            self.plot_corr.setLabel('left', 'Pearson Coefficient |r|')
            
            # Extract the correlation data for this specific byte
            corr_data = self.correlations[target_byte]
            
            # --- BUG FIX: Prevent disappearing correlation graphs ---
            # If the DSP pipeline collapsed the traces to a single sum, duplicate 
            # the correlation point so Pyqtgraph has a horizontal line to draw.
            if corr_data.shape[1] == 1:
                corr_data = np.repeat(corr_data, 2, axis=1)

            for kguess in range(guess_space):
                if kguess == best_guess:
                    self.plot_corr.plot(corr_data[kguess], pen=pg.mkPen('r', width=2))
                else:
                    self.plot_corr.plot(corr_data[kguess], pen=pg.mkPen((100, 100, 100, 40)))
                    
        elif view_mode == "Welch's t-test (TVLA)":
            self.plot_corr.setTitle(f"Welch's t-test Leakage Map (Index {target_byte})")
            self.plot_corr.setLabel('left', 't-statistic Value')
            if self.tvla_matrix is not None:
                self.plot_corr.plot(self.tvla_matrix[target_byte], pen=pg.mkPen('c', width=1.5))
                num_samples = self.tvla_matrix.shape[1]
                self.plot_corr.plot(np.ones(num_samples) * 4.5, pen=pg.mkPen('y', style=Qt.PenStyle.DashLine))
                self.plot_corr.plot(np.ones(num_samples) * -4.5, pen=pg.mkPen('y', style=Qt.PenStyle.DashLine))

    def start_auto_sniper(self):
        if self.traces is None: return
        self.btn_crack.setEnabled(False)
        self.btn_sniper.setEnabled(False)
        self.btn_load.setEnabled(False)
        self.progress_bar.setValue(0)
        self.lbl_key.setStyleSheet("font-size: 18px; font-weight: bold; color: #ffaa00; background: #222; padding: 5px 15px;")

        # Grab baseline settings from UI to start the grid search
        base_dsp = self.get_current_dsp_dictionary()
        
        self.sniper_worker = AutoSniperWorker(self.traces, self.textins, base_dsp, true_keys=self.keys)
        self.sniper_worker.progress_signal.connect(self.update_sniper_progress)
        self.sniper_worker.finished_signal.connect(self.on_sniper_finished)
        self.sniper_worker.start()

    def update_sniper_progress(self, current_byte, total_bytes, msg):
        self.progress_bar.setValue(current_byte)
        self.lbl_key.setText(msg)

    def on_sniper_finished(self, recovered_key, best_pges):
        self.btn_crack.setEnabled(True)
        self.btn_sniper.setEnabled(True)
        self.btn_load.setEnabled(True)
        # self.probability_matrix = probability_matrix
        # self.btn_beam_search.setEnabled(True)
        model_cls = self.leakage_models_map.get(self.combo_model.currentText())
        num_targets = model_cls.num_targets if model_cls else 16
        
        # We don't have full matrix data for this mode, so just store the keys
        self.recovered_key = recovered_key
        self.pge_list = best_pges
        
        key_hex = " ".join([f"{b:02x}" for b in recovered_key])
        actual_hex = " ".join([f"{b:02x}" for b in self.keys[0][:num_targets]])
        avg_pge = np.mean(best_pges)

        if key_hex == actual_hex:
            self.lbl_key.setStyleSheet("font-size: 18px; font-weight: bold; color: #00ff00; background: #222; padding: 5px 15px;")
            self.lbl_key.setText(f"SNIPER SUCCESS! Key: {key_hex}")
        else:
            self.lbl_key.setStyleSheet("font-size: 18px; font-weight: bold; color: #ffaa00; background: #222; padding: 5px 15px;")
            self.lbl_key.setText(f"SNIPER PARTIAL: {key_hex} | Avg PGE: {avg_pge:.1f}")


    def start_probabilistic_search(self):
        """
        Executes a Beam Search through the probability matrix.
        For each byte, it tests the Top 12 candidates against the true key.
        """
        if not hasattr(self, 'probability_matrix') or self.probability_matrix is None:
            QMessageBox.warning(self, "No Matrix", "Run Auto-Sniper Grid Search first!")
            return

        # 1. Access the Ground Truth (for validation)
        true_key = self.keys[0] # Assuming first trace key is ground truth
        final_key = [0] * 16
        
        # 2. Iterate bytes 0-15
        for b in range(16):
            candidates = self.probability_matrix[b] # The top 12 guesses
            
            # 3. Beam Search: Test candidates in order of probability
            found = False
            for rank, guess in enumerate(candidates):
                if guess == true_key[b]:
                    final_key[b] = guess
                    print(f"Byte {b} locked at Rank {rank}: 0x{guess:02x}")
                    found = True
                    break
            
            if not found:
                print(f"Byte {b} FAILED in Top 12 search.")
                final_key[b] = candidates[0] # Default to top guess
        
        # 4. Display result
        key_hex = " ".join([f"{b:02x}" for b in final_key])
        self.lbl_key.setText(f"BEAM SEARCH KEY: {key_hex}")
        self.lbl_key.setStyleSheet("font-size: 18px; font-weight: bold; color: #00ffff; background: #222; padding: 5px 15px;")