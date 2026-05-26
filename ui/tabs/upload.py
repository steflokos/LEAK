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
    QMessageBox,
    QProgressBar,
)
from PyQt6.QtCore import Qt
from ui.workers import CompileWorker
from pathlib import Path


class UploadTab(QWidget):
    def __init__(self):
        super().__init__()
        self.worker = None
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout()
        layout.setSpacing(15)

        root_dir = Path(__file__).resolve().parent.parent.parent
        assets_dir = root_dir / "assets"

        # --- File Selection Group ---
        file_group = QGroupBox("Target Firmware Files")
        file_layout = QVBoxLayout()

        self.path_aes_c = self._create_file_selector(
            file_layout, "AES Source (.c):", "C Files (*.c)"
        )
        self.path_aes_h = self._create_file_selector(
            file_layout, "AES Header (.h):", "Header Files (*.h)"
        )
        self.path_main_c = self._create_file_selector(
            file_layout, "Main Wrapper (.c):", "C Files (*.c)"
        )

        if (assets_dir / "aes.c").exists():
            self.path_aes_c.setText(str(assets_dir / "aes.c"))
        if (assets_dir / "aes.h").exists():
            self.path_aes_h.setText(str(assets_dir / "aes.h"))
        if (assets_dir / "main.c").exists():
            self.path_main_c.setText(str(assets_dir / "main.c"))

        file_group.setLayout(file_layout)
        layout.addWidget(file_group)

        # --- Action Buttons Layout ---
        action_layout = QHBoxLayout()
        self.btn_compile = QPushButton("Compile Firmware")
        self.btn_compile.setMinimumHeight(40)
        self.btn_compile.setStyleSheet("font-weight: bold;")
        self.btn_compile.clicked.connect(self.start_compilation)

        self.btn_cancel = QPushButton("Cancel Setup")
        self.btn_cancel.setMinimumHeight(40)
        self.btn_cancel.setEnabled(False)
        self.btn_cancel.clicked.connect(self.cancel_compilation)

        action_layout.addWidget(self.btn_compile, stretch=3)
        action_layout.addWidget(self.btn_cancel, stretch=1)
        layout.addLayout(action_layout)

        # --- Toolchain Download Progress Bar ---
        self.download_progress = QProgressBar()
        self.download_progress.setVisible(False)
        self.download_progress.setTextVisible(True)
        self.download_progress.setFormat("Downloading Toolchain: %p% (%v/%m MB)")
        layout.addWidget(self.download_progress)

        # --- Log Output ---
        log_group = QGroupBox("Build Output")
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

    def _create_file_selector(self, parent_layout, label_text, filter_text):
        row_layout = QHBoxLayout()
        lbl = QLabel(label_text)
        lbl.setFixedWidth(120)

        line_edit = QLineEdit()
        line_edit.setReadOnly(True)
        line_edit.setPlaceholderText("No file selected...")

        btn_browse = QPushButton("Browse...")
        btn_browse.clicked.connect(lambda: self._browse_file(line_edit, filter_text))

        row_layout.addWidget(lbl)
        row_layout.addWidget(line_edit)
        row_layout.addWidget(btn_browse)

        parent_layout.addLayout(row_layout)
        return line_edit

    def _browse_file(self, line_edit, filter_text):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select File", "", filter_text)
        if file_path:
            line_edit.setText(file_path)

    def log_message(self, message):
        self.text_log.append(message)

    def start_compilation(self):
        aes_c = self.path_aes_c.text()
        aes_h = self.path_aes_h.text()
        main_c = self.path_main_c.text()

        if not all([aes_c, aes_h, main_c]):
            QMessageBox.warning(
                self, "Missing Files", "Please select all three files before compiling."
            )
            return

        self.btn_compile.setEnabled(False)
        self.btn_compile.setText("Compiling... Please Wait")
        self.btn_cancel.setEnabled(True)
        self.text_log.clear()

        self.worker = CompileWorker([aes_c, aes_h, main_c])
        self.worker.log_signal.connect(self.log_message)
        self.worker.download_progress_signal.connect(self.update_download_progress)
        self.worker.finished_signal.connect(self.compilation_finished)
        self.worker.start()

    def cancel_compilation(self):
        if self.worker and self.worker.isRunning():
            self.log_message("Requesting build cancellation...")
            self.worker.cancel()
            self.btn_cancel.setEnabled(False)

    def update_download_progress(self, current_bytes, total_bytes):
        if total_bytes <= 0:
            return
        if not self.download_progress.isVisible():
            self.download_progress.setVisible(True)

        # Convert bytes to megabytes for user formatting comfort
        mb_current = current_bytes / (1024 * 1024)
        mb_total = total_bytes / (1024 * 1024)

        self.download_progress.setMaximum(int(mb_total))
        self.download_progress.setValue(int(mb_current))

    def compilation_finished(self, success, output_path):
        self.btn_compile.setEnabled(True)
        self.btn_compile.setText("Compile Firmware")
        self.btn_cancel.setEnabled(False)
        self.download_progress.setVisible(False)

        if success:
            QMessageBox.information(
                self,
                "Success",
                f"Firmware compiled successfully!\nSaved to:\n{output_path}",
            )
        else:
            self.log_message("Compilation stopped or encountered problems.")
