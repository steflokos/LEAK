from PyQt6.QtWidgets import QMainWindow, QTabWidget
from ui.tabs.upload import UploadTab
from ui.tabs.capture import CaptureTab
from ui.tabs.crack import CrackTab


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("LEAK - Leakage Evaluation Analysis Kit")
        self.setMinimumSize(950, 650)

        self.tabs = QTabWidget()
        # Optional: Clean layout document alignment
        self.tabs.setDocumentMode(True)
        self.tabs.setMovable(False)
        self.setCentralWidget(self.tabs)

        self.upload_tab = UploadTab()
        self.capture_tab = CaptureTab()
        self.crack_tab = CrackTab()

        self.tabs.addTab(self.upload_tab, "Upload && Compile")
        self.tabs.addTab(self.capture_tab, "Capture Traces")
        self.tabs.addTab(self.crack_tab, "Analysis && Cracking")
