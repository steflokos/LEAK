import sys
from PyQt6.QtWidgets import QApplication
from ui.main_window import MainWindow

def main():
    # Initialize the PyQt application
    app = QApplication(sys.argv)
    
    # Set a global style (optional, but makes it look professional)
    app.setStyle("Fusion")
    
    # Create and display the main window
    window = MainWindow()
    window.show()
    
    # Start the event loop
    sys.exit(app.exec())

if __name__ == "__main__":
    main()