import sys
from pathlib import Path
from PyQt6.QtWidgets import QApplication
from ui.main_window import MainWindow


_ICONS = {
    "cb_unchecked": (
        b'<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16">'
        b'<rect x="1.5" y="1.5" width="13" height="13" rx="2.5"'
        b' fill="#232323" stroke="#707070" stroke-width="1.5"/></svg>'
    ),
    "cb_unchecked_hover": (
        b'<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16">'
        b'<rect x="1.5" y="1.5" width="13" height="13" rx="2.5"'
        b' fill="#2e2e2e" stroke="#a0a0a0" stroke-width="1.5"/></svg>'
    ),
    "cb_unchecked_disabled": (
        b'<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16">'
        b'<rect x="1.5" y="1.5" width="13" height="13" rx="2.5"'
        b' fill="#1a1a1a" stroke="#3a3a3a" stroke-width="1.5"/></svg>'
    ),
    "cb_checked": (
        b'<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16">'
        b'<rect width="16" height="16" rx="3" fill="#0078d4"/>'
        b'<path d="M3.5 8.5l3 3 6-6.5" stroke="white" stroke-width="2"'
        b' fill="none" stroke-linecap="round" stroke-linejoin="round"/></svg>'
    ),
    "cb_checked_hover": (
        b'<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16">'
        b'<rect width="16" height="16" rx="3" fill="#1484d8"/>'
        b'<path d="M3.5 8.5l3 3 6-6.5" stroke="white" stroke-width="2"'
        b' fill="none" stroke-linecap="round" stroke-linejoin="round"/></svg>'
    ),
    "cb_checked_disabled": (
        b'<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16">'
        b'<rect width="16" height="16" rx="3" fill="#4a4a4a"/>'
        b'<path d="M3.5 8.5l3 3 6-6.5" stroke="#888" stroke-width="2"'
        b' fill="none" stroke-linecap="round" stroke-linejoin="round"/></svg>'
    ),
}


def _checkbox_stylesheet() -> str:
    icons_dir = Path(__file__).parent / "assets" / "icons"
    icons_dir.mkdir(parents=True, exist_ok=True)

    for name, svg in _ICONS.items():
        (icons_dir / f"{name}.svg").write_bytes(svg)

    def u(name: str) -> str:
        path = (icons_dir / f"{name}.svg").as_posix()
        return f'url("{path}")'

    return f"""
        QCheckBox::indicator {{
            width: 16px;
            height: 16px;
        }}
        QCheckBox::indicator:unchecked           {{ image: {u("cb_unchecked")}; }}
        QCheckBox::indicator:unchecked:hover     {{ image: {u("cb_unchecked_hover")}; }}
        QCheckBox::indicator:unchecked:disabled  {{ image: {u("cb_unchecked_disabled")}; }}
        QCheckBox::indicator:checked             {{ image: {u("cb_checked")}; }}
        QCheckBox::indicator:checked:hover       {{ image: {u("cb_checked_hover")}; }}
        QCheckBox::indicator:checked:disabled    {{ image: {u("cb_checked_disabled")}; }}
    """


def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setStyleSheet(_checkbox_stylesheet())

    window = MainWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
