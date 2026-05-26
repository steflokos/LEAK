import sys
import tarfile
import zipfile
import subprocess
from pathlib import Path
import requests


class CompilerManager:
    """
    Manages the auto-downloading of the ARM GCC toolchain and handles
    compilation of AES/Main files for the ChipWhisperer Nano.
    """

    TOOLCHAIN_URLS = {
        "win32": "https://github.com/xpack-dev-tools/arm-none-eabi-gcc-xpack/releases/download/v15.2.1-1.1/xpack-arm-none-eabi-gcc-15.2.1-1.1-win32-x64.zip",
        "linux": "https://github.com/xpack-dev-tools/arm-none-eabi-gcc-xpack/releases/download/v15.2.1-1.1/xpack-arm-none-eabi-gcc-15.2.1-1.1-linux-x64.tar.gz",
        "darwin": "https://github.com/xpack-dev-tools/arm-none-eabi-gcc-xpack/releases/download/v15.2.1-1.1/xpack-arm-none-eabi-gcc-15.2.1-1.1-darwin-arm64.tar.gz",
    }

    def __init__(self):
        self.root_dir = Path(__file__).resolve().parent.parent
        self.toolchain_dir = self.root_dir / ".toolchain"
        self.gcc_path = self._get_gcc_path()

    def _get_gcc_path(self):
        """Locates the gcc executable depending on the OS."""
        executable = (
            "arm-none-eabi-gcc.exe" if sys.platform == "win32" else "arm-none-eabi-gcc"
        )
        if self.toolchain_dir.exists():
            search_path = list(self.toolchain_dir.rglob(executable))
            if search_path:
                return search_path[0]
        return None

    def ensure_toolchain(self, status_callback=None, progress_callback=None):
        """
        Checks if the compiler exists; if not, downloads and extracts it.
        Communicates with UI via decoupled functional callbacks.
        """
        self.gcc_path = self._get_gcc_path()
        if self.gcc_path and self.gcc_path.exists():
            return True

        if status_callback:
            status_callback(
                "ARM GCC Toolchain not found. Initiating portable download..."
            )

        self.toolchain_dir.mkdir(exist_ok=True)
        platform = sys.platform
        if platform not in self.TOOLCHAIN_URLS:
            raise OSError(f"Unsupported OS for auto-download: {platform}")

        url = self.TOOLCHAIN_URLS[platform]
        archive_path = self.toolchain_dir / url.split("/")[-1]

        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()
            total_size = int(response.headers.get("content-length", 0))

            downloaded = 0
            with open(archive_path, "wb") as file:
                for data in response.iter_content(chunk_size=4096):
                    if (
                        progress_callback
                        and progress_callback(downloaded, total_size) is False
                    ):
                        file.close()
                        if archive_path.exists():
                            archive_path.unlink()
                        return False
                    size = file.write(data)
                    downloaded += size

            if status_callback:
                status_callback("Extracting archive toolchain... Please wait.")

            if archive_path.suffix == ".zip":
                with zipfile.ZipFile(archive_path, "r") as zip_ref:
                    zip_ref.extractall(self.toolchain_dir)
            elif archive_path.name.endswith(".tar.gz"):
                with tarfile.open(archive_path, "r:gz") as tar_ref:
                    tar_ref.extractall(self.toolchain_dir)
        finally:
            if archive_path.exists():
                archive_path.unlink()

        self.gcc_path = self._get_gcc_path()
        if not self.gcc_path:
            raise FileNotFoundError(
                "Extraction failed. arm-none-eabi-gcc executable not found."
            )

        if status_callback:
            status_callback("Toolchain successfully prepared.")
        return True

    def compile_firmware(
        self,
        target_main: Path,
        dependencies: list[Path],  # <-- Fixed: renamed to reflect it is a list of Path objects
        output_hex: Path,
        cw_firmware_dir: Path,
    ):
        """Compiles C firmware into a micro-target hex file natively."""
        if not self.gcc_path:
            raise FileNotFoundError(
                "Compiler not initialized. Run ensure_toolchain() first."
            )

        hal_dir = cw_firmware_dir / "hal"
        f0_dir = hal_dir / "stm32f0"
        nano_dir = hal_dir / "stm32f0_nano"
        ss_dir = cw_firmware_dir / "simpleserial"

        # Initialize base framework dependencies
        source_files = [
            str(target_main),
            str(ss_dir / "simpleserial.c"),
            str(hal_dir / "hal.c"),
            str(nano_dir / "stm32f0_hal_nano.c"),
            str(f0_dir / "stm32f0_hal_lowlevel.c"),
            str(f0_dir / "stm32f0_startup.S"),
        ]

        # --- FIX START: Dynamically append list elements instead of stringifying the list object ---
        for dep_path in dependencies:
            source_files.append(str(dep_path))
        # --- FIX END ---

        # Validate existence of all generated file path strings
        for f in source_files:
            if not Path(f).exists():
                raise FileNotFoundError(f"Missing required build dependency: {f}")

        includes = [
            f"-I{ss_dir}",
            f"-I{hal_dir}",
            f"-I{nano_dir}",
            f"-I{f0_dir}",
            f"-I{f0_dir / 'CMSIS'}",
            f"-I{f0_dir / 'CMSIS' / 'core'}",
            f"-I{f0_dir / 'CMSIS' / 'device'}",
            f"-I{f0_dir / 'Legacy'}",
            f"-I{target_main.parent}",
        ]

        defines = [
            "-DSTM32F030x6",
            "-DSTM32F0",
            "-DSTM32",
            "-DDEBUG",
            "-DPLATFORM=CWNANO",
            "-DHAL_TYPE=HAL_stm32f0_nano",
            "-DSS_VER=SS_VER_2_1",     
            "-DF_CPU=7500000",         
            "-DHSE_VALUE=7500000",     
        ]

        cflags = [
            str(self.gcc_path),
            "-mcpu=cortex-m0",
            "-mthumb",
            "-mfloat-abi=soft",
            "-O3",
            "-ffunction-sections",
            "-fdata-sections",
            "-Wno-implicit-function-declaration",
            *includes,
            *defines,
            *source_files,
            "--specs=nano.specs",
            "--specs=nosys.specs",
            "-T",
            str(nano_dir / "LinkerScript.ld"),
            "-Wl,--gc-sections",
            "-lm",
            "-o",
            str(output_hex.with_suffix(".elf")),
        ]

        try:
            subprocess.run(
                cflags,
                check=True,
                capture_output=True,
                text=True,
                cwd=target_main.parent,
            )
            objcopy_path = self.gcc_path.parent / (
                "arm-none-eabi-objcopy.exe"
                if sys.platform == "win32"
                else "arm-none-eabi-objcopy"
            )

            hex_flags = [
                str(objcopy_path),
                "-O",
                "ihex",
                str(output_hex.with_suffix(".elf")),
                str(output_hex),
            ]
            subprocess.run(
                hex_flags,
                check=True,
                capture_output=True,
                text=True,
                cwd=target_main.parent,
            )
            return True
        except subprocess.CalledProcessError as e:
            error_msg = e.stderr if e.stderr else e.stdout
            raise RuntimeError(f"GCC Compilation Failed:\n{error_msg}")
