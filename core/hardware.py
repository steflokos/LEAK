import time
import numpy as np
import chipwhisperer as cw


class HardwareManager:
    """
    Handles all communication with the physical ChipWhisperer Nano.
    Responsible for connecting, flashing firmware, and gathering power traces.
    """

    def __init__(self):
        self.scope = None
        self.target = None
        self.is_connected = False

    def connect(self):
        """Connects to the ChipWhisperer Nano using SimpleSerial V1.1"""
        if self.is_connected:
            return True

        try:
            self.scope = cw.scope()
            self.target = cw.target(self.scope, cw.targets.SimpleSerial2)
            self.scope.default_setup()

            self.scope.adc.samples = 5000
            self.is_connected = True
            return True
        except Exception as e:
            self.is_connected = False
            raise RuntimeError(f"Failed to connect to ChipWhisperer: {str(e)}")

    def disconnect(self):
        """Safely closes the USB connection without throwing errors."""
        if self.is_connected:
            try:
                if self.target:
                    self.target.dis()
                if self.scope:
                    self.scope.dis()
            except Exception:
                pass
            finally:
                self.is_connected = False
                self.scope = None
                self.target = None

    def flash_firmware(self, hex_path: str):
        """Programs the STM32F030 on the Nano."""
        if not self.is_connected:
            raise ConnectionError("Must connect to hardware before flashing.")
        try:
            cw.program_target(self.scope, cw.programmers.STM32FProgrammer, hex_path)
            time.sleep(0.5)
            return True
        except Exception as e:
            raise RuntimeError(f"Flashing failed: {str(e)}")

    def capture_traces(self, num_traces: int, progress_callback=None):
        """
        Captures N power traces. Pre-allocates NumPy arrays for high speed
        and checks for cancellation requests from the UI.
        """
        if not self.is_connected:
            raise ConnectionError("Must connect to hardware before capturing.")

        # Pre-allocate contiguous arrays to eliminate incremental memory allocations
        num_samples = self.scope.adc.samples
        traces = np.empty((num_traces, num_samples), dtype=np.float32)
        textins = np.empty((num_traces, 16), dtype=np.uint8)
        keys = np.empty((num_traces, 16), dtype=np.uint8)

        ktp = cw.ktp.Basic()

        # --- INITIALIZATION ---
        self.scope.io.nrst = False
        time.sleep(0.05)
        self.scope.io.nrst = True
        time.sleep(0.05)
        self.target.flush()

        # Send Key once
        key, _ = ktp.next()
        self.target.simpleserial_write("k", key)
        time.sleep(0.05)
        # self.target.flush()

        # --- LIGHTNING CAPTURE LOOP ---
        idx = 0
        for i in range(num_traces):
            # Bidirectional UI Callback check: Returning False signals an early cancel
            if progress_callback and progress_callback(idx, num_traces) is False:
                break

            _, text = ktp.next()
            self.scope.arm()
            self.target.simpleserial_write("p", text)

            if self.scope.capture():
                self.target.flush()
                continue

            wave = self.scope.get_last_trace()
            self.target.simpleserial_read("r", 16, timeout=250)
            self.target.flush()

            # Assign directly into pre-allocated slots
            traces[idx] = wave
            textins[idx] = text
            keys[idx] = key
            idx += 1

        if idx == 0:
            raise RuntimeError(
                "Capture failed: No traces triggered. Re-flash the chip."
            )

        # Slice off any items left uncaptured if timeouts occurred or execution was cancelled
        return traces[:idx], textins[:idx], keys[:idx]
