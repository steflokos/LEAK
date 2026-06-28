import numpy as np
from scipy.ndimage import gaussian_filter1d, uniform_filter1d
from scipy.signal import butter, sosfilt, find_peaks, savgol_filter
from functools import lru_cache
from scipy.signal import butter, sosfilt, sosfiltfilt, find_peaks, savgol_filter


@lru_cache(maxsize=8)
def get_hw_reference(space_size=256):
    """Generates a cached Hamming Weight reference array for any arbitrary byte/nibble width space."""
    return np.array([bin(x).count("1") for x in range(space_size)], dtype=np.uint8)


# --- LEAKAGE MODEL REGISTRY FOR MULTI-CIPHER GENERALIZATION ---
class LeakageModel:
    name = "Generic Base Target"
    guess_space = 256
    num_targets = 16

    @staticmethod
    def get_intermediate(pt_bytes, kguesses):
        raise NotImplementedError("Leakage models must override get_intermediate().")


class AES128_SBox_Out(LeakageModel):
    name = "AES-128 S-Box Output (Round 1)"
    guess_space = 256
    num_targets = 16

    sbox = np.array(
        [
            0x63,
            0x7C,
            0x77,
            0x7B,
            0xF2,
            0x6B,
            0x6F,
            0xC5,
            0x30,
            0x01,
            0x67,
            0x2B,
            0xFE,
            0xD7,
            0xAB,
            0x76,
            0xCA,
            0x82,
            0xC9,
            0x7D,
            0xFA,
            0x59,
            0x47,
            0xF0,
            0xAD,
            0xD4,
            0xA2,
            0xAF,
            0x9C,
            0xA4,
            0x72,
            0xC0,
            0xB7,
            0xFD,
            0x93,
            0x26,
            0x36,
            0x3F,
            0xF7,
            0xCC,
            0x34,
            0xA5,
            0xE5,
            0xF1,
            0x71,
            0xD8,
            0x31,
            0x15,
            0x04,
            0xC7,
            0x23,
            0xC3,
            0x18,
            0x96,
            0x05,
            0x9A,
            0x07,
            0x12,
            0x80,
            0xE2,
            0xEB,
            0x27,
            0xB2,
            0x75,
            0x09,
            0x83,
            0x2C,
            0x1A,
            0x1B,
            0x6E,
            0x5A,
            0xA0,
            0x52,
            0x3B,
            0xD6,
            0xB3,
            0x29,
            0xE3,
            0x2F,
            0x84,
            0x53,
            0xD1,
            0x00,
            0xED,
            0x20,
            0xFC,
            0xB1,
            0x5B,
            0x6A,
            0xCB,
            0xBE,
            0x39,
            0x4A,
            0x4C,
            0x58,
            0xCF,
            0xD0,
            0xEF,
            0xAA,
            0xFB,
            0x43,
            0x4D,
            0x33,
            0x85,
            0x45,
            0xF9,
            0x02,
            0x7F,
            0x50,
            0x3C,
            0x9F,
            0xA8,
            0x51,
            0xA3,
            0x40,
            0x8F,
            0x92,
            0x9D,
            0x38,
            0xF5,
            0xBC,
            0xB6,
            0xDA,
            0x21,
            0x10,
            0xFF,
            0xF3,
            0xD2,
            0xCD,
            0x0C,
            0x13,
            0xEC,
            0x5F,
            0x97,
            0x44,
            0x17,
            0xC4,
            0xA7,
            0x7E,
            0x3D,
            0x64,
            0x5D,
            0x19,
            0x73,
            0x60,
            0x81,
            0x4F,
            0xDC,
            0x22,
            0x2A,
            0x90,
            0x88,
            0x46,
            0xEE,
            0xB8,
            0x14,
            0xDE,
            0x5E,
            0x0B,
            0xDB,
            0xE0,
            0x32,
            0x3A,
            0x0A,
            0x49,
            0x06,
            0x24,
            0x5C,
            0xC2,
            0xD3,
            0xAC,
            0x62,
            0x91,
            0x95,
            0xE4,
            0x79,
            0xE7,
            0xC8,
            0x37,
            0x6D,
            0x8D,
            0xD5,
            0x4E,
            0xA9,
            0x6C,
            0x56,
            0xF4,
            0xEA,
            0x65,
            0x7A,
            0xAE,
            0x08,
            0xBA,
            0x78,
            0x25,
            0x2E,
            0x1C,
            0xA6,
            0xB4,
            0xC6,
            0xE8,
            0xDD,
            0x74,
            0x1F,
            0x4B,
            0xBD,
            0x8B,
            0x8A,
            0x70,
            0x3E,
            0xB5,
            0x66,
            0x48,
            0x03,
            0xF6,
            0x0E,
            0x61,
            0x35,
            0x57,
            0xB9,
            0x86,
            0xC1,
            0x1D,
            0x9E,
            0xE1,
            0xF8,
            0x98,
            0x11,
            0x69,
            0xD9,
            0x8E,
            0x94,
            0x9B,
            0x1E,
            0x87,
            0xE9,
            0xCE,
            0x55,
            0x28,
            0xDF,
            0x8C,
            0xA1,
            0x89,
            0x0D,
            0xBF,
            0xE6,
            0x42,
            0x68,
            0x41,
            0x99,
            0x2D,
            0x0F,
            0xB0,
            0x54,
            0xBB,
            0x16,
        ],
        dtype=np.uint8,
    )

    @staticmethod
    def get_intermediate(pt_bytes, kguesses):
        return AES128_SBox_Out.sbox[pt_bytes ^ kguesses]


class AES128_InvSBox_Out(LeakageModel):
    name = "AES-128 Last Round Inverse S-Box (Ciphertext)"
    guess_space = 256
    num_targets = 16

    rsbox = np.array(
        [
            0x52,
            0x09,
            0x6A,
            0xD5,
            0x30,
            0x36,
            0xA5,
            0x38,
            0xBF,
            0x40,
            0xA3,
            0x9E,
            0x81,
            0xF3,
            0xD7,
            0xFB,
            0x7C,
            0xE3,
            0x39,
            0x82,
            0x9B,
            0x2F,
            0xFF,
            0x87,
            0x34,
            0x8E,
            0x43,
            0x44,
            0xC4,
            0xDE,
            0xE9,
            0xCB,
            0x54,
            0x7B,
            0x94,
            0x32,
            0xA6,
            0xC2,
            0x23,
            0x3D,
            0xEE,
            0x4C,
            0x95,
            0x0B,
            0x42,
            0xFA,
            0xC3,
            0x4E,
            0x08,
            0x2E,
            0xA1,
            0x66,
            0x28,
            0xD9,
            0x24,
            0xB2,
            0x76,
            0x5B,
            0xA2,
            0x49,
            0x6D,
            0x8B,
            0xD1,
            0x25,
            0x72,
            0xF8,
            0xF6,
            0x64,
            0x86,
            0x68,
            0x98,
            0x16,
            0xD4,
            0xA4,
            0x5C,
            0xCC,
            0x5D,
            0x65,
            0xB6,
            0x92,
            0x6C,
            0x70,
            0x48,
            0x50,
            0xFD,
            0xED,
            0xB9,
            0xDA,
            0x5E,
            0x15,
            0x46,
            0x57,
            0xA7,
            0x8D,
            0x9D,
            0x84,
            0x90,
            0xD8,
            0xAB,
            0x00,
            0x8C,
            0xBC,
            0xD3,
            0x0A,
            0xF7,
            0xE4,
            0x58,
            0x05,
            0xB8,
            0xB3,
            0x45,
            0x06,
            0xD0,
            0x2C,
            0x1E,
            0x8F,
            0xCA,
            0x3F,
            0x0F,
            0x02,
            0xC1,
            0xAF,
            0xBD,
            0x03,
            0x01,
            0x13,
            0x8A,
            0x6B,
            0x3A,
            0x91,
            0x11,
            0x41,
            0x4F,
            0x67,
            0xDC,
            0xEA,
            0x97,
            0xF2,
            0xCF,
            0xCE,
            0xF0,
            0xB4,
            0xE6,
            0x73,
            0x96,
            0xAC,
            0x74,
            0x22,
            0xE7,
            0xAD,
            0x35,
            0x85,
            0xE2,
            0xF9,
            0x37,
            0xE8,
            0x1C,
            0x75,
            0xDF,
            0x6E,
            0x47,
            0xF1,
            0x1A,
            0x71,
            0x1D,
            0x29,
            0xC5,
            0x89,
            0x6F,
            0xB7,
            0x62,
            0x0E,
            0xAA,
            0x18,
            0xBE,
            0x1B,
            0xFC,
            0x56,
            0x3E,
            0x4B,
            0xC6,
            0xD2,
            0x79,
            0x20,
            0x9A,
            0xDB,
            0xC0,
            0xFE,
            0x78,
            0xCD,
            0x5A,
            0xF4,
            0x1F,
            0xDD,
            0xA8,
            0x33,
            0x88,
            0x07,
            0xC7,
            0x31,
            0xB1,
            0x12,
            0x10,
            0x59,
            0x27,
            0x80,
            0xEC,
            0x5F,
            0x60,
            0x51,
            0x7F,
            0xA9,
            0x19,
            0xB5,
            0x4A,
            0x0D,
            0x2D,
            0xE5,
            0x7A,
            0x9F,
            0x93,
            0xC9,
            0x9C,
            0xEF,
            0xA0,
            0xE0,
            0x3B,
            0x4D,
            0xAE,
            0x2A,
            0xF5,
            0xB0,
            0xC8,
            0xEB,
            0xBB,
            0x3C,
            0x83,
            0x53,
            0x99,
            0x61,
            0x17,
            0x2B,
            0x04,
            0x7E,
            0xBA,
            0x77,
            0xD6,
            0x26,
            0xE1,
            0x69,
            0x14,
            0x63,
            0x55,
            0x21,
            0x0C,
            0x7D,
        ],
        dtype=np.uint8,
    )

    @staticmethod
    def get_intermediate(pt_bytes, kguesses):
        return AES128_InvSBox_Out.rsbox[pt_bytes ^ kguesses]


class AES128_Combined_SBox_Out(LeakageModel):
    name = "AES-128 Combined SBox+RSBox (Round 1 Shuffle Defeat)"
    guess_space = 256
    num_targets = 16

    @staticmethod
    def get_intermediate(pt_bytes, kguesses):
        # Both real (sbox) and fake (rsbox) operations in the shuffled round use
        # the same intermediate value (pt ^ k), so their combined HW is exploitable.
        x = pt_bytes ^ kguesses
        combined = AES128_SBox_Out.sbox[x].astype(np.uint16) + AES128_InvSBox_Out.rsbox[x].astype(np.uint16)
        return (combined & 0xFF).astype(np.uint8)


class Plaintext_XOR_Key(LeakageModel):
    name = "Plaintext XOR Key Input"
    guess_space = 256
    num_targets = 16

    @staticmethod
    def get_intermediate(pt_bytes, kguesses):
        return pt_bytes ^ kguesses


class PRESENT_Nibble_SBox_Out(LeakageModel):
    name = "PRESENT-80 S-Box Out (4-bit Nibble)"
    guess_space = 16
    num_targets = 16
    sbox = np.array(
        [
            0xC,
            0x5,
            0x6,
            0xB,
            0x9,
            0x0,
            0xA,
            0xD,
            0x3,
            0xE,
            0xF,
            0x8,
            0x4,
            0x7,
            0x1,
            0x2,
        ],
        dtype=np.uint8,
    )

    @staticmethod
    def get_intermediate(pt_nibbles, kguesses):
        return PRESENT_Nibble_SBox_Out.sbox[(np.asarray(pt_nibbles) & 0x0F) ^ kguesses]


LEAKAGE_MODELS = [
    AES128_SBox_Out,
    AES128_InvSBox_Out,
    AES128_Combined_SBox_Out,
    Plaintext_XOR_Key,
    PRESENT_Nibble_SBox_Out,
]

# --- AES-128 KEY SCHEDULE UTILITIES ---

def aes128_forward_key_schedule(k0_bytes):
    """Derive the AES-128 round-10 key (16 bytes) from the original key k0."""
    _sbox = AES128_SBox_Out.sbox
    _rcon = np.array([0x00, 0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x1b, 0x36], dtype=np.uint8)
    W = np.zeros((44, 4), dtype=np.uint8)
    k0 = np.array(list(k0_bytes), dtype=np.uint8)
    for i in range(4):
        W[i] = k0[i * 4 : (i + 1) * 4]
    for i in range(4, 44):
        temp = W[i - 1].copy()
        if i % 4 == 0:
            temp = _sbox[np.roll(temp, -1)]
            temp[0] ^= _rcon[i // 4]
        W[i] = W[i - 4] ^ temp
    return W[40:44].flatten().tolist()


def aes128_invert_key_schedule(k10_bytes):
    """Recover the AES-128 round-0 key k0 from the round-10 key k10 (16 bytes)."""
    _sbox = AES128_SBox_Out.sbox
    _rcon = np.array([0x00, 0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x1b, 0x36], dtype=np.uint8)
    # Represent as 4 words: rk[0]=W[4r], rk[1]=W[4r+1], rk[2]=W[4r+2], rk[3]=W[4r+3]
    rk = np.array(list(k10_bytes), dtype=np.uint8).reshape(4, 4)
    for r in range(10, 0, -1):
        A, B, C, D = rk[0].copy(), rk[1].copy(), rk[2].copy(), rk[3].copy()
        # Invert:  W[4r-1]=D^C, W[4r-2]=C^B, W[4r-3]=B^A, W[4r-4]=A^SubWord(RotWord(W[4r-1]))^Rcon[r]
        S = D ^ C
        R = C ^ B
        Q = B ^ A
        P = A ^ _sbox[np.roll(S, -1)]
        P[0] ^= _rcon[r]
        rk = np.array([P, Q, R, S])
    return rk.flatten().tolist()


# --- DSP FILTERS & ALIGNMENT ENGINES ---


def apply_poc_alignment(traces, window_start=0, window_end=500):
    num_traces, num_samples = traces.shape
    if window_end is None or window_end > num_samples: window_end = num_samples
    aligned = np.empty_like(traces)
    
    L = window_end - window_start
    window = np.hanning(L)
    
    raw_ref = np.mean(traces[:min(10, num_traces)], axis=0)[window_start:window_end]
    ref = (raw_ref - np.mean(raw_ref)) * window
    
    # FIX: Pad to 2*L - 1 to force Linear Convolution instead of Circular
    pad_len = 2 * L - 1
    F = np.fft.fft(ref, n=pad_len)
    
    for i in range(num_traces):
        raw_targ = traces[i, window_start:window_end]
        targ = (raw_targ - np.mean(raw_targ)) * window
        G = np.fft.fft(targ, n=pad_len)
        
        R = F * np.conj(G)
        R /= (np.abs(R) + 1e-12) 
        r = np.real(np.fft.ifft(R)) # Real part is sufficient for cross-corr
        
        shift = np.argmax(r)
        
        # Adjust for the padding offset
        if shift >= L: 
            shift -= pad_len
            
        # FIX: Shift MUST be inverted to pull the target back to the reference
        aligned[i] = shift_trace_safe(traces[i], -shift) 
    return aligned

# def apply_peak_alignment(traces, window_start=0, window_end=500):
#     num_traces, num_samples = traces.shape
#     aligned_traces = np.empty_like(traces)
#     reference_trace = traces[0]
#     ref_window = reference_trace[window_start:window_end]
#     aligned_traces[0] = reference_trace
    
#     for i in range(1, num_traces):
#         target_window = traces[i, window_start:window_end]
#         correlation = np.correlate(target_window, ref_window, mode='same')
#         best_offset = np.argmax(correlation) - (len(ref_window) // 2)
#         # BUG FIX: Safe Shift
#         aligned_traces[i] = shift_trace_safe(traces[i], -best_offset)
#     return aligned_traces


def apply_peak_alignment(traces, window_start=0, window_end=500):
    num_traces, num_samples = traces.shape
    aligned_traces = np.empty_like(traces)
    
    # 1. Use EXACTLY Trace 0 as the absolute reference (Never mean-average desynchronized traces)
    reference_trace = traces[0]

    # Validate window
    if window_end <= window_start: window_end = window_start + 1
    ref_window = reference_trace[window_start:window_end]

    aligned_traces[0] = reference_trace

    for i in range(1, num_traces):
        # 2. Expand search area massively to catch the 3200-cycle drift
        search_start = max(0, window_start - 4000)
        search_end = min(num_samples, window_end + 4000)
        search_window = traces[i, search_start:search_end]

        if len(search_window) < len(ref_window):
            aligned_traces[i] = traces[i]
            continue

        # 3. 'valid' mode slides the smaller template completely inside the massive search window
        correlation = np.correlate(search_window, ref_window, mode='valid')
        best_idx = np.argmax(correlation)

        # 4. Calculate exactly how many samples the trace drifted from Trace 0
        offset = (search_start + best_idx) - window_start

        # 5. Pull the trace perfectly back into alignment
        aligned_traces[i] = shift_trace_safe(traces[i], -offset)

    return aligned_traces


def apply_segment_alignment(traces, num_segments=8, max_warp=20):
    """Elastic windowed alignment with robust boundary edge padding."""
    num_traces, num_samples = traces.shape
    aligned_traces = np.empty_like(traces)
    reference = traces[0]
    aligned_traces[0] = reference
    seg_size = num_samples // num_segments

    if seg_size <= max_warp:
        # FIX: If segments are too small, fallback to Peak Alignment 
        # but artificially bound it so it doesn't violate max_warp constraints.
        print("Segments too small. Falling back to bounded peak alignment.")
        # Restrict the trace window so peak alignment can only search within max_warp bounds
        start_bound = max(0, (num_samples // 2) - max_warp * 2)
        end_bound = min(num_samples, (num_samples // 2) + max_warp * 2)
        return apply_peak_alignment(traces, start_bound, end_bound)

    for i in range(1, num_traces):
        reconstructed = []
        for s in range(num_segments):
            start_idx = s * seg_size
            end_idx = (s + 1) * seg_size if s < num_segments - 1 else num_samples
            ref_seg = reference[start_idx:end_idx]

            # Boundary-safe dynamic extraction padding
            pad_left = max(0, max_warp - start_idx)
            pad_right = max(0, (end_idx + max_warp) - num_samples)

            t_start = max(0, start_idx - max_warp)
            t_end = min(num_samples, end_idx + max_warp)
            target_window = traces[i, t_start:t_end]

            if pad_left > 0 or pad_right > 0:
                target_window = np.pad(
                    target_window, (pad_left, pad_right), mode="edge"
                )

            corr = np.correlate(target_window, ref_seg, mode="valid")
            best_offset = np.argmax(corr) - max_warp if len(corr) > 0 else 0

            shifted_start = max(0, start_idx + best_offset)
            shifted_end = shifted_start + (end_idx - start_idx)

            chunk = np.zeros(end_idx - start_idx, dtype=traces.dtype)
            actual_start = min(num_samples, shifted_start)
            actual_end = min(num_samples, shifted_end)

            copy_len = actual_end - actual_start
            if copy_len > 0:
                chunk[:copy_len] = traces[i, actual_start:actual_end]
            if copy_len < len(chunk):
                chunk = np.pad(chunk[:copy_len], (0, len(chunk) - copy_len), "edge")

            reconstructed.append(chunk)
        aligned_traces[i] = np.concatenate(reconstructed)[:num_samples]
    return aligned_traces


def apply_dtw_alignment(traces, window_start=0, window_end=500, warp_radius=20):
    try:
        from fastdtw import fastdtw
    except ImportError:
        print("fastdtw module missing. Skipping DTW Alignment.")
        return traces

    num_traces, num_samples = traces.shape
    aligned_traces = np.empty_like(traces)
    ref = traces[0, window_start:window_end]
    aligned_traces[0] = traces[0]

    # DTW Distance metric
    scalar_dist = lambda x, y: abs(x - y)

    for i in range(1, num_traces):
        targ = traces[i, window_start:window_end]
        
        # Calculate warp path
        _, path = fastdtw(ref, targ, radius=warp_radius, dist=scalar_dist)

        warp_trace = np.zeros(num_samples, dtype=np.float64)
        mapped = set()

        # Nearest-neighbor mapping to preserve voltage variance (critical for CPA)
        for px, py in path:
            abs_px = window_start + px
            abs_py = window_start + py
            
            if abs_px < num_samples and abs_py < num_samples:
                if abs_px not in mapped:
                    warp_trace[abs_px] = traces[i][abs_py]
                    mapped.add(abs_px)

        # Interpolate empty gaps to prevent zero-dropouts (dead zones)
        mask = warp_trace == 0
        if np.any(mask):
            valid_idx = np.where(~mask)[0]
            if len(valid_idx) > 0:
                warp_trace[mask] = np.interp(np.where(mask)[0], valid_idx, warp_trace[valid_idx])

        # Stitch the unaligned edges back onto the aligned window
        warp_trace[:window_start] = traces[i, :window_start]
        if window_end < num_samples:
            warp_trace[window_end:] = traces[i, window_end:]

        aligned_traces[i] = warp_trace.astype(traces.dtype)

        # Optional: Print progress so the user doesn't think the tool froze
        if i % 100 == 0:
            print(f"DTW Aligned {i}/{num_traces} traces...")

    return aligned_traces


def apply_peak_slicing(
    traces,
    window_size=15,
    expected_peaks=32,
    start_search=0,
    end_search=None,
    distance=20,
    prominence=0.01,
):
    num_traces, num_samples = traces.shape
    if end_search is None or end_search > num_samples:
        end_search = num_samples
    new_samples = expected_peaks * window_size
    compressed_traces = np.zeros((num_traces, new_samples), dtype=traces.dtype)
    half_w = window_size // 2

    ref_signal = np.abs(traces[0, start_search:end_search])
    ref_peaks, properties = find_peaks(
        ref_signal, distance=distance, prominence=prominence
    )

    if len(ref_peaks) > expected_peaks:
        best_peak_idx = np.argsort(properties["prominences"])[::-1][:expected_peaks]
        ref_peaks = np.sort(ref_peaks[best_peak_idx])
    ref_peaks = ref_peaks + start_search

    for i in range(num_traces):
        raw_signal = np.abs(traces[i])
        sub_signal = raw_signal[start_search:end_search]
        peaks, properties = find_peaks(
            sub_signal, distance=distance, prominence=prominence
        )

        if len(peaks) > expected_peaks:
            best_peak_idx = np.argsort(properties["prominences"])[::-1][:expected_peaks]
            t_peaks = np.sort(peaks[best_peak_idx]) + start_search
        elif len(peaks) < expected_peaks and len(ref_peaks) == expected_peaks:
            corr = np.correlate(sub_signal, ref_signal, mode="same")
            offset = np.argmax(corr) - (len(ref_signal) // 2)
            t_peaks = ref_peaks + offset
        else:
            t_peaks = peaks + start_search

        idx = 0
        for p in t_peaks:
            if idx >= new_samples:
                break
            p = int(np.clip(p, 0, num_samples - 1))
            start = max(0, p - half_w)
            end = min(num_samples, start + window_size)
            chunk = traces[i, start:end]

            if len(chunk) < window_size:
                chunk = np.pad(chunk, (0, window_size - len(chunk)), "edge")

            compressed_traces[i, idx : idx + window_size] = chunk
            idx += window_size

        # BUG FIX: Prevent trailing zeros if fewer peaks were found
        if idx < new_samples:
            remaining = new_samples - idx
            if idx > 0:
                # Repeat the last valid sequence of chunks to maintain variance
                repeats = np.tile(
                    compressed_traces[i, max(0, idx - window_size) : idx],
                    (remaining // window_size) + 1,
                )
                compressed_traces[i, idx:] = repeats[:remaining]
            else:
                # Total failure fallback: just copy standard data
                fallback_chunk = raw_signal[start_search : start_search + remaining]
                if len(fallback_chunk) < remaining:
                    fallback_chunk = np.pad(
                        fallback_chunk, (0, remaining - len(fallback_chunk)), "edge"
                    )
                compressed_traces[i, idx:] = fallback_chunk

    return compressed_traces


def apply_bandpass_filter(traces, lowcut=1_000_000, highcut=10_000_000, fs=20_000_000):
    nyquist = fs / 2.0
    # Strict bound limits for Scipy
    if highcut >= nyquist: highcut = nyquist - 1000
    if lowcut >= highcut: lowcut = highcut - 1000
    if lowcut <= 0: lowcut = 100 
    
    sos = butter(N=4, Wn=[lowcut, highcut], btype='band', fs=fs, output='sos')
    return sosfiltfilt(sos, traces, axis=1)


def apply_wavelet_denoising(traces, wavelet="db4", level=2):
    try:
        import pywt
    except ImportError:
        return savgol_filter(traces, window_length=9, polyorder=3, axis=1)

    num_traces, num_samples = traces.shape
    filtered_traces = np.empty_like(traces)

    mean_trace = np.mean(traces[: min(50, num_traces)], axis=0)
    global_coeffs = pywt.wavedec(mean_trace, wavelet, level=level)
    
    # FIX: Sigma must ONLY be estimated from the highest frequency detail (last array)
    sigma = np.median(np.abs(global_coeffs[-1])) / 0.6745
    
    global_thresholds = []
    for j in range(1, len(global_coeffs)):
        # Calculate universal threshold based on the single sigma
        global_thresholds.append(sigma * np.sqrt(2 * np.log(len(global_coeffs[j]))))

    for i in range(num_traces):
        coeffs = pywt.wavedec(traces[i], wavelet, level=level)
        for j in range(1, len(coeffs)):
            # FIX: Use "hard" thresholding to preserve amplitude scaling for CPA
            coeffs[j] = pywt.threshold(
                coeffs[j], value=global_thresholds[j - 1], mode="hard"
            )
        filtered_traces[i] = pywt.waverec(coeffs, wavelet)[:num_samples]

    return filtered_traces


def apply_gaussian_filter(traces, sigma=2.0):
    filtered_traces = np.empty_like(traces)
    gaussian_filter1d(traces, sigma=sigma, axis=1, output=filtered_traces)
    return filtered_traces


def apply_fft_magnitude(traces, start_sample=0, end_sample=None, cutoff_bin=None):
    if end_sample is not None: windowed_traces = traces[:, start_sample:end_sample]
    else: windowed_traces = traces[:, start_sample:]
        
    mean_removed = windowed_traces - np.mean(windowed_traces, axis=1, keepdims=True)
    hanning_window = np.hanning(mean_removed.shape[1])
    windowed_signals = mean_removed * hanning_window
    fft_data = np.abs(np.fft.rfft(windowed_signals, axis=1))
    # Keep DC removal but account for it in cutoff interpretation:
    if cutoff_bin is not None and cutoff_bin > 0:
        cutoff_bin = min(cutoff_bin + 1, fft_data.shape[1])  # compensate for the [:, 1:] shift
    fft_data = fft_data[:, 1:]  # remove DC after bounds calculation
    
    # BUG FIX: Respect the 0 value for dynamic unbounded processing
    if cutoff_bin is None or cutoff_bin == 0:
        cutoff_bin = fft_data.shape[1]
    else:
        cutoff_bin = min(cutoff_bin, fft_data.shape[1])
        
    return fft_data[:, :cutoff_bin]


def apply_pca_filtering(traces, n_components=5, drop_pc0=False):
    from sklearn.decomposition import PCA
    num_traces, num_samples = traces.shape
    comp_count = min(n_components + (1 if drop_pc0 else 0), num_samples, num_traces)
    
    if comp_count < 2: return traces

    pca = PCA(n_components=comp_count)
    transformed = pca.fit_transform(traces)

    if drop_pc0:
        transformed[:, 0] = 0 # Only drop if explicitly targeting massive macro-noise

    return pca.inverse_transform(transformed)


# --- VECTORIZED ENGINE PLUGINS ---

def shift_trace_safe(trace, shift_val):
    """Mathematically safe trace shifting without edge-wrapping."""
    shift_val = int(np.clip(shift_val, -(len(trace) - 1), len(trace) - 1))
    shifted = np.empty_like(trace)
    if shift_val > 0:
        shifted[shift_val:] = trace[:-shift_val]
        shifted[:shift_val] = trace[0] # Pad left
    elif shift_val < 0:
        shifted[:shift_val] = trace[-shift_val:]
        shifted[shift_val:] = trace[-1] # Pad right
    else:
        shifted[:] = trace
    return shifted


def analyze_byte(bnum, traces, pt_byte_column, leakage_model_class):
    """Robust Vectorized Target Leakage Engine using explicit shape enforcement."""
    traces = np.asarray(traces, dtype=np.float32)
    t_bar = np.mean(traces, axis=0)
    t_dev = traces - t_bar  # (N, S) float32
    # einsum avoids materializing the (N, S) squared temporary array
    sum_sq_t = np.einsum('ij,ij->j', t_dev, t_dev)

    space = leakage_model_class.guess_space

    # Force 1D Array then into Strict N x 1 shape (Fixes broken broadasts)
    pt_byte_column = np.asarray(pt_byte_column).flatten()
    kguesses = np.arange(space)[None, :]  # Shape: (1, Guess Space)

    intermediate = leakage_model_class.get_intermediate(
        pt_byte_column[:, None], kguesses
    )

    HW = get_hw_reference(space)
    # float32 keeps dot product in float32 (avoids numpy upcasting t_dev to float64)
    hw_guesses = HW[intermediate].astype(np.float32)  # Shape: (N, Guess Space)

    # Correlate properly matching the Trace Axis (axis=0)
    h_bar = np.mean(hw_guesses, axis=0, keepdims=True)
    h_dev = hw_guesses - h_bar
    sum_sq_h = np.sum(h_dev**2, axis=0)

    sum_prod = np.dot(h_dev.T, t_dev)  # (256, N) @ (N, S) → float32, no upcast

    # Epsilon prevents nan collapse when var approaches exact zero
    denominator = np.sqrt(sum_sq_h[:, None] * sum_sq_t[None, :])

    with np.errstate(divide="ignore", invalid="ignore"):
        corr = np.nan_to_num(sum_prod / (denominator + 1e-12))

    byte_correlations = np.abs(np.clip(corr, -1.0, 1.0))
    best_guess = int(np.argmax(np.max(byte_correlations, axis=1)))
    return bnum, best_guess, byte_correlations


def apply_max_pooling(traces, window_start, window_end, downsample_factor=10):
    num_traces, num_samples = traces.shape
    zone = traces[:, window_start:window_end]
    trimmed_width = (zone.shape[1] // downsample_factor) * downsample_factor
    reshaped = zone[:, :trimmed_width].reshape(
        (num_traces, trimmed_width // downsample_factor, downsample_factor)
    )
    # FIX: Removed np.abs() to preserve Pearson linearity
    return np.max(reshaped, axis=2)

def apply_sum_pooling(traces, window_start, window_end, downsample_factor=10):
    num_traces, num_samples = traces.shape
    zone = traces[:, window_start:window_end]
    trimmed_width = (zone.shape[1] // downsample_factor) * downsample_factor
    reshaped = zone[:, :trimmed_width].reshape(
        (num_traces, trimmed_width // downsample_factor, downsample_factor)
    )
    # FIX: Removed np.abs()
    return np.sum(reshaped, axis=2)

def apply_sliding_window_integration(traces, window_size=10):
    # FIX: Removed np.abs()
    return uniform_filter1d(traces, size=window_size, axis=1, mode="nearest") * window_size

def apply_variance_poi(traces, top_k=200):
    """Keep only the top-K highest-variance samples across all traces.

    Variance is a keyless proxy for SNR: samples where the device leaks
    information tend to have higher inter-trace variance than idle samples.
    """
    top_k = min(int(top_k), traces.shape[1])
    if top_k <= 0:
        return traces
    variances = np.var(traces, axis=0)
    poi_indices = np.argsort(variances)[-top_k:]
    return traces[:, np.sort(poi_indices)]

def compute_tvla(
    traces, pt_bytes, true_key_byte, leakage_model_class, fixed_vs_random_mask=None
):
    if fixed_vs_random_mask is not None:
        valid_traces_mask = np.ones(traces.shape[0], dtype=bool)
        group1_mask = fixed_vs_random_mask == 0
        group2_mask = ~group1_mask
    else:
        intermediate = leakage_model_class.get_intermediate(
            pt_bytes, np.array([[true_key_byte]])
        )[0]
        hw_arr = get_hw_reference(leakage_model_class.guess_space)[intermediate]

        num_bits = (leakage_model_class.guess_space - 1).bit_length()
        center_val = num_bits / 2.0

        # BUG FIX: Discard the statistical center to prevent mean dilution
        group1_mask = hw_arr < center_val
        group2_mask = hw_arr > center_val
        valid_traces_mask = hw_arr != center_val  # Ignore perfect center

    traces_g1 = traces[group1_mask & valid_traces_mask]
    traces_g2 = traces[group2_mask & valid_traces_mask]
    n1, n2 = traces_g1.shape[0], traces_g2.shape[0]

    if n1 < 2 or n2 < 2:
        return np.zeros(traces.shape[1])

    mu1, mu2 = np.mean(traces_g1, axis=0), np.mean(traces_g2, axis=0)
    var1, var2 = np.var(traces_g1, axis=0, ddof=1), np.var(traces_g2, axis=0, ddof=1)

    with np.errstate(divide="ignore", invalid="ignore"):
        t_stat = np.nan_to_num((mu1 - mu2) / np.sqrt((var1 / n1) + (var2 / n2) + 1e-12))
    return t_stat


def compute_pge(byte_correlations, true_key_byte):
    max_corr_per_guess = np.max(byte_correlations, axis=1)
    sorted_guesses = np.argsort(max_corr_per_guess)[::-1]
    return int(np.where(sorted_guesses == true_key_byte)[0][0])





def apply_dsp_pipeline(traces, dsp, full_traces=None):
    # Slice the window BEFORE copying so a narrow window avoids allocating
    # the full trace array (critical for large datasets like 100K×25K).
    pre_start = int(np.clip(int(dsp.get("slice_start", 0)), 0, traces.shape[1] - 1))
    pre_end   = int(np.clip(int(dsp.get("slice_end", traces.shape[1])), pre_start + 1, traces.shape[1]))
    working_traces = np.array(traces[:, pre_start:pre_end], dtype=np.float32)

    # =====================================================================
    # 1. CONTINUOUS FILTERS (Signal Conditioning)
    # Rule: Always clean the signal before aligning or processing it.
    # =====================================================================
    if dsp.get("bandpass_enabled", False):
        working_traces = apply_bandpass_filter(
            working_traces,
            dsp.get("bp_low", 1_000_000),
            dsp.get("bp_high", 10_000_000),
            dsp.get("bp_fs", 20_000_000),
        )
    if dsp.get("wavelet_enabled", False):
        working_traces = apply_wavelet_denoising(
            working_traces, dsp.get("wavelet_type", "db4"), dsp.get("wavelet_level", 1)
        )
    if dsp.get("gauss_enabled", False):
        working_traces = apply_gaussian_filter(
            working_traces, dsp.get("gauss_sigma", 2.0)
        )

    # =====================================================================
    # 2. TIME-DOMAIN STRUCTURAL ALIGNMENTS
    # Rule: Cross-correlation MUST happen on signed, raw voltage phases.
    # =====================================================================
    align_mode = dsp.get("align_mode", "None")
    if align_mode == "Peak Cross-Correlation":
        working_traces = apply_peak_alignment(
            working_traces, dsp.get("align_start", 0), dsp.get("align_end", 500)
        )
    elif align_mode == "Phase-Only Correlation (POC)":
        working_traces = apply_poc_alignment(
            working_traces, dsp.get("align_start", 0), dsp.get("align_end", 500)
        )
    elif align_mode == "Dynamic Time Warping (DTW)":
        working_traces = apply_dtw_alignment(
            working_traces, dsp.get("align_start", 0), dsp.get("align_end", 500)
        )

    if dsp.get("elastic_enabled", False):
        working_traces = apply_segment_alignment(
            working_traces, dsp.get("elastic_segs", 8), dsp.get("elastic_warp", 20)
        )

    # =====================================================================
    # 3. AMPLITUDE DEMODULATION (Masking Defeats)
    # Rule: Apply ONLY after signals are structurally aligned.
    # =====================================================================
    masking_mode = dsp.get("masking_mode", "None")
    if masking_mode == "Absolute Value Centering":
        working_traces = np.abs(working_traces - np.mean(working_traces, axis=0))
    elif masking_mode == "Trace Squaring":
        working_traces = np.power(working_traces - np.mean(working_traces, axis=0), 2)

    # =====================================================================
    # 4. JITTER DEFEATS / POOLING (Hardware Async-Clock Absorption)
    # Rule: Must be done strictly in the Time-Domain.
    # =====================================================================
    win_size = dsp.get("slice_size", 45)
    shuffle_mode = dsp.get("shuffle_mode", "None")

    if shuffle_mode != "None":
        # All shuffle modes respect the same start/end window bounds.
        w_start = int(np.clip(dsp.get("slice_start", 0), 0, working_traces.shape[1] - 1))
        w_end = int(np.clip(dsp.get("slice_end", working_traces.shape[1]), w_start + 1, working_traces.shape[1]))

        if shuffle_mode == "Integrated-Sum (Global)":
            zone = working_traces[:, w_start:w_end]
            np.abs(zone, out=zone)  # in-place: no copy of the (N, S) array
            working_traces = np.sum(zone, axis=1, keepdims=True)
        elif shuffle_mode == "Sliding Window Integration (SWI)":
            working_traces = apply_sliding_window_integration(
                working_traces[:, w_start:w_end], window_size=win_size
            )
        elif shuffle_mode == "Window-Max Pooling":
            working_traces = apply_max_pooling(
                working_traces, w_start, w_end, win_size
            )
        elif shuffle_mode == "Window-Sum Pooling":
            working_traces = apply_sum_pooling(
                working_traces, w_start, w_end, win_size
            )

    # =====================================================================
    # 4.5. VARIANCE POI SELECTION
    # Rule: Keep only the highest-variance samples before PCA/slicing.
    #       Mutually exclusive with peak slicing (both reduce dimensionality).
    # =====================================================================
    if dsp.get("poi_enabled", False) and not dsp.get("slice_enabled", False):
        working_traces = apply_variance_poi(working_traces, int(dsp.get("poi_topk", 200)))

    # =====================================================================
    # 5. SURGICAL COMPRESSION / SLICING
    # Rule: Cut specific peaks out before PCA.
    # =====================================================================
    if dsp.get("slice_enabled", False) and working_traces.shape[1] > 1:
        w_start = int(np.clip(dsp.get("slice_start", 0), 0, working_traces.shape[1] - 1))
        w_end = int(np.clip(dsp.get("slice_end", working_traces.shape[1]), w_start + 1, working_traces.shape[1]))

        working_traces = apply_peak_slicing(
            working_traces,
            window_size=win_size,
            expected_peaks=dsp.get("slice_count", 32),
            start_search=w_start,
            end_search=w_end,
            distance=dsp.get("slice_dist", 95),
            prominence=dsp.get("slice_prom", 0.02),
        )

    # =====================================================================
    # 6. MULTIVARIATE DIMENSIONALITY REDUCTION (PCA)
    # Rule: Compress the final time-domain arrays.
    # =====================================================================
    if dsp.get("pca_enabled", False):
        from sklearn.decomposition import PCA

        fit_target = (
            full_traces
            if (full_traces is not None and len(full_traces) <= 5000)
            else working_traces
        )

        if fit_target.shape[1] != working_traces.shape[1]:
            fit_target = working_traces

        comp_count = min(
            dsp.get("pca_comps", 5), fit_target.shape[1], fit_target.shape[0]
        )
        if comp_count > 0:
            pca = PCA(n_components=comp_count)
            pca.fit(fit_target)
            working_traces = pca.inverse_transform(pca.transform(working_traces))

    # =====================================================================
    # 7. DOMAIN TRANSFORMATION (Time -> Frequency)
    # Rule: MUST BE LAST. Converts the X-Axis to Frequency Bins.
    # =====================================================================
    if dsp.get("fft_enabled", False):
        if dsp.get("slice_enabled", False) or shuffle_mode == "Integrated-Sum (Global)":
            print("Warning: FFT bypassed. Cannot compute FFT on sliced/collapsed arrays.")
        else:
            f_start = int(np.clip(dsp.get("fft_start", 0), 0, working_traces.shape[1] - 1))
            f_end = dsp.get("fft_end")
            if f_end is not None and f_end > 0:
                f_end = int(np.clip(f_end, f_start + 1, working_traces.shape[1]))
            else:
                f_end = None
            
            working_traces = apply_fft_magnitude(
                working_traces, f_start, f_end, cutoff_bin=dsp.get("fft_cutoff", None)
            )

    return working_traces
