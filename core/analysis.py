import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import butter, sosfilt, find_peaks, savgol_filter

def get_hw_reference(space_size=256):
    """Generates a cacheable Hamming Weight reference array for any arbitrary byte/nibble width space."""
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
    
    sbox = np.array([
        0x63, 0x7C, 0x77, 0x7B, 0xF2, 0x6B, 0x6F, 0xC5, 0x30, 0x01, 0x67, 0x2B, 0xFE, 0xD7, 0xAB, 0x76,
        0xCA, 0x82, 0xC9, 0x7D, 0xFA, 0x59, 0x47, 0xF0, 0xAD, 0xD4, 0xA2, 0xAF, 0x9C, 0xA4, 0x72, 0xC0,
        0xB7, 0xFD, 0x93, 0x26, 0x36, 0x3F, 0xF7, 0xCC, 0x34, 0xA5, 0xE5, 0xF1, 0x71, 0xD8, 0x31, 0x15,
        0x04, 0xC7, 0x23, 0xC3, 0x18, 0x96, 0x05, 0x9A, 0x07, 0x12, 0x80, 0xE2, 0xEB, 0x27, 0xB2, 0x75,
        0x09, 0x83, 0x2C, 0x1A, 0x1B, 0x6E, 0x5A, 0xA0, 0x52, 0x3B, 0xD6, 0xB3, 0x29, 0xE3, 0x2F, 0x84,
        0x53, 0xD1, 0x00, 0xED, 0x20, 0xFC, 0xB1, 0x5B, 0x6A, 0xCB, 0xBE, 0x39, 0x4A, 0x4C, 0x58, 0xCF,
        0xD0, 0xEF, 0xAA, 0xFB, 0x43, 0x4D, 0x33, 0x85, 0x45, 0xF9, 0x02, 0x7F, 0x50, 0x3C, 0x9F, 0xA8,
        0x51, 0xA3, 0x40, 0x8F, 0x92, 0x9D, 0x38, 0xF5, 0xBC, 0xB6, 0xDA, 0x21, 0x10, 0xFF, 0xF3, 0xD2,
        0xCD, 0x0C, 0x13, 0xEC, 0x5F, 0x97, 0x44, 0x17, 0xC4, 0xA7, 0x7E, 0x3D, 0x64, 0x5D, 0x19, 0x73,
        0x60, 0x81, 0x4F, 0xDC, 0x22, 0x2A, 0x90, 0x88, 0x46, 0xEE, 0xB8, 0x14, 0xDE, 0x5E, 0x0B, 0xDB,
        0xE0, 0x32, 0x3A, 0x0A, 0x49, 0x06, 0x24, 0x5C, 0xC2, 0xD3, 0xAC, 0x62, 0x91, 0x95, 0xE4, 0x79,
        0xE7, 0xC8, 0x37, 0x6D, 0x8D, 0xD5, 0x4E, 0xA9, 0x6C, 0x56, 0xF4, 0xEA, 0x65, 0x7A, 0xAE, 0x08,
        0xBA, 0x78, 0x25, 0x2E, 0x1C, 0xA6, 0xB4, 0xC6, 0xE8, 0xDD, 0x74, 0x1F, 0x4B, 0xBD, 0x8B, 0x8A,
        0x70, 0x3E, 0xB5, 0x66, 0x48, 0x03, 0xF6, 0x0E, 0x61, 0x35, 0x57, 0xB9, 0x86, 0xC1, 0x1D, 0x9E,
        0xE1, 0xF8, 0x98, 0x11, 0x69, 0xD9, 0x8E, 0x94, 0x9B, 0x1E, 0x87, 0xE9, 0xCE, 0x55, 0x28, 0xDF,
        0x8C, 0xA1, 0x89, 0x0D, 0xBF, 0xE6, 0x42, 0x68, 0x41, 0x99, 0x2D, 0x0F, 0xB0, 0x54, 0xBB, 0x16
    ], dtype=np.uint8)

    @staticmethod
    def get_intermediate(pt_bytes, kguesses):
        return AES128_SBox_Out.sbox[pt_bytes ^ kguesses]

class AES128_InvSBox_Out(LeakageModel):
    name = "AES-128 Dummy Round Inverse S-Box"
    guess_space = 256
    num_targets = 16
    
    rsbox = np.array([
        0x52, 0x09, 0x6A, 0xD5, 0x30, 0x36, 0xA5, 0x38, 0xBF, 0x40, 0xA3, 0x9E, 0x81, 0xF3, 0xD7, 0xFB,
        0x7C, 0xE3, 0x39, 0x82, 0x9B, 0x2F, 0xFF, 0x87, 0x34, 0x8E, 0x43, 0x44, 0xC4, 0xDE, 0xE9, 0xCB,
        0x54, 0x7B, 0x94, 0x32, 0xA6, 0xC2, 0x23, 0x3D, 0xEE, 0x4C, 0x95, 0x0B, 0x42, 0xFA, 0xC3, 0x4E,
        0x08, 0x2E, 0xA1, 0x66, 0x28, 0xD9, 0x24, 0xB2, 0x76, 0x5B, 0xA2, 0x49, 0x6D, 0x8B, 0xD1, 0x25,
        0x72, 0xF8, 0xF6, 0x64, 0x86, 0x68, 0x98, 0x16, 0xD4, 0xA4, 0x5C, 0xCC, 0x5D, 0x65, 0xB6, 0x92,
        0x6C, 0x70, 0x48, 0x50, 0xFD, 0xED, 0xB9, 0xDA, 0x5E, 0x15, 0x46, 0x57, 0xA7, 0x8D, 0x9D, 0x84,
        0x90, 0xD8, 0xAB, 0x00, 0x8C, 0xBC, 0xD3, 0x0A, 0xF7, 0xE4, 0x58, 0x05, 0xB8, 0xB3, 0x45, 0x06,
        0xD0, 0x2C, 0x1E, 0x8F, 0xCA, 0x3F, 0x0F, 0x02, 0xC1, 0xAF, 0xBD, 0x03, 0x01, 0x13, 0x8A, 0x6B,
        0x3A, 0x91, 0x11, 0x41, 0x4F, 0x67, 0xDC, 0xEA, 0x97, 0xF2, 0xCF, 0xCE, 0xF0, 0xB4, 0xE6, 0x73,
        0x96, 0xAC, 0x74, 0x22, 0xE7, 0xAD, 0x35, 0x85, 0xE2, 0xF9, 0x37, 0xE8, 0x1C, 0x75, 0xDF, 0x6E,
        0x47, 0xF1, 0x1A, 0x71, 0x1D, 0x29, 0xC5, 0x89, 0x6F, 0xB7, 0x62, 0x0E, 0xAA, 0x18, 0xBE, 0x1B,
        0xFC, 0x56, 0x3E, 0x4B, 0xC6, 0xD2, 0x79, 0x20, 0x9A, 0xDB, 0xC0, 0xFE, 0x78, 0xCD, 0x5A, 0xF4,
        0x1F, 0xDD, 0xA8, 0x33, 0x88, 0x07, 0xC7, 0x31, 0xB1, 0x12, 0x10, 0x59, 0x27, 0x80, 0xEC, 0x5F,
        0x60, 0x51, 0x7F, 0xA9, 0x19, 0xB5, 0x4A, 0x0D, 0x2D, 0xE5, 0x7A, 0x9F, 0x93, 0xC9, 0x9C, 0xEF,
        0xA0, 0xE0, 0x3B, 0x4D, 0xAE, 0x2A, 0xF5, 0xB0, 0xC8, 0xEB, 0xBB, 0x3C, 0x83, 0x53, 0x99, 0x61,
        0x17, 0x2B, 0x04, 0x7E, 0xBA, 0x77, 0xD6, 0x26, 0xE1, 0x69, 0x14, 0x63, 0x55, 0x21, 0x0C, 0x7D
    ], dtype=np.uint8)

    @staticmethod
    def get_intermediate(pt_bytes, kguesses):
        return AES128_InvSBox_Out.rsbox[pt_bytes ^ kguesses]

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
    sbox = np.array([0xC, 0x5, 0x6, 0xB, 0x9, 0x0, 0xA, 0xD, 0x3, 0xE, 0xF, 0x8, 0x4, 0x7, 0x1, 0x2], dtype=np.uint8)

    @staticmethod
    def get_intermediate(pt_nibbles, kguesses):
        return PRESENT_Nibble_SBox_Out.sbox[pt_nibbles ^ kguesses]

LEAKAGE_MODELS = [
    AES128_SBox_Out,
    Plaintext_XOR_Key,
    PRESENT_Nibble_SBox_Out,
    AES128_InvSBox_Out
]

# --- DSP FILTERS & ALIGNMENT ENGNES ---

def apply_poc_alignment(traces, window_start=0, window_end=500):
    """Phase-Only Correlation to rigidly align traces overcoming massive random delay horizontal shifting."""
    num_traces, num_samples = traces.shape
    if window_end is None or window_end > num_samples: window_end = num_samples
    aligned = np.empty_like(traces)
    ref = traces[0, window_start:window_end]
    aligned[0] = traces[0]
    
    F = np.fft.fft(ref)
    for i in range(1, num_traces):
        targ = traces[i, window_start:window_end]
        G = np.fft.fft(targ)
        R = F * np.conj(G)
        R /= (np.abs(R) + 1e-10) 
        r = np.fft.ifft(R)
        shift = np.argmax(np.abs(r))
        if shift > len(ref) // 2:
            shift -= len(ref)
        aligned[i] = np.roll(traces[i], shift)
    return aligned

def apply_peak_alignment(traces, window_start=0, window_end=500):
    """Standard time-domain cross-correlation alignment."""
    num_traces, num_samples = traces.shape
    aligned_traces = np.empty_like(traces)
    reference_trace = traces[0]
    ref_window = reference_trace[window_start:window_end]
    aligned_traces[0] = reference_trace
    
    for i in range(1, num_traces):
        target_window = traces[i, window_start:window_end]
        correlation = np.correlate(target_window, ref_window, mode='same')
        best_offset = np.argmax(correlation) - (len(ref_window) // 2)
        aligned_traces[i] = np.roll(traces[i], -best_offset)
    return aligned_traces

def apply_segment_alignment(traces, num_segments=8, max_warp=20):
    """Elastic windowed alignment for standard uniform shift variations."""
    num_traces, num_samples = traces.shape
    aligned_traces = np.empty_like(traces)
    reference = traces[0]
    aligned_traces[0] = reference
    seg_size = num_samples // num_segments
    if seg_size <= max_warp:
        return traces

    for i in range(1, num_traces):
        reconstructed = []
        for s in range(num_segments):
            start_idx = s * seg_size
            end_idx = (s + 1) * seg_size if s < num_segments - 1 else num_samples
            ref_seg = reference[start_idx:end_idx]
            t_start = max(0, start_idx - max_warp)
            t_end = min(num_samples, end_idx + max_warp)
            target_window = traces[i, t_start:t_end]
            corr = np.correlate(target_window, ref_seg, mode='valid')
            best_offset = (np.argmax(corr) - max_warp) if len(corr) > 0 else 0
            shifted_start = max(0, start_idx + best_offset)
            shifted_end = min(num_samples, shifted_start + (end_idx - start_idx))
            chunk = traces[i, shifted_start:shifted_end]
            if len(chunk) < (end_idx - start_idx):
                chunk = np.pad(chunk, (0, (end_idx - start_idx) - len(chunk)), 'edge')
            reconstructed.append(chunk)
        aligned_traces[i] = np.concatenate(reconstructed)[:num_samples]
    return aligned_traces

def apply_dtw_alignment(traces, window_start=0, window_end=500, warp_radius=5): # Reduced default radius
    """Dynamic Time Warping (Throttled Multithreading to prevent OOM crashes)."""
    try:
        from fastdtw import fastdtw
        import concurrent.futures
    except ImportError:
        print("fastdtw module missing. Skipping DTW Alignment.")
        return traces
        
    num_traces, num_samples = traces.shape
    aligned_traces = np.empty_like(traces)
    ref = traces[0, window_start:window_end]
    aligned_traces[0] = traces[0]
    
    scalar_dist = lambda x, y: abs(x - y)
    
    def process_trace(i):
        targ = traces[i, window_start:window_end]
        _, path = fastdtw(ref, targ, radius=warp_radius, dist=scalar_dist)
        
        warp_trace = np.zeros_like(traces[0])
        counts = np.zeros_like(traces[0])
        
        for px, py in path: 
            abs_px = window_start + px
            abs_py = window_start + py
            if abs_px < num_samples and abs_py < num_samples:
                warp_trace[abs_px] += traces[i][abs_py] 
                counts[abs_px] += 1
                
        counts[counts == 0] = 1 
        result_trace = warp_trace / counts
        result_trace[:window_start] = traces[i, :window_start]
        if window_end < num_samples:
            result_trace[window_end:] = traces[i, window_end:]
            
        return i, result_trace

    import os
    # CRITICAL FIX: Throttle DTW to prevent memory exhaustion (OOM crashes)
    workers = min(4, os.cpu_count() or 1) 
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(process_trace, i): i for i in range(1, num_traces)}
        for future in concurrent.futures.as_completed(futures):
            idx, res = future.result()
            aligned_traces[idx] = res
            
    return aligned_traces

def apply_peak_slicing(traces, window_size=15, expected_peaks=32, start_search=0, end_search=None, distance=20, prominence=0.01):
    """Isolates operation zones by finding the most prominent peaks, ignoring initial noise."""
    num_traces, num_samples = traces.shape
    if end_search is None or end_search > num_samples:
        end_search = num_samples
    new_samples = expected_peaks * window_size
    compressed_traces = np.zeros((num_traces, new_samples), dtype=traces.dtype)
    half_w = window_size // 2
    
    # Golden Reference Extraction
    ref_signal = np.abs(traces[0, start_search:end_search])
    ref_peaks, properties = find_peaks(ref_signal, distance=distance, prominence=prominence)
    
    if len(ref_peaks) > expected_peaks:
        best_peak_idx = np.argsort(properties['prominences'])[::-1][:expected_peaks]
        ref_peaks = np.sort(ref_peaks[best_peak_idx])
        
    ref_peaks = ref_peaks + start_search
    
    for i in range(num_traces):
        raw_signal = np.abs(traces[i])
        sub_signal = raw_signal[start_search:end_search]
        peaks, properties = find_peaks(sub_signal, distance=distance, prominence=prominence)
        
        if len(peaks) > expected_peaks:
            best_peak_idx = np.argsort(properties['prominences'])[::-1][:expected_peaks]
            t_peaks = np.sort(peaks[best_peak_idx]) + start_search
        elif len(peaks) < expected_peaks and len(ref_peaks) == expected_peaks:
            # Fallback to cross-correlation matching using the reference template's peak structure
            corr = np.correlate(sub_signal, ref_signal, mode='same')
            offset = np.argmax(corr) - (len(ref_signal) // 2)
            t_peaks = ref_peaks + offset 
        else:
            t_peaks = peaks + start_search # Failsafe
            
        idx = 0
        for p in t_peaks:
            # --- BUG FIX: Clamp peak index to bounds ---
            p = int(np.clip(p, 0, num_samples - 1))
            
            start = max(0, p - half_w)
            end = min(num_samples, start + window_size)
            chunk = traces[i, start:end]
            
            # --- BUG FIX: Handle empty chunks gracefully ---
            if len(chunk) == 0:
                chunk = np.zeros(window_size, dtype=traces.dtype)
            elif len(chunk) < window_size:
                chunk = np.pad(chunk, (0, window_size - len(chunk)), 'edge')
                
            compressed_traces[i, idx:idx+window_size] = chunk
            idx += window_size
            
    return compressed_traces

def apply_bandpass_filter(traces, lowcut=1_000_000, highcut=10_000_000, fs=20_000_000):
    if highcut >= (fs / 2):
        highcut = (fs / 2) - 1000
    sos = butter(N=4, Wn=[lowcut, highcut], btype='band', fs=fs, output='sos')
    return sosfilt(sos, traces, axis=1)

def apply_wavelet_denoising(traces, wavelet='db4', level=1):
    try:
        import pywt
    except ImportError:
        return savgol_filter(traces, window_length=9, polyorder=3, axis=1)
        
    num_traces, num_samples = traces.shape
    filtered_traces = np.empty_like(traces)
    for i in range(num_traces):
        coeffs = pywt.wavedec(traces[i], wavelet, level=level)
        for j in range(1, len(coeffs)):
            sigma = np.median(np.abs(coeffs[j])) / 0.6745
            threshold = sigma * np.sqrt(2 * np.log(len(coeffs[j])))
            coeffs[j] = pywt.threshold(coeffs[j], value=threshold, mode='soft')
        filtered_traces[i] = pywt.waverec(coeffs, wavelet)[:num_samples]
    return filtered_traces

def apply_gaussian_filter(traces, sigma=2.0):
    filtered_traces = np.empty_like(traces)
    gaussian_filter1d(traces, sigma=sigma, axis=1, output=filtered_traces)
    return filtered_traces

def apply_fft_magnitude(traces, start_sample=0, end_sample=None):
    if end_sample is not None:
        windowed_traces = traces[:, start_sample:end_sample]
    else:
        windowed_traces = traces[:, start_sample:]
    mean_removed = windowed_traces - np.mean(windowed_traces, axis=1, keepdims=True)
    return np.abs(np.fft.rfft(mean_removed, axis=1))[:, 1:]

def apply_pca_filtering(traces, n_components=5):
    from sklearn.decomposition import PCA
    num_traces, num_samples = traces.shape
    comp_count = min(n_components, num_samples, num_traces)
    pca = PCA(n_components=comp_count)
    return pca.inverse_transform(pca.fit_transform(traces))

# --- VECTORIZED ENGINE PLUGINS ---

def analyze_byte(bnum, traces, pt_byte_column, leakage_model_class):
    """Vectorized Generic Target Leakage Engine."""
    t_bar = np.mean(traces, axis=0)
    t_dev = traces - t_bar
    sum_sq_t = np.sum(t_dev**2, axis=0)

    space = leakage_model_class.guess_space
    kguesses = np.arange(space)[:, None]
    intermediate = leakage_model_class.get_intermediate(pt_byte_column, kguesses)
    
    HW = get_hw_reference(space)
    hw_guesses = HW[intermediate].astype(np.float64)

    h_bar = np.mean(hw_guesses, axis=1, keepdims=True)
    h_dev = hw_guesses - h_bar
    sum_sq_h = np.sum(h_dev**2, axis=1, keepdims=True)

    sum_prod = np.dot(h_dev, t_dev)
    denominator = np.sqrt(sum_sq_h * sum_sq_t)

    with np.errstate(divide="ignore", invalid="ignore"):
        corr = np.nan_to_num(sum_prod / denominator)

    byte_correlations = np.abs(corr)
    best_guess = int(np.argmax(np.max(byte_correlations, axis=1)))
    return bnum, best_guess, byte_correlations

def apply_max_pooling(traces, window_start, window_end, downsample_factor=10):
    num_traces, num_samples = traces.shape
    zone = traces[:, window_start:window_end]
    zone_width = zone.shape[1]
    trimmed_width = (zone_width // downsample_factor) * downsample_factor
    zone_trimmed = zone[:, :trimmed_width]
    
    reshaped = zone_trimmed.reshape((num_traces, trimmed_width // downsample_factor, downsample_factor))
    return np.max(np.abs(reshaped), axis=2)

def apply_sum_pooling(traces, window_start, window_end, downsample_factor=10):
    """Collapses the time domain by integrating energy. Best defense against internal cycle clock jitter."""
    num_traces, num_samples = traces.shape
    zone = traces[:, window_start:window_end]
    zone_width = zone.shape[1]
    trimmed_width = (zone_width // downsample_factor) * downsample_factor
    zone_trimmed = zone[:, :trimmed_width]
    
    reshaped = zone_trimmed.reshape((num_traces, trimmed_width // downsample_factor, downsample_factor))
    return np.sum(np.abs(reshaped), axis=2)

def compute_tvla(traces, pt_bytes, true_key_byte, leakage_model_class, fixed_vs_random_mask=None):
    """Performs dynamic Welch's t-test calculations using robust Hamming Weight thresholds."""
    if fixed_vs_random_mask is not None:
        group1_mask = (fixed_vs_random_mask == 0)
    else:
        intermediate = leakage_model_class.get_intermediate(pt_bytes, np.array([[true_key_byte]]))[0]
        # True non-specific thresholding based on the data architecture
        hw_arr = get_hw_reference(leakage_model_class.guess_space)[intermediate]
        group1_mask = hw_arr < (leakage_model_class.guess_space.bit_length() / 2)
        
    group2_mask = ~group1_mask
    traces_g1 = traces[group1_mask]
    traces_g2 = traces[group2_mask]
    n1, n2 = traces_g1.shape[0], traces_g2.shape[0]
    
    if n1 < 2 or n2 < 2:
        return np.zeros(traces.shape[1])
        
    mu1, mu2 = np.mean(traces_g1, axis=0), np.mean(traces_g2, axis=0)
    var1, var2 = np.var(traces_g1, axis=0, ddof=1), np.var(traces_g2, axis=0, ddof=1)
    
    with np.errstate(divide='ignore', invalid='ignore'):
        t_stat = np.nan_to_num((mu1 - mu2) / np.sqrt((var1 / n1) + (var2 / n2)))
    return t_stat

def compute_pge(byte_correlations, true_key_byte):
    max_corr_per_guess = np.max(byte_correlations, axis=1)
    sorted_guesses = np.argsort(max_corr_per_guess)[::-1]
    return int(np.where(sorted_guesses == true_key_byte)[0][0])


def apply_sliding_window_integration(traces, window_size=10):
    """Absorbs clock jitter without needing a rigid reference trace alignment."""
    kernel = np.ones(window_size)
    # Mode 'same' maintains trace length, convolving across the time axis
    return np.apply_along_axis(lambda m: np.convolve(np.abs(m), kernel, mode='same'), axis=1, arr=traces)

# --- NEW UNIFIED PIPELINE (ADD TO BOTTOM OF FILE) ---
def apply_dsp_pipeline(traces, dsp, full_traces=None):
    """Single source of truth for the DSP execution order to fix UI/Worker drift."""
    working_traces = traces.copy()

    # 1. Alignments
    if dsp.get('align_mode') == 'Peak Cross-Correlation':
        working_traces = apply_peak_alignment(working_traces, dsp.get('align_start', 0), dsp.get('align_end', 500))
    elif dsp.get('align_mode') == 'Phase-Only Correlation (POC)':
        working_traces = apply_poc_alignment(working_traces, dsp.get('align_start', 0), dsp.get('align_end', 500))
    elif dsp.get('align_mode') == 'Dynamic Time Warping (DTW)':
        working_traces = apply_dtw_alignment(working_traces, dsp.get('align_start', 0), dsp.get('align_end', 500))

    if dsp.get('elastic_enabled', False):
        working_traces = apply_segment_alignment(working_traces, dsp.get('elastic_segs', 8), dsp.get('elastic_warp', 20))

    # 2. Continuous Filters
    if dsp.get('bandpass_enabled', False):
        working_traces = apply_bandpass_filter(working_traces, dsp.get('bp_low', 1_000_000), dsp.get('bp_high', 10_000_000), dsp.get('bp_fs', 20_000_000))
    if dsp.get('wavelet_enabled', False):
        working_traces = apply_wavelet_denoising(working_traces, dsp.get('wavelet_type', 'db4'), dsp.get('wavelet_level', 1))
    if dsp.get('gauss_enabled', False):
        working_traces = apply_gaussian_filter(working_traces, dsp.get('gauss_sigma', 2.0))

    # 3. Frequency & Dim Transforms
    if dsp.get('fft_enabled', False):
        working_traces = apply_fft_magnitude(working_traces, dsp.get('fft_start', 0), dsp.get('fft_end', None))
    
    if dsp.get('pca_enabled', False):
        from sklearn.decomposition import PCA
        # If in preview mode, fit PCA on a larger subset to get accurate components, not just 10 traces
        fit_target = full_traces if (full_traces is not None and len(full_traces) <= 5000) else working_traces
        comp_count = min(dsp.get('pca_comps', 5), fit_target.shape[1], fit_target.shape[0])
        pca = PCA(n_components=comp_count)
        pca.fit(fit_target)
        working_traces = pca.inverse_transform(pca.transform(working_traces))

    # 4. Masking Defeats (Higher-Order)
    masking_mode = dsp.get('masking_mode', 'None')
    if masking_mode == 'Absolute Value Centering':
        working_traces = np.abs(working_traces - np.mean(working_traces, axis=0))
    elif masking_mode == 'Trace Squaring':
        working_traces = np.power(working_traces - np.mean(working_traces, axis=0), 2)

    # 5. Peak Slicing (Desync Extraction)
    w_start = dsp.get('slice_start', 1000)
    w_end = dsp.get('slice_end', 4800)
    win_size = dsp.get('slice_size', 45) # Global window size parameter

    if dsp.get('slice_enabled', False):
        working_traces = apply_peak_slicing(
            working_traces, 
            window_size=win_size, 
            expected_peaks=dsp.get('slice_count', 32),
            start_search=w_start, 
            end_search=w_end,
            distance=dsp.get('slice_dist', 95), 
            prominence=dsp.get('slice_prom', 0.02)
        )

    # 6. Jitter/Shuffling Defeats (Time Compression)
    shuffle_mode = dsp.get('shuffle_mode', 'None')
    
    if shuffle_mode == 'Integrated-Sum (Global)':
        # If sliced, sum the remaining sliced array. If not sliced, sum the target window.
        if dsp.get('slice_enabled', False):
            working_traces = np.sum(np.abs(working_traces), axis=1, keepdims=True)
        else:
            working_traces = np.sum(np.abs(working_traces[:, w_start:w_end]), axis=1, keepdims=True)
            
    elif shuffle_mode == 'Sliding Window Integration (SWI)':
        working_traces = apply_sliding_window_integration(working_traces, window_size=win_size)
        
    elif shuffle_mode == 'Window-Max Pooling':
        working_traces = apply_max_pooling(working_traces, 0, working_traces.shape[1], win_size)
        
    elif shuffle_mode == 'Window-Sum Pooling':
        working_traces = apply_sum_pooling(working_traces, 0, working_traces.shape[1], win_size)

    return working_traces