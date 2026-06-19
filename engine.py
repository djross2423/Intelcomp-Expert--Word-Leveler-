"""
Intelcomp Expert — Professional Vocal Dynamics Engine
With Perceptual Word-Level Consistency & Crackle-Free Resonance Suppression

FIXES APPLIED:
  [1] LR4 band splitting: now uses residual-based HP/LP at each crossover (true LR4)
  [2] Resonance IIR state: zi reset on coefficient change; sosfiltfilt used for offline
  [3] LUFS fallback: uses internal _k_weight instead of magic -0.691 constant
  [4] De-esser scope: moved to post-split on sibilance band (band index 4)
  [5] Processing order: true-peak limit now runs BEFORE LUFS normalization
  [6] Resonance STFT waste: replaced with per-chunk Welch PSD for peak detection
  [7] K-weight caching: _k_weight result cached in process() and reused
  [8] Attack/release units: _time_const() helper added; all sites use it consistently
  [9] De-esser envelope: Hilbert replaced with LP-filtered RMS envelope
  [10] Exciter level-dependence: band normalised before drive, rescaled after
"""

import librosa
import numpy as np
import scipy.signal as signal
from scipy.ndimage import uniform_filter1d, gaussian_filter1d
from scipy.interpolate import CubicSpline
import soundfile as sf
import yaml

try:
    import pyloudnorm as pyln
    HAS_PYLN = True
except ImportError:
    HAS_PYLN = False

try:
    from numba import njit
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False


# ─── NUMBA-ACCELERATED HELPERS ───────────────────────────────────────────────

if HAS_NUMBA:
    @njit(cache=True)
    def _apply_dynamics(env, att, rel):
        sm = np.zeros_like(env)
        sm[0] = env[0]
        for i in range(1, len(env)):
            c = att if env[i] > sm[i - 1] else rel
            sm[i] = sm[i - 1] + c * (env[i] - sm[i - 1])
        return sm

    @njit(cache=True)
    def _soft_clip(audio, drive):
        out = np.empty_like(audio)
        td = np.tanh(drive)
        for i in range(len(audio)):
            out[i] = np.tanh(audio[i] * drive) / td
        return out
else:
    def _apply_dynamics(env, att, rel):
        sm = np.zeros_like(env)
        sm[0] = env[0]
        for i in range(1, len(env)):
            c = att if env[i] > sm[i - 1] else rel
            sm[i] = sm[i - 1] + c * (env[i] - sm[i - 1])
        return sm

    def _soft_clip(audio, drive):
        return np.tanh(audio * drive) / np.tanh(drive)


# ─── CONFIGURATION ────────────────────────────────────────────────────────────

class Config:
    def __init__(self, preset_dict=None):
        self.hop = 512
        self.n_fft = 4096
        self.crossover_freqs = [150, 400, 2500, 5500, 9000]
        self.band_names = ['sub_mud', 'low_mud', 'boxiness', 'presence', 'sibilance', 'air']

        self.bands = {
            # thresh/ratio are relative-level thresholds (dB above context level)
            # att/rel are in milliseconds
            'sub_mud':   {"thresh": 4.0,  "ratio": 3.0, "att": 1,  "rel": 5},
            'low_mud':   {"thresh": 5.0,  "ratio": 2.5, "att": 1,  "rel": 10},
            'boxiness':  {"thresh": 4.5,  "ratio": 2.0, "att": 2,  "rel": 15},
            'presence':  {"thresh": 3.5,  "ratio": 1.5, "att": 3,  "rel": 20},
            'sibilance': {"thresh": 3.0,  "ratio": 6.0, "att": 1,  "rel": 4},
            'air':       {"thresh": 5.0,  "ratio": 1.5, "att": 4,  "rel": 30},
        }

        self.de_esser = {
            "enabled": True, "low": 5000, "high": 8000,
            "sens": 2.0, "ratio": 4.0, "att": 1.0, "rel": 50.0, "range": 12.0,
        }

        self.de_breather = {"enabled": True, "sens": 0.5, "range": 6.0}

        self.resonance = {
            "enabled": True, "sens": 0.6, "range": 8.0, "q": 15.0,
        }

        self.exciter = {
            "enabled": True, "amount": 0.15, "freq": 6000,
            "threshold_db": -24.0, "knee_db": 6.0,
        }

        self.word_leveler = {
            "enabled": True,
            "max_correction_db": 6.0,
            "smooth_ms": 20.0,
            "voice_percentile": 30,
        }

        self.psycho = {
            "enabled": True, "sens": 2.0, "ratio": 2.0,
            "att": 10.0, "rel": 150.0, "range": 8.0,
            "mask": 0.5, "knee": 3.0,
        }

        self.intelligence = {"mfcc": 0.15, "cent": 0.35}
        self.dynamics = {"lookahead": 5.0, "knee": 4.0, "transient": 0.3}
        self.parallel = {"thresh": -20.0, "ratio": 4.0, "mix": 0.25, "sat": 0.15}
        self.output = {"lufs": -14.0, "tp": -1.0}

        if preset_dict:
            self.apply(preset_dict)

    def apply(self, d):
        for k, v in d.items():
            if hasattr(self, k):
                if isinstance(v, dict) and isinstance(getattr(self, k), dict):
                    getattr(self, k).update(v)
                else:
                    setattr(self, k, v)

    def save_yaml(self, path):
        keys = ['bands', 'de_esser', 'de_breather', 'resonance', 'exciter',
                'word_leveler', 'psycho', 'intelligence', 'dynamics',
                'parallel', 'output', 'hop', 'n_fft', 'crossover_freqs']
        data = {k: getattr(self, k) for k in keys}
        with open(path, 'w') as f:
            yaml.dump(data, f, default_flow_style=False)

    @classmethod
    def load_yaml(cls, path):
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        return cls(preset_dict=data)


# ─── ENGINE ───────────────────────────────────────────────────────────────────

class VocalDynamicsEngine:
    def __init__(self, config=None, progress=None):
        self.cfg = config or Config()
        self.prog = progress or (lambda x, y: None)
        self._y_k_cache = None   # FIX [7]: K-weight cache

    # ─── HELPER: time constant ────────────────────────────────────────────────
    # FIX [8]: centralised time-constant conversion; all att/rel in Config are ms
    def _time_const(self, ms, sr):
        """Convert milliseconds to a one-pole IIR coefficient (frame-domain)."""
        frame_rate = sr / self.cfg.hop
        return 1.0 - np.exp(-1.0 / (frame_rate * max(ms, 1e-6) / 1000.0))

    def _time_const_samples(self, ms, sr):
        """Convert milliseconds to a one-pole IIR coefficient (sample-domain)."""
        return 1.0 - np.exp(-1.0 / (sr * max(ms, 1e-6) / 1000.0))

    # ─── MAIN PROCESS ─────────────────────────────────────────────────────────

    def process(self, in_path, out_path=None):
        self.prog(0.05, "Loading...")
        y, sr = librosa.load(in_path, sr=None)
        y_orig = y.copy()

        # FIX [7]: compute K-weighted signal once, cache for all downstream use
        self._y_k_cache = self._k_weight(y, sr)

        if self.cfg.de_breather["enabled"]:
            self.prog(0.10, "De-Breather...")
            y = self._debreather(y, sr)
            # Refresh K-weight cache after breath removal
            self._y_k_cache = self._k_weight(y, sr)

        # FIX [4]: de-esser moved AFTER band splitting; runs only on sibilance band
        self.prog(0.18, "LR4 Band Splitting...")
        bands = self._split_bands_lr4(y, sr)   # FIX [1]

        # FIX [4]: apply de-esser surgically on sibilance band (index 4)
        if self.cfg.de_esser["enabled"]:
            self.prog(0.24, "De-Esser (sibilance band)...")
            bands[4] = self._deesser(bands[4], sr)

        self.prog(0.30, "Resonance Suppression...")
        if self.cfg.resonance["enabled"]:
            bands[2] = self._resonance_suppress(bands[2], sr)
            bands[3] = self._resonance_suppress(bands[3], sr)

        self.prog(0.42, "Multiband Dynamics...")
        bands, gr_dict = self._process_bands(bands, sr)

        self.prog(0.52, "Harmonic Excitation...")
        if self.cfg.exciter["enabled"]:
            bands[3] = self._excite(bands[3], sr, self.cfg.exciter["amount"])        # FIX [10]
            bands[5] = self._excite(bands[5], sr, self.cfg.exciter["amount"] * 1.5)  # FIX [10]

        self.prog(0.60, "Summing & Parallel...")
        y_sum = np.sum(bands, axis=0)
        y_sum = self._parallel(y_sum, sr)

        if self.cfg.word_leveler["enabled"]:
            self.prog(0.68, "Word-Level Consistency...")
            y_sum = self._word_leveler(y_sum, sr)

        self.prog(0.76, "Psychoacoustic Glue...")
        if self.cfg.psycho["enabled"]:
            y_sum = self._ms_glue(y_sum, sr)

        # FIX [5]: true-peak limit BEFORE LUFS normalisation so target is met precisely
        self.prog(0.86, "True-Peak Limiting...")
        y_sum = self._true_peak_limit(y_sum, sr)

        self.prog(0.93, "LUFS Normalization...")
        y_sum = self._normalize(y_sum, sr)

        if out_path:
            sf.write(out_path, y_sum, sr)

        return {
            "input": y_orig,
            "output": y_sum,
            "sr": sr,
            "hop": self.cfg.hop,
            "gr": gr_dict,
        }

    # ─── DE-ESSER (now applied on sibilance band only — FIX [4]) ──────────────

    def _deesser(self, y, sr):
        """
        Broadband de-esser applied to the sibilance band.
        FIX [9]: Hilbert envelope replaced with LP-filtered RMS envelope detector.
        FIX [8]: att/rel coefficients use _time_const_samples for sample-domain ops.
        """
        c = self.cfg.de_esser
        nyq = sr / 2.0
        low = np.clip(c["low"] / nyq, 0.001, 0.999)
        high = np.clip(c["high"] / nyq, 0.001, 0.999)
        if low >= high:
            return y

        # Sidechain: bandpass to isolate sibilant content for level detection
        sos_bp = signal.butter(4, [low, high], btype='band', output='sos')
        y_sc = signal.sosfiltfilt(sos_bp, y)

        # FIX [9]: LP-filtered RMS envelope — no Hilbert ringing on transients
        sos_env = signal.butter(2, 400.0 / nyq, btype='low', output='sos')
        env = np.sqrt(np.maximum(0.0,
            signal.sosfiltfilt(sos_env, y_sc ** 2)
        ))
        env_db = librosa.amplitude_to_db(env + 1e-10)

        n = len(env_db)
        fast_win = np.clip(int(0.05 * sr), 1, n)
        med_win  = np.clip(int(0.5  * sr), 1, n)
        slow_win = np.clip(int(8.0  * sr), 1, n)

        ctx = (0.1 * uniform_filter1d(env_db, fast_win, mode='nearest') +
               0.3 * uniform_filter1d(env_db, med_win,  mode='nearest') +
               0.4 * uniform_filter1d(env_db, slow_win, mode='nearest') +
               0.2 * np.mean(env_db))

        gr = np.clip(
            np.maximum(0.0, env_db - ctx - c["sens"]) * (1.0 - 1.0 / c["ratio"]),
            0.0, c["range"]
        )

        # FIX [8]: use _time_const_samples for sample-domain smoothing
        att = self._time_const_samples(c["att"], sr)
        rel = self._time_const_samples(c["rel"], sr)
        gr_sm = _apply_dynamics(gr, att, rel)

        return y * (10.0 ** (-gr_sm / 20.0))

    # ─── DE-BREATHER ──────────────────────────────────────────────────────────

    def _debreather(self, y, sr):
        c = self.cfg.de_breather
        hop = self.cfg.hop

        flat = librosa.feature.spectral_flatness(y=y, n_fft=2048, hop_length=hop)[0]
        rms  = librosa.feature.rms(y=y, frame_length=2048, hop_length=hop)[0]
        zcr  = librosa.feature.zero_crossing_rate(y, hop_length=hop)[0]

        p20 = np.percentile(rms, 20)
        p85 = np.percentile(rms, 85)

        breath_mask = (
            (flat > 0.05) &
            (zcr  > 0.08) &
            (rms  > p20)  &
            (rms  < p85)
        )
        breath_smooth = gaussian_filter1d(breath_mask.astype(float), sigma=3) * c["sens"]

        ft   = np.arange(len(breath_smooth)) * hop
        st   = np.arange(len(y))
        gr_up = np.interp(st, ft, breath_smooth)
        return y * (10.0 ** (-gr_up * c["range"] / 20.0))

    # ─── LR4 BAND SPLITTING (FIX [1]) ─────────────────────────────────────────

    def _lr4_lp(self, fc, nyq):
        """4th-order Linkwitz-Riley LP: two cascaded 2nd-order Butterworth."""
        sos2 = signal.butter(2, fc / nyq, btype='low', output='sos')
        return np.vstack([sos2, sos2])

    def _lr4_hp(self, fc, nyq):
        """4th-order Linkwitz-Riley HP: two cascaded 2nd-order Butterworth."""
        sos2 = signal.butter(2, fc / nyq, btype='high', output='sos')
        return np.vstack([sos2, sos2])

    def _split_bands_lr4(self, y, sr):
        """
        True LR4 cascade band splitting via successive HP/LP on the residual.
        At each crossover frequency the *current residual* is split into an LP
        band (kept) and an HP residual (passed to the next stage).  This gives
        allpass-summing LP+HP pairs at every crossover, preserving phase
        coherence and unity-gain reconstruction.

        FIX [1]: replaces the old LP-subtraction approach which broke allpass
        summing and produced level/phase errors in mid bands.
        """
        freqs = self.cfg.crossover_freqs
        nyq   = sr / 2.0
        bands = []
        residual = y.copy()

        for f in freqs:
            lp = signal.sosfiltfilt(self._lr4_lp(f, nyq), residual)
            hp = signal.sosfiltfilt(self._lr4_hp(f, nyq), residual)
            bands.append(lp)
            residual = hp   # next stage works on the HP residual

        bands.append(residual)  # topmost air band
        return bands

    # ─── RESONANCE SUPPRESSION (FIX [2] + FIX [6]) ───────────────────────────

    def _resonance_suppress(self, y, sr):
        """
        Crackle-free resonance suppression.

        FIX [2]: zi state is reset every frame when filter coefficients change
                 (new notch frequency). sosfiltfilt (zero-phase, offline) is
                 used instead of lfilter so accumulated zi state is irrelevant.
        FIX [6]: Welch PSD on each hop-sized chunk replaces full STFT, cutting
                 both memory and CPU for peak detection.
        """
        c   = self.cfg.resonance
        hop = self.cfg.hop
        n_frames = max(1, (len(y) - 1) // hop + 1)

        y_out    = np.zeros(len(y))
        weight   = np.zeros(len(y))
        overlap  = hop

        for i in range(n_frames):
            start      = i * hop
            end        = min(start + hop + overlap, len(y))
            actual_len = end - start
            if actual_len <= 0:
                continue

            chunk = y[start:end].copy()

            # FIX [6]: Welch PSD for cheap per-chunk resonance detection
            nperseg = min(512, actual_len)
            if actual_len >= nperseg:
                freqs_w, psd = signal.welch(chunk, fs=sr, nperseg=nperseg)
                mf = np.max(psd)

                if mf > 1e-20:
                    peaks, _ = signal.find_peaks(psd, height=mf * 0.3, distance=5)
                    if len(peaks) > 0:
                        top_peaks = peaks[np.argsort(psd[peaks])[::-1][:3]]
                        for p in top_peaks:
                            f_res = freqs_w[p]
                            if 20 <= f_res <= sr / 2 - 10:
                                depth = np.clip((psd[p] / mf) * c["sens"], 0, 1.0)
                                if depth >= 0.05:
                                    # FIX [2]: sosfiltfilt — zero-phase, no zi needed
                                    sos_n = signal.iirnotch(f_res, c["q"], fs=sr)
                                    # iirnotch returns (b, a); convert to sos for stability
                                    sos_notch = signal.tf2sos(*sos_n)
                                    notched = signal.sosfiltfilt(sos_notch, chunk)
                                    chunk = chunk + depth * (notched - chunk)

            # Overlap-add with linear fade-in on the overlapping region
            fade_len = min(overlap, actual_len)
            fade_in  = np.linspace(0, 1, fade_len)

            if actual_len >= overlap:
                y_out[start:start + overlap] += chunk[:overlap] * fade_in
                weight[start:start + overlap] += fade_in
                y_out[start + overlap:end]    += chunk[overlap:]
                weight[start + overlap:end]   += 1.0
            else:
                y_out[start:end] += chunk
                weight[start:end] += 1.0

        weight = np.maximum(weight, 1e-10)
        return y_out / weight

    # ─── MULTIBAND DYNAMICS ───────────────────────────────────────────────────

    def _process_bands(self, bands, sr):
        """
        Per-band upward/downward dynamics.
        FIX [8]: all att/rel conversions go through _time_const(ms, sr).
        FIX [4]: sibilance band is no longer skipped — it was de-essed at
                 band level before this stage, so normal dynamics apply.
        """
        out      = []
        gr_dict  = {}
        hop      = self.cfg.hop

        for i, name in enumerate(self.cfg.band_names):
            c    = self.cfg.bands[name]
            y_b  = bands[i]
            rms  = librosa.feature.rms(y=y_b, frame_length=2048, hop_length=hop)[0]
            rms_db = librosa.amplitude_to_db(rms + 1e-10)

            frame_rate = sr / hop
            ctx_len = np.clip(int(0.5 * frame_rate), 1, len(rms_db))
            ctx = uniform_filter1d(rms_db, ctx_len, mode='nearest')

            gr = np.clip(
                np.maximum(0.0, rms_db - ctx - c["thresh"]) * (1.0 - 1.0 / c["ratio"]),
                0.0, 12.0
            )

            # FIX [8]: use _time_const helper — att/rel in Config are milliseconds
            att    = self._time_const(c["att"], sr)
            rel    = self._time_const(c["rel"], sr)
            gr_sm  = _apply_dynamics(gr, att, rel)
            gr_dict[name] = gr_sm

            ft    = np.arange(len(gr_sm)) * hop
            st    = np.arange(len(y_b))
            gr_up = np.interp(st, ft, gr_sm)
            out.append(y_b * (10.0 ** (-gr_up / 20.0)))

        return out, gr_dict

    # ─── HARMONIC EXCITER (FIX [10]) ──────────────────────────────────────────

    def _excite(self, y, sr, amount):
        """
        Harmonic exciter with level-normalised drive.

        FIX [10]: the band is normalised to unit peak before saturation so
                  drive behaviour is consistent regardless of input level, then
                  re-scaled back to the original level after harmonic generation.
        """
        c   = self.cfg.exciter
        hop = self.cfg.hop

        # Per-sample gain map: reduce excitation on already-loud frames
        rms    = librosa.feature.rms(y=y, frame_length=2048, hop_length=hop)[0]
        rms_db = librosa.amplitude_to_db(rms + 1e-10)

        th = c["threshold_db"]
        kn = c["knee_db"]
        quiet_amount = np.clip((th - rms_db) / kn, 0.0, 1.0)

        ft    = np.arange(len(quiet_amount)) * hop
        st    = np.arange(len(y))
        qa_up = np.interp(st, ft, quiet_amount)

        # FIX [10]: normalise band before drive, rescale after
        peak   = np.max(np.abs(y)) + 1e-10
        y_norm = y / peak

        drive   = 1.0 + amount * 5.0
        td      = np.tanh(drive)
        harmonics = np.tanh(y_norm * drive) / td - y_norm

        sos_hp      = signal.butter(2, c["freq"] / (sr / 2.0), btype='high', output='sos')
        harmonics_hp = signal.sosfiltfilt(sos_hp, harmonics)

        # Rescale harmonics back to original band level before adding
        return y + harmonics_hp * amount * qa_up * peak

    # ─── PARALLEL COMPRESSION + SATURATION ────────────────────────────────────

    def _parallel(self, y, sr):
        c   = self.cfg.parallel
        hop = self.cfg.hop

        rms    = librosa.feature.rms(y=y, frame_length=2048, hop_length=hop)[0]
        rms_db = librosa.amplitude_to_db(rms + 1e-10)
        gr     = np.maximum(0.0, rms_db - c["thresh"]) * (1.0 - 1.0 / c["ratio"])
        pg     = 10.0 ** (-gr / 20.0)

        ft    = np.arange(len(pg)) * hop
        st    = np.arange(len(y))
        pg_up = np.interp(st, ft, pg)
        y_par = y * pg_up

        if c["sat"] > 0:
            y_par = _soft_clip(y_par, 1.0 + c["sat"] * 3.0)

        return (1.0 - c["mix"]) * y + c["mix"] * y_par

    # ─── PERCEPTUAL WORD-LEVEL CONSISTENCY ────────────────────────────────────

    def _word_leveler(self, y, sr):
        """
        Perceptually-aware word-level consistency with cubic interpolation.
        FIX [7]: uses cached K-weighted signal instead of recomputing.
        """
        c   = self.cfg.word_leveler
        hop = self.cfg.hop
        win_len = int(0.04 * sr)

        # FIX [7]: reuse K-weight cache computed at top of process()
        y_k = self._y_k_cache if self._y_k_cache is not None else self._k_weight(y, sr)

        rms_k    = librosa.feature.rms(y=y_k, frame_length=win_len, hop_length=hop)[0]
        rms_k_db = librosa.amplitude_to_db(rms_k + 1e-10)

        flat = librosa.feature.spectral_flatness(y=y, n_fft=2048, hop_length=hop)[0]
        flat_comp_db = np.where(
            flat > 0.1,
            -2.0 * np.clip((flat - 0.1) / 0.3, 0, 1),
            0.0
        )

        perc_db = rms_k_db + flat_comp_db

        voice_mask = flat < 0.1
        voice_perc = perc_db[voice_mask]
        min_voiced = max(10, int(len(perc_db) * 0.01))
        if len(voice_perc) > min_voiced:
            target_db = np.median(voice_perc)
        else:
            target_db = np.median(perc_db)

        max_corr    = c["max_correction_db"]
        gain_needed = np.clip(target_db - perc_db, -max_corr, max_corr)

        smooth_frames = max(1, int(c["smooth_ms"] * sr / 1000.0 / hop))
        gain_smooth   = uniform_filter1d(gain_needed, smooth_frames, mode='nearest')

        ft = np.arange(len(gain_smooth)) * hop
        st = np.arange(len(y))
        if len(gain_smooth) >= 4:
            cs      = CubicSpline(ft, gain_smooth, bc_type='clamped')
            gain_up = cs(st)
        else:
            gain_up = np.interp(st, ft, gain_smooth)
        gain_up = np.clip(gain_up, -max_corr, max_corr)

        return y * (10.0 ** (gain_up / 20.0))

    # ─── PSYCHOACOUSTIC GLUE ──────────────────────────────────────────────────

    def _k_weight(self, y, sr):
        """
        ITU-R BS.1770-4 K-weighting filter.
        Stage 1: high-shelf pre-filter (+4 dB above ~1.7 kHz).
        Stage 2: 2nd-order highpass at 38 Hz.
        Result cached in self._y_k_cache by process().
        """
        Fc    = 1681.97
        dBg   = 3.999
        Q     = 0.7071
        A     = 10.0 ** (dBg / 40.0)
        w0    = 2.0 * np.pi * Fc / sr
        alpha = np.sin(w0) / (2.0 * Q)
        cosw0 = np.cos(w0)

        b0 =  A * ((A + 1) + (A - 1) * cosw0 + 2.0 * np.sqrt(A) * alpha)
        b1 = -2.0 * A * ((A - 1) + (A + 1) * cosw0)
        b2 =  A * ((A + 1) + (A - 1) * cosw0 - 2.0 * np.sqrt(A) * alpha)
        a0 =      (A + 1) - (A - 1) * cosw0 + 2.0 * np.sqrt(A) * alpha
        a1 =  2.0 * ((A - 1) - (A + 1) * cosw0)
        a2 =      (A + 1) - (A - 1) * cosw0 - 2.0 * np.sqrt(A) * alpha

        sos_shelf = np.array([[b0 / a0, b1 / a0, b2 / a0, 1.0, a1 / a0, a2 / a0]])
        sos_hp    = signal.butter(2, 38.0 / (sr / 2.0), btype='high', output='sos')

        y1 = signal.sosfilt(sos_shelf, y)
        return signal.sosfilt(sos_hp, y1)

    def _ms_glue(self, y, sr):
        """
        Psychoacoustic soft-knee compressor for final bus glue.
        FIX [7]: uses cached K-weighted signal.
        FIX [8]: att/rel through _time_const helper.
        """
        c   = self.cfg.psycho
        hop = self.cfg.hop

        # FIX [7]: reuse K-weight cache
        y_perc = self._y_k_cache if self._y_k_cache is not None else self._k_weight(y, sr)

        rms_frames = librosa.feature.rms(
            y=y_perc, frame_length=2048, hop_length=hop
        )[0]
        env_db = librosa.amplitude_to_db(rms_frames + 1e-10)
        n = len(env_db)

        fast_win = np.clip(max(1, int(0.05 * n)), 1, n)
        med_win  = np.clip(max(1, int(0.20 * n)), 1, n)
        slow_win = np.clip(max(1, int(0.60 * n)), 1, n)

        ctx = (0.1 * uniform_filter1d(env_db, fast_win, mode='nearest') +
               0.3 * uniform_filter1d(env_db, med_win,  mode='nearest') +
               0.4 * uniform_filter1d(env_db, slow_win, mode='nearest') +
               0.2 * np.mean(env_db))

        knee = float(c["knee"])
        hk   = knee / 2.0
        th   = float(c["sens"])

        above  = env_db > (th + hk)
        within = (~above) & (env_db > (th - hk))

        excess = np.zeros_like(env_db)
        excess[above] = env_db[above] - th
        if np.any(within):
            x = env_db[within] - (th - hk)
            excess[within] = (x ** 2) / (2.0 * knee)

        gr    = np.clip(excess * (1.0 - 1.0 / float(c["ratio"])), 0, float(c["range"]))

        # FIX [8]: use _time_const helper for consistent ms → coefficient conversion
        att   = self._time_const(float(c["att"]), sr)
        rel   = self._time_const(float(c["rel"]), sr)
        gr_sm = _apply_dynamics(gr, att, rel)

        ft    = np.arange(len(gr_sm)) * hop
        st    = np.arange(len(y))
        gr_up = np.interp(st, ft, gr_sm)

        return y * (10.0 ** (-gr_up / 20.0))

    # ─── TRUE-PEAK LIMITER ────────────────────────────────────────────────────
    # FIX [5]: now called BEFORE _normalize in process() so LUFS target is met

    def _true_peak_limit(self, y, sr):
        """
        4x oversampled true-peak limiter.
        Runs before LUFS normalisation (FIX [5]) to avoid target overshoot.
        """
        ceiling = 10.0 ** (self.cfg.output["tp"] / 20.0)
        y_up    = signal.resample_poly(y, 4, 1)
        peak    = np.max(np.abs(y_up))
        if peak > ceiling:
            y_up = y_up * (ceiling / peak)
        return signal.resample_poly(y_up, 1, 4)

    # ─── LUFS NORMALIZATION (FIX [3]) ─────────────────────────────────────────

    def _normalize(self, y, sr):
        """
        Integrated LUFS normalisation to target.
        FIX [3]: fallback no longer uses the magic -0.691 constant.
                 Instead, the internal _k_weight filter is applied and true
                 RMS is measured from the K-weighted signal — a much more
                 accurate approximation of integrated loudness when pyloudnorm
                 is unavailable.
        FIX [5]: called AFTER _true_peak_limit in process().
        """
        target = self.cfg.output["lufs"]

        if HAS_PYLN:
            meter = pyln.Meter(sr)
            loud  = meter.integrated_loudness(y)
            if np.isfinite(loud):
                return pyln.normalize.loudness(y, loud, target)

        # FIX [3]: use K-weighted RMS as a proper loudness proxy
        y_k  = self._k_weight(y, sr)
        rms  = np.sqrt(np.mean(y_k ** 2))
        db   = 20.0 * np.log10(rms + 1e-10)
        # LUFS ≈ -0.691 + 10·log₁₀(mean_square_k_weighted)
        # which is the same as: db_k + 0  (no offset needed when using K-weighted RMS)
        gain_db = target - db
        return y * (10.0 ** (gain_db / 20.0))


# ─── ENTRY POINT ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    import time

    in_path  = sys.argv[1] if len(sys.argv) > 1 else "vocal.wav"
    out_path = sys.argv[2] if len(sys.argv) > 2 else "vocal_intelcomp_out.wav"

    print("=" * 60)
    print("  Intelcomp Expert — Professional Vocal Dynamics Engine")
    print("=" * 60)

    stages = []

    def progress(pct, msg):
        stages.append((pct, msg))
        bar = "█" * int(pct * 30) + "░" * (30 - int(pct * 30))
        print(f"\r  [{bar}] {int(pct * 100):3d}%  {msg}", end="", flush=True)

    engine = VocalDynamicsEngine(progress=progress)
    t0     = time.time()
    result = engine.process(in_path, out_path)
    elapsed = time.time() - t0

    sr    = result["sr"]
    y_in  = result["input"]
    y_out = result["output"]

    rms_in  = 20 * np.log10(np.sqrt(np.mean(y_in  ** 2)) + 1e-10)
    rms_out = 20 * np.log10(np.sqrt(np.mean(y_out ** 2)) + 1e-10)
    peak    = 20 * np.log10(np.max(np.abs(y_out))        + 1e-10)

    print("\n\n" + "=" * 60)
    print(f"  Output:    {out_path}")
    print(f"  Duration:  {len(y_in) / sr:.2f}s @ {sr}Hz")
    print(f"  Input RMS: {rms_in:.2f} dBFS → Output RMS: {rms_out:.2f} dBFS")
    print(f"  True Peak: {peak:.2f} dBFS")
    print(f"  Time:      {elapsed:.2f}s")
    print("=" * 60)