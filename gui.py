"""Intelcomp Pro VDE - GUI Application (Standard Tkinter)"""

import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import threading
import numpy as np
import librosa
import soundfile as sf
from pathlib import Path
import matplotlib
matplotlib.use('TkAgg')
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from engine import VocalDynamicsEngine, Config
from presets import PRESETS, get_preset, list_presets


class WaveformDisplay(tk.Frame):
    """Waveform visualization widget"""
    
    def __init__(self, master, **kwargs):
        super().__init__(master, **kwargs)
        
        self.figure = Figure(figsize=(8, 2.5), dpi=100, facecolor='#2b2b2b')
        self.ax = self.figure.add_subplot(111)
        self.ax.set_facecolor('#1e1e1e')
        self.ax.tick_params(colors='white', labelsize=8)
        for spine in self.ax.spines.values():
            spine.set_color('white')
        
        self.canvas = FigureCanvasTkAgg(self.figure, self)
        self.canvas.get_tk_widget().pack(fill='both', expand=True)
        
        self.dry_audio = None
        self.wet_audio = None
        self.sr = None
        self.show_wet = True
    
    def set_audio(self, dry, wet, sr):
        self.dry_audio = dry
        self.wet_audio = wet
        self.sr = sr
        self._redraw()
    
    def toggle_view(self, show_wet):
        self.show_wet = show_wet
        self._redraw()
    
    def _redraw(self):
        self.ax.clear()
        self.ax.set_facecolor('#1e1e1e')
        
        if self.dry_audio is None:
            self.ax.text(0.5, 0.5, "No audio loaded", 
                        ha='center', va='center', color='gray',
                        transform=self.ax.transAxes)
            self.canvas.draw()
            return
        
        audio = self.wet_audio if (self.show_wet and self.wet_audio is not None) else self.dry_audio
        time = np.arange(len(audio)) / self.sr
        
        # Downsample for display
        max_points = 5000
        if len(audio) > max_points:
            step = len(audio) // max_points
            audio_disp = audio[::step]
            time_disp = time[::step]
        else:
            audio_disp = audio
            time_disp = time
        
        color = '#4a9eff' if self.show_wet else '#888888'
        self.ax.plot(time_disp, audio_disp, color=color, linewidth=0.5)
        self.ax.set_xlabel('Time (s)', color='white', fontsize=9)
        self.ax.set_ylabel('Amplitude', color='white', fontsize=9)
        self.ax.set_xlim(0, time[-1])
        self.ax.set_ylim(-1.05, 1.05)
        self.ax.grid(True, alpha=0.2, color='white')
        
        label = "PROCESSED" if self.show_wet else "ORIGINAL"
        self.ax.set_title(label, color=color, fontsize=10, fontweight='bold')
        
        self.figure.tight_layout()
        self.canvas.draw()


class GainReductionDisplay(tk.Frame):
    """Gain reduction visualization"""
    
    def __init__(self, master, **kwargs):
        super().__init__(master, **kwargs)
        
        self.figure = Figure(figsize=(8, 2.5), dpi=100, facecolor='#2b2b2b')
        self.ax = self.figure.add_subplot(111)
        self.ax.set_facecolor('#1e1e1e')
        self.ax.tick_params(colors='white', labelsize=8)
        for spine in self.ax.spines.values():
            spine.set_color('white')
        
        self.canvas = FigureCanvasTkAgg(self.figure, self)
        self.canvas.get_tk_widget().pack(fill='both', expand=True)
    
    def set_data(self, gain_reductions, sr, hop_length):
        self.ax.clear()
        self.ax.set_facecolor('#1e1e1e')
        
        colors = {
            'sub_mud': '#ff6b6b',
            'low_mud': '#ffa500',
            'boxiness': '#ffd93d',
            'presence': '#6bcf7f',
            'sibilance': '#4a9eff',
            'air': '#a78bfa',
        }
        
        for name, gr in gain_reductions.items():
            time = np.arange(len(gr)) * hop_length / sr
            self.ax.plot(time, -gr, label=name, color=colors.get(name, 'white'), 
                        linewidth=1, alpha=0.8)
        
        self.ax.set_xlabel('Time (s)', color='white', fontsize=9)
        self.ax.set_ylabel('Gain Reduction (dB)', color='white', fontsize=9)
        self.ax.set_title('Gain Reduction by Band', color='white', fontsize=10, fontweight='bold')
        self.ax.legend(loc='upper right', fontsize=7, facecolor='#2b2b2b', 
                      labelcolor='white', ncol=3)
        self.ax.grid(True, alpha=0.2, color='white')
        self.ax.invert_yaxis()
        
        self.figure.tight_layout()
        self.canvas.draw()


class IntelcompProVDE(tk.Tk):
    def __init__(self):
        super().__init__()
        
        self.title("Intelcomp Pro VDE - Vocal Dynamics Engine")
        self.geometry("1400x900")
        self.minsize(1200, 800)
        
        self.configure(bg='#1e1e1e')
        
        self.input_path = None
        self.output_path = None
        self.engine = None
        self.current_config = Config()
        self.is_processing = False
        
        self._build_ui()
    
    def _build_ui(self):
        # Main grid
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(1, weight=1)
        
        # Left panel - Controls
        self.left_panel = tk.Frame(self, bg='#2b2b2b', width=320)
        self.left_panel.grid(row=0, column=0, rowspan=2, sticky="nsew", padx=5, pady=5)
        self.left_panel.grid_propagate(False)
        self._build_left_panel()
        
        # Center - Visualizations
        self.center_panel = tk.Frame(self, bg='#1e1e1e')
        self.center_panel.grid(row=0, column=1, rowspan=2, sticky="nsew", padx=5, pady=5)
        self.center_panel.grid_columnconfigure(0, weight=1)
        self.center_panel.grid_rowconfigure(1, weight=1)
        self.center_panel.grid_rowconfigure(3, weight=1)
        self._build_center_panel()
        
        # Bottom - Status
        self.status_bar = tk.Frame(self, bg='#2b2b2b', height=40)
        self.status_bar.grid(row=2, column=0, columnspan=2, sticky="ew", padx=5, pady=5)
        self.status_bar.grid_propagate(False)
        self._build_status_bar()
    
    def _build_left_panel(self):
        panel = self.left_panel
        
        # Create scrollable frame
        canvas = tk.Canvas(panel, bg='#2b2b2b', highlightthickness=0)
        scrollbar = ttk.Scrollbar(panel, orient="vertical", command=canvas.yview)
        scrollable_frame = tk.Frame(canvas, bg='#2b2b2b')
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Title
        title = tk.Label(scrollable_frame, text="INTELCOMP PRO VDE", 
                        font=("Arial", 18, "bold"), bg='#2b2b2b', fg='white')
        title.pack(pady=(10, 5))
        
        subtitle = tk.Label(scrollable_frame, text="Vocal Dynamics Engine", 
                           font=("Arial", 12), bg='#2b2b2b', fg='#aaaaaa')
        subtitle.pack(pady=(0, 15))
        
        # File controls
        file_frame = tk.LabelFrame(scrollable_frame, text="INPUT FILE", 
                                   bg='#2b2b2b', fg='white', font=("Arial", 10, "bold"))
        file_frame.pack(fill='x', padx=10, pady=5)
        
        self.input_label = tk.Label(file_frame, text="No file selected", 
                                   bg='#2b2b2b', fg='#888888')
        self.input_label.pack(anchor='w', padx=10, pady=5)
        
        btn_frame = tk.Frame(file_frame, bg='#2b2b2b')
        btn_frame.pack(fill='x', padx=10, pady=(0, 10))
        tk.Button(btn_frame, text="Load Audio", command=self._load_audio, 
                 width=15, bg='#4a9eff', fg='white').pack(side='left', padx=(0, 5))
        tk.Button(btn_frame, text="Save As...", command=self._save_as, 
                 width=15, bg='#4a9eff', fg='white').pack(side='left')
        
        # Preset selector
        preset_frame = tk.LabelFrame(scrollable_frame, text="PRESET", 
                                     bg='#2b2b2b', fg='white', font=("Arial", 10, "bold"))
        preset_frame.pack(fill='x', padx=10, pady=5)
        
        self.preset_var = tk.StringVar(value="Pop Vocal")
        preset_menu = ttk.Combobox(preset_frame, textvariable=self.preset_var,
                                   values=list_presets(), state='readonly')
        preset_menu.pack(fill='x', padx=10, pady=10)
        preset_menu.bind("<<ComboboxSelected>>", self._on_preset_change)
        
        # Dynamics parameters
        dyn_frame = tk.LabelFrame(scrollable_frame, text="DYNAMICS", 
                                  bg='#2b2b2b', fg='white', font=("Arial", 10, "bold"))
        dyn_frame.pack(fill='x', padx=10, pady=5)
        
        self.lookahead_slider = self._make_slider(dyn_frame, "Lookahead (ms)", 0, 20, 5, self._on_param_change)
        self.knee_slider = self._make_slider(dyn_frame, "Knee Width (dB)", 0, 12, 4, self._on_param_change)
        self.transient_slider = self._make_slider(dyn_frame, "Transient Preserve", 0, 1, 0.3, self._on_param_change)
        
        # Parallel compression
        par_frame = tk.LabelFrame(scrollable_frame, text="PARALLEL COMPRESSION", 
                                  bg='#2b2b2b', fg='white', font=("Arial", 10, "bold"))
        par_frame.pack(fill='x', padx=10, pady=5)
        
        self.par_mix_slider = self._make_slider(par_frame, "Mix (%)", 0, 100, 25, self._on_param_change)
        self.par_sat_slider = self._make_slider(par_frame, "Saturation", 0, 1, 0.15, self._on_param_change)
        
        # Output
        out_frame = tk.LabelFrame(scrollable_frame, text="OUTPUT", 
                                  bg='#2b2b2b', fg='white', font=("Arial", 10, "bold"))
        out_frame.pack(fill='x', padx=10, pady=5)
        
        self.loudness_slider = self._make_slider(out_frame, "Target LUFS", -20, -8, -14, self._on_param_change)
        
        # Process button
        self.process_btn = tk.Button(scrollable_frame, text="PROCESS", 
                                     font=("Arial", 16, "bold"),
                                     height=2, command=self._process,
                                     bg='#4a9eff', fg='white')
        self.process_btn.pack(fill='x', padx=10, pady=20)
        
        # A/B Toggle
        ab_frame = tk.LabelFrame(scrollable_frame, text="A/B COMPARISON", 
                                 bg='#2b2b2b', fg='white', font=("Arial", 10, "bold"))
        ab_frame.pack(fill='x', padx=10, pady=5)
        
        self.ab_var = tk.StringVar(value="Processed")
        tk.Radiobutton(ab_frame, text="Original", variable=self.ab_var, 
                      value="Original", command=self._on_ab_change,
                      bg='#2b2b2b', fg='white', selectcolor='#3b3b3b').pack(anchor='w', padx=10, pady=2)
        tk.Radiobutton(ab_frame, text="Processed", variable=self.ab_var, 
                      value="Processed", command=self._on_ab_change,
                      bg='#2b2b2b', fg='white', selectcolor='#3b3b3b').pack(anchor='w', padx=10, pady=(2, 10))
    
    def _make_slider(self, parent, label, from_, to, default, command):
        frame = tk.Frame(parent, bg='#2b2b2b')
        frame.pack(fill='x', padx=10, pady=5)
        
        header = tk.Frame(frame, bg='#2b2b2b')
        header.pack(fill='x')
        tk.Label(header, text=label, font=("Arial", 9), bg='#2b2b2b', fg='white').pack(side='left')
        value_label = tk.Label(header, text=f"{default:.2f}", font=("Arial", 9), bg='#2b2b2b', fg='white')
        value_label.pack(side='right')
        
        slider = tk.Scale(frame, from_=from_, to=to, orient='horizontal',
                         resolution=0.1, command=lambda v: self._slider_changed(v, value_label, command),
                         bg='#2b2b2b', fg='white', troughcolor='#3b3b3b', highlightthickness=0)
        slider.set(default)
        slider.pack(fill='x', pady=(2, 0))
        
        return slider
    
    def _slider_changed(self, value, label, command):
        label.configure(text=f"{float(value):.2f}")
        command()
    
    def _build_center_panel(self):
        panel = self.center_panel
        
        # Waveform
        tk.Label(panel, text="WAVEFORM", font=("Arial", 10, "bold"), 
                bg='#1e1e1e', fg='white').grid(row=0, column=0, sticky='w', padx=10, pady=(10, 0))
        self.waveform_display = WaveformDisplay(panel, bg='#1e1e1e')
        self.waveform_display.grid(row=1, column=0, sticky='nsew', padx=10, pady=5)
        
        # Gain reduction
        tk.Label(panel, text="GAIN REDUCTION", font=("Arial", 10, "bold"), 
                bg='#1e1e1e', fg='white').grid(row=2, column=0, sticky='w', padx=10, pady=(10, 0))
        self.gr_display = GainReductionDisplay(panel, bg='#1e1e1e')
        self.gr_display.grid(row=3, column=0, sticky='nsew', padx=10, pady=(5, 10))
    
    def _build_status_bar(self):
        self.status_label = tk.Label(self.status_bar, text="Ready", 
                                    font=("Arial", 11), bg='#2b2b2b', fg='white')
        self.status_label.pack(side='left', padx=10)
        
        self.loudness_label = tk.Label(self.status_bar, text="", 
                                      font=("Arial", 11), bg='#2b2b2b', fg='white')
        self.loudness_label.pack(side='right', padx=10)
        
        self.progress = ttk.Progressbar(self.status_bar, length=300, mode='determinate')
        self.progress.pack(side='right', padx=10)
    
    def _load_audio(self):
        path = filedialog.askopenfilename(
            title="Select Audio File",
            filetypes=[("Audio Files", "*.wav *.mp3 *.flac *.ogg"), ("All Files", "*.*")]
        )
        if path:
            self.input_path = path
            self.input_label.configure(text=Path(path).name, fg='white')
            self.status_label.configure(text=f"Loaded: {Path(path).name}")
    
    def _save_as(self):
        path = filedialog.asksaveasfilename(
            title="Save Processed Audio",
            defaultextension=".wav",
            filetypes=[("WAV Files", "*.wav")]
        )
        if path:
            self.output_path = path
    
    def _on_preset_change(self, event=None):
        preset = get_preset(self.preset_var.get())
        self.current_config = Config(preset_dict=preset)
        self._update_sliders_from_config()
        self.status_label.configure(text=f"Preset: {self.preset_var.get()}")
    
    def _update_sliders_from_config(self):
        cfg = self.current_config
        self.lookahead_slider.set(cfg.dynamics["lookahead_ms"])
        self.knee_slider.set(cfg.dynamics["knee_width"])
        self.transient_slider.set(cfg.dynamics["transient_preserve"])
        self.par_mix_slider.set(cfg.parallel["mix"] * 100)
        self.par_sat_slider.set(cfg.parallel["saturation"])
        self.loudness_slider.set(cfg.output["target_loudness_lufs"])
    
    def _on_param_change(self):
        cfg = self.current_config
        cfg.dynamics["lookahead_ms"] = float(self.lookahead_slider.get())
        cfg.dynamics["knee_width"] = float(self.knee_slider.get())
        cfg.dynamics["transient_preserve"] = float(self.transient_slider.get())
        cfg.parallel["mix"] = float(self.par_mix_slider.get()) / 100
        cfg.parallel["saturation"] = float(self.par_sat_slider.get())
        cfg.output["target_loudness_lufs"] = float(self.loudness_slider.get())
    
    def _on_ab_change(self):
        show_wet = (self.ab_var.get() == "Processed")
        self.waveform_display.toggle_view(show_wet)
    
    def _process(self):
        if not self.input_path:
            messagebox.showerror("Error", "Please load an audio file first.")
            return
        
        if self.is_processing:
            return
        
        self.is_processing = True
        self.process_btn.configure(state='disabled', text="PROCESSING...")
        self.progress['value'] = 0
        
        def progress_callback(progress, message):
            self.after(0, lambda: self._update_progress(progress, message))
        
        def worker():
            try:
                engine = VocalDynamicsEngine(self.current_config, progress_callback)
                result = engine.process(self.input_path, self.output_path)
                self.after(0, lambda: self._processing_complete(result))
            except Exception as e:
                self.after(0, lambda: self._processing_error(str(e)))
        
        thread = threading.Thread(target=worker, daemon=True)
        thread.start()
    
    def _update_progress(self, progress, message):
        self.progress['value'] = progress * 100
        self.status_label.configure(text=message)
    
    def _processing_complete(self, result):
        self.is_processing = False
        self.process_btn.configure(state='normal', text="PROCESS")
        self.progress['value'] = 100
        
        # Update displays
        self.waveform_display.set_audio(result["input"], result["output"], result["sr"])
        self.gr_display.set_data(result["gain_reductions"], result["sr"], result["hop_length"])
        
        # Loudness stats
        engine = VocalDynamicsEngine(self.current_config)
        stats = engine.get_loudness_stats(result["output"], result["sr"])
        self.loudness_label.configure(text=f"Output: {stats['integrated_lufs']:.1f} LUFS")
        
        self.status_label.configure(text="Processing complete!")
        messagebox.showinfo("Success", f"Processing complete!\n\nOutput: {stats['integrated_lufs']:.1f} LUFS")
    
    def _processing_error(self, error_msg):
        self.is_processing = False
        self.process_btn.configure(state='normal', text="PROCESS")
        self.progress['value'] = 0
        self.status_label.configure(text="Error")
        messagebox.showerror("Processing Error", f"An error occurred:\n\n{error_msg}")


def main():
    app = IntelcompProVDE()
    app.mainloop()


if __name__ == "__main__":
    main()