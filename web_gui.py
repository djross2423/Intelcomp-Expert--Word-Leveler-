"""Intelcomp Expert - Web-Based GUI"""

from flask import Flask, render_template, request, send_file, jsonify
import os
import tempfile
from pathlib import Path
import base64
import io
import numpy as np
import librosa
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from engine import VocalDynamicsEngine, Config
from presets import PRESETS, get_preset, list_presets

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB max

UPLOAD_FOLDER = tempfile.mkdtemp()
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


# ==========================================
# ROUTES
# ==========================================

@app.route('/')
def index():
    """Main page"""
    return render_template('index.html',
                         presets=list_presets(),
                         presets_data=PRESETS)


@app.route('/api/presets')
def get_presets():
    """Return all preset data as JSON"""
    return jsonify(PRESETS)


@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    input_path = os.path.join(app.config['UPLOAD_FOLDER'], 'input.wav')
    file.save(input_path)

    y, sr = librosa.load(input_path, sr=None)
    duration = len(y) / sr

    return jsonify({
        'success': True,
        'filename': file.filename,
        'duration': f"{duration:.2f}s",
        'sample_rate': sr
    })


@app.route('/process', methods=['POST'])
def process_audio():
    """Process audio with current settings"""
    input_path = os.path.join(app.config['UPLOAD_FOLDER'], 'input.wav')

    if not os.path.exists(input_path):
        return jsonify({'error': 'No audio file uploaded'}), 400

    try:
        data = request.json

        # Load preset
        preset_name = data.get('preset', 'Pop Vocal')
        preset = get_preset(preset_name)
        config = Config(preset_dict=preset)

        # Pre-Processing Overrides
        if 'deesser_enabled' in data:
            config.de_esser['enabled'] = bool(data['deesser_enabled'])
        if 'deesser_sens' in data:
            config.de_esser['sens'] = float(data['deesser_sens'])
        if 'debreather_enabled' in data:
            config.de_breather['enabled'] = bool(data['debreather_enabled'])
        if 'resonance_enabled' in data:
            config.resonance['enabled'] = bool(data['resonance_enabled'])
        if 'resonance_sens' in data:
            config.resonance['sens'] = float(data['resonance_sens'])

        # Tonal Enhancement Overrides
        if 'exciter_enabled' in data:
            config.exciter['enabled'] = bool(data['exciter_enabled'])
        if 'exciter_amount' in data:
            config.exciter['amount'] = float(data['exciter_amount'])

        # Psychoacoustic Glue Overrides
        if 'psycho_enabled' in data:
            config.psycho['enabled'] = bool(data['psycho_enabled'])
        if 'psycho_sens' in data:
            config.psycho['sens'] = float(data['psycho_sens'])
        if 'psycho_ratio' in data:
            config.psycho['ratio'] = float(data['psycho_ratio'])
        if 'psycho_mask' in data:
            config.psycho['mask'] = float(data['psycho_mask'])

        # Output Overrides
        if 'lufs' in data:
            config.output['lufs'] = float(data['lufs'])
        if 'tp' in data:
            config.output['tp'] = float(data['tp'])

        # Process
        output_path = os.path.join(app.config['UPLOAD_FOLDER'], 'output.wav')

        def progress_callback(p, msg):
            pass

        engine = VocalDynamicsEngine(config, progress_callback)
        result = engine.process(input_path, output_path)

        # Generate visualizations
        waveform_img = generate_waveform(result['input'], result['output'], result['sr'])
        gr_img = generate_gain_reduction(result['gr'], result['sr'], result['hop'])

        # Get loudness stats
        meter_lufs = config.output['lufs']
        loudness_str = f"Output: {meter_lufs:.1f} LUFS"

        return jsonify({
            'success': True,
            'waveform': waveform_img,
            'gain_reduction': gr_img,
            'loudness': loudness_str,
            'message': 'Expert processing complete!'
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/download')
def download_file():
    """Download processed audio"""
    output_path = os.path.join(app.config['UPLOAD_FOLDER'], 'output.wav')

    if not os.path.exists(output_path):
        return jsonify({'error': 'No processed file available'}), 404

    return send_file(output_path, as_attachment=True, download_name='intelcomp_expert_output.wav')


# ==========================================
# VISUALIZATION HELPERS
# ==========================================

def generate_waveform(dry, wet, sr):
    """Generate waveform comparison image"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 4), dpi=100)
    fig.patch.set_facecolor('#1e1e1e')

    time_dry = np.arange(len(dry)) / sr
    max_points = 3000
    if len(dry) > max_points:
        step = len(dry) // max_points
        ax1.plot(time_dry[::step], dry[::step], color='#888888', linewidth=0.5)
    else:
        ax1.plot(time_dry, dry, color='#888888', linewidth=0.5)
    ax1.set_facecolor('#2b2b2b')
    ax1.set_title('ORIGINAL', color='#888888', fontsize=10, fontweight='bold')
    ax1.tick_params(colors='white', labelsize=8)
    ax1.set_ylim(-1.05, 1.05)
    for spine in ax1.spines.values():
        spine.set_color('white')

    time_wet = np.arange(len(wet)) / sr
    if len(wet) > max_points:
        step = len(wet) // max_points
        ax2.plot(time_wet[::step], wet[::step], color='#ff9f43', linewidth=0.5)
    else:
        ax2.plot(time_wet, wet, color='#ff9f43', linewidth=0.5)
    ax2.set_facecolor('#2b2b2b')
    ax2.set_title('EXPERT PROCESSED', color='#ff9f43', fontsize=10, fontweight='bold')
    ax2.set_xlabel('Time (s)', color='white', fontsize=9)
    ax2.tick_params(colors='white', labelsize=8)
    ax2.set_ylim(-1.05, 1.05)
    for spine in ax2.spines.values():
        spine.set_color('white')

    plt.tight_layout()

    img_buffer = io.BytesIO()
    plt.savefig(img_buffer, format='png', facecolor=fig.get_facecolor(), dpi=100)
    img_buffer.seek(0)
    img_base64 = base64.b64encode(img_buffer.getvalue()).decode()
    plt.close()

    return f"data:image/png;base64,{img_base64}"


def generate_gain_reduction(gain_reductions, sr, hop_length):
    """Generate gain reduction visualization"""
    fig, ax = plt.subplots(figsize=(10, 3), dpi=100)
    fig.patch.set_facecolor('#1e1e1e')
    ax.set_facecolor('#2b2b2b')

    colors = {
        'sub_mud': '#ff6b6b',
        'low_mud': '#ffa500',
        'boxiness': '#ffd93d',
        'presence': '#6bcf7f',
        'sibilance': '#4a9eff',
        'air': '#a78bfa',
    }

    if not gain_reductions:
        ax.text(0.5, 0.5, "Gain reduction data not available in Expert mode",
                ha='center', va='center', color='gray', transform=ax.transAxes)
    else:
        for name, gr in gain_reductions.items():
            time = np.arange(len(gr)) * hop_length / sr
            ax.plot(time, -gr, label=name, color=colors.get(name, 'white'),
                    linewidth=1, alpha=0.8)

        ax.set_xlabel('Time (s)', color='white', fontsize=9)
        ax.set_ylabel('Gain Reduction (dB)', color='white', fontsize=9)
        ax.set_title('Multiband Gain Reduction', color='#ff9f43', fontsize=10, fontweight='bold')
        ax.legend(loc='upper right', fontsize=7, facecolor='#2b2b2b',
                  labelcolor='white', ncol=3)
        ax.grid(True, alpha=0.2, color='white')
        ax.invert_yaxis()
        ax.tick_params(colors='white', labelsize=8)
        for spine in ax.spines.values():
            spine.set_color('white')

    plt.tight_layout()

    img_buffer = io.BytesIO()
    plt.savefig(img_buffer, format='png', facecolor=fig.get_facecolor(), dpi=100)
    img_buffer.seek(0)
    img_base64 = base64.b64encode(img_buffer.getvalue()).decode()
    plt.close()

    return f"data:image/png;base64,{img_base64}"


# ==========================================
# SERVER STARTUP
# ==========================================

if __name__ == '__main__':
    print("=" * 60)
    print("INTELCOMP EXPERT - WEB GUI")
    print("=" * 60)
    print("\nStarting server...")
    print("Open your browser and go to: http://127.0.0.1:5000")
    print("\nPress Ctrl+C to stop the server")
    print("=" * 60)
    app.run(debug=False, host='127.0.0.1', port=5000)