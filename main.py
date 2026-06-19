"""Intelcomp Expert - Main Entry Point"""

import argparse
import sys
from pathlib import Path

from engine import VocalDynamicsEngine, Config
from presets import PRESETS, get_preset, list_presets


def cli_mode(args):
    """Run in command-line mode"""
    print("=" * 60)
    print("INTELCOMP EXPERT - CLI MODE")
    print("=" * 60)

    if not Path(args.input).exists():
        print(f"Error: Input file not found: {args.input}")
        sys.exit(1)

    # Load config
    if args.preset:
        if args.preset not in PRESETS:
            print(f"Error: Unknown preset '{args.preset}'")
            print(f"Available presets: {', '.join(list_presets())}")
            sys.exit(1)
        preset = get_preset(args.preset)
        config = Config(preset_dict=preset)
        print(f"Preset: {args.preset}")
    elif args.config:
        config = Config.load_yaml(args.config)
        print(f"Config: {args.config}")
    else:
        config = Config()
        print("Preset: Pop Vocal (Default)")

    # Output path
    output = args.output or str(Path(args.input).stem) + "_expert.wav"

    # Progress callback
    def progress(p, msg):
        bar_len = 30
        filled = int(bar_len * p)
        bar = '█' * filled + '░' * (bar_len - filled)
        print(f"\r[{bar}] {p*100:5.1f}% - {msg}", end='', flush=True)

    print(f"\nInput: {args.input}")
    print(f"Output: {output}")
    print()

    # Process
    engine = VocalDynamicsEngine(config, progress)
    result = engine.process(args.input, output)

    print("\n")
    print(f"Duration: {len(result['output']) / result['sr']:.2f}s")
    print(f"Target LUFS: {config.output['lufs']:.1f}")
    print(f"True Peak Ceiling: {config.output['tp']:.1f} dBTP")
    print(f"Saved to: {output}")
    print("=" * 60)


def web_gui_mode():
    """Run web GUI mode"""
    from web_gui import app
    print("=" * 60)
    print("INTELCOMP EXPERT - WEB GUI")
    print("=" * 60)
    print("\nStarting server...")
    print("Open your browser and go to: http://127.0.0.1:5000")
    print("\nPress Ctrl+C to stop the server")
    print("=" * 60)
    app.run(debug=False, host='127.0.0.1', port=5000)


def main():
    parser = argparse.ArgumentParser(
        description="Intelcomp Expert - Professional Vocal Dynamics Engine",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --gui                    # Launch web GUI
  python main.py vocal.wav                # Process with default settings
  python main.py vocal.wav --preset rock  # Process with Rock preset
  python main.py vocal.wav --config my.yaml -o out.wav
        """
    )

    parser.add_argument('input', nargs='?', help='Input audio file (omit to launch GUI)')
    parser.add_argument('-o', '--output', help='Output file path')
    parser.add_argument('-p', '--preset', choices=list_presets(), help='Processing preset')
    parser.add_argument('-c', '--config', help='YAML config file')
    parser.add_argument('--gui', action='store_true', help='Launch web GUI')
    parser.add_argument('--list-presets', action='store_true', help='List available presets')

    args = parser.parse_args()

    if args.list_presets:
        print("Available presets:")
        for name in list_presets():
            desc = PRESETS[name].get('description', '')
            print(f"  - {name}: {desc}")
        return

    if args.gui or not args.input:
        web_gui_mode()
    else:
        cli_mode(args)


if __name__ == "__main__":
    main()