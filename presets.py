PRESETS = {
    "Pop Vocal": {
        "bands": {'sub_mud': {"thresh": 4.0, "ratio": 3.0, "att": 1, "rel": 5}, 'low_mud': {"thresh": 5.0, "ratio": 2.5, "att": 1, "rel": 10}, 'boxiness': {"thresh": 4.5, "ratio": 2.0, "att": 2, "rel": 15}, 'presence': {"thresh": 3.5, "ratio": 1.5, "att": 3, "rel": 20}, 'sibilance': {"thresh": 3.0, "ratio": 2.0, "att": 1, "rel": 4}, 'air': {"thresh": 5.0, "ratio": 1.5, "att": 4, "rel": 30}},
        "de_esser": {"enabled": True, "low": 5000, "high": 8000, "sens": 2.0, "ratio": 4.0, "att": 1.0, "rel": 50.0, "range": 12.0},
        "de_breather": {"enabled": True, "sens": 0.4, "range": 6.0},
        "resonance": {"enabled": True, "sens": 0.5, "range": 6.0, "q": 15.0},
        "exciter": {"enabled": True, "amount": 0.15, "freq": 6000},
        "psycho": {"enabled": True, "sens": 2.0, "ratio": 2.0, "att": 10.0, "rel": 150.0, "range": 8.0, "mask": 0.5, "knee": 3.0, "ms_width": 1.2},
        "intelligence": {"mfcc": 0.15, "cent": 0.35}, "dynamics": {"lookahead": 5.0, "knee": 4.0, "transient": 0.3},
        "parallel": {"thresh": -20.0, "ratio": 4.0, "mix": 0.25, "sat": 0.15}, "output": {"lufs": -14.0, "tp": -1.0}
    },
    "Rock Vocal": {
        "bands": {'sub_mud': {"thresh": 3.5, "ratio": 4.0, "att": 1, "rel": 5}, 'low_mud': {"thresh": 4.5, "ratio": 3.0, "att": 1, "rel": 8}, 'boxiness': {"thresh": 4.0, "ratio": 2.5, "att": 2, "rel": 12}, 'presence': {"thresh": 3.0, "ratio": 2.0, "att": 2, "rel": 15}, 'sibilance': {"thresh": 2.5, "ratio": 2.0, "att": 1, "rel": 3}, 'air': {"thresh": 4.5, "ratio": 2.0, "att": 3, "rel": 25}},
        "de_esser": {"enabled": True, "low": 5000, "high": 8000, "sens": 1.5, "ratio": 6.0, "att": 0.5, "rel": 40.0, "range": 15.0},
        "de_breather": {"enabled": True, "sens": 0.3, "range": 8.0},
        "resonance": {"enabled": True, "sens": 0.7, "range": 10.0, "q": 20.0},
        "exciter": {"enabled": True, "amount": 0.25, "freq": 5500},
        "psycho": {"enabled": True, "sens": 2.5, "ratio": 3.0, "att": 8.0, "rel": 120.0, "range": 10.0, "mask": 0.7, "knee": 2.0, "ms_width": 1.0},
        "intelligence": {"mfcc": 0.2, "cent": 0.3}, "dynamics": {"lookahead": 3.0, "knee": 2.0, "transient": 0.5},
        "parallel": {"thresh": -18.0, "ratio": 6.0, "mix": 0.35, "sat": 0.35}, "output": {"lufs": -12.0, "tp": -0.5}
    },
    "Ballad Vocal": {
        "bands": {'sub_mud': {"thresh": 5.0, "ratio": 2.0, "att": 2, "rel": 8}, 'low_mud': {"thresh": 6.0, "ratio": 2.0, "att": 2, "rel": 15}, 'boxiness': {"thresh": 5.5, "ratio": 1.5, "att": 3, "rel": 20}, 'presence': {"thresh": 4.0, "ratio": 1.3, "att": 4, "rel": 25}, 'sibilance': {"thresh": 3.5, "ratio": 1.5, "att": 2, "rel": 6}, 'air': {"thresh": 6.0, "ratio": 1.3, "att": 5, "rel": 40}},
        "de_esser": {"enabled": True, "low": 5000, "high": 8000, "sens": 3.0, "ratio": 3.0, "att": 2.0, "rel": 60.0, "range": 10.0},
        "de_breather": {"enabled": False, "sens": 0.2, "range": 4.0},
        "resonance": {"enabled": True, "sens": 0.3, "range": 4.0, "q": 10.0},
        "exciter": {"enabled": True, "amount": 0.05, "freq": 7000},
        "psycho": {"enabled": True, "sens": 1.5, "ratio": 1.5, "att": 20.0, "rel": 250.0, "range": 6.0, "mask": 0.3, "knee": 5.0, "ms_width": 1.3},
        "intelligence": {"mfcc": 0.1, "cent": 0.2}, "dynamics": {"lookahead": 8.0, "knee": 6.0, "transient": 0.2},
        "parallel": {"thresh": -22.0, "ratio": 3.0, "mix": 0.15, "sat": 0.08}, "output": {"lufs": -16.0, "tp": -1.0}
    },
    "Rap Vocal": {
        "bands": {'sub_mud': {"thresh": 3.0, "ratio": 4.0, "att": 1, "rel": 4}, 'low_mud': {"thresh": 4.0, "ratio": 3.5, "att": 1, "rel": 8}, 'boxiness': {"thresh": 3.5, "ratio": 2.5, "att": 2, "rel": 12}, 'presence': {"thresh": 2.5, "ratio": 2.5, "att": 2, "rel": 15}, 'sibilance': {"thresh": 2.0, "ratio": 2.0, "att": 1, "rel": 3}, 'air': {"thresh": 4.0, "ratio": 2.0, "att": 3, "rel": 20}},
        "de_esser": {"enabled": True, "low": 5000, "high": 8000, "sens": 1.0, "ratio": 8.0, "att": 0.5, "rel": 30.0, "range": 18.0},
        "de_breather": {"enabled": True, "sens": 0.6, "range": 10.0},
        "resonance": {"enabled": True, "sens": 0.8, "range": 12.0, "q": 25.0},
        "exciter": {"enabled": True, "amount": 0.2, "freq": 5000},
        "psycho": {"enabled": True, "sens": 3.0, "ratio": 4.0, "att": 5.0, "rel": 80.0, "range": 12.0, "mask": 0.8, "knee": 1.5, "ms_width": 0.9},
        "intelligence": {"mfcc": 0.2, "cent": 0.4}, "dynamics": {"lookahead": 2.0, "knee": 1.5, "transient": 0.6},
        "parallel": {"thresh": -16.0, "ratio": 8.0, "mix": 0.4, "sat": 0.25}, "output": {"lufs": -11.0, "tp": -0.3}
    },
    "Podcast / Voice": {
        "bands": {'sub_mud': {"thresh": 3.5, "ratio": 4.0, "att": 1, "rel": 5}, 'low_mud': {"thresh": 4.5, "ratio": 3.0, "att": 1, "rel": 10}, 'boxiness': {"thresh": 4.0, "ratio": 2.5, "att": 2, "rel": 15}, 'presence': {"thresh": 3.0, "ratio": 2.0, "att": 3, "rel": 20}, 'sibilance': {"thresh": 3.0, "ratio": 2.0, "att": 1, "rel": 5}, 'air': {"thresh": 5.0, "ratio": 1.5, "att": 4, "rel": 30}},
        "de_esser": {"enabled": True, "low": 5000, "high": 8000, "sens": 2.0, "ratio": 4.0, "att": 1.0, "rel": 50.0, "range": 12.0},
        "de_breather": {"enabled": True, "sens": 0.5, "range": 8.0},
        "resonance": {"enabled": True, "sens": 0.6, "range": 8.0, "q": 15.0},
        "exciter": {"enabled": False, "amount": 0.1, "freq": 6000},
        "psycho": {"enabled": True, "sens": 1.5, "ratio": 1.8, "att": 15.0, "rel": 200.0, "range": 6.0, "mask": 0.4, "knee": 4.0, "ms_width": 1.1},
        "intelligence": {"mfcc": 0.1, "cent": 0.25}, "dynamics": {"lookahead": 5.0, "knee": 4.0, "transient": 0.2},
        "parallel": {"thresh": -18.0, "ratio": 4.0, "mix": 0.2, "sat": 0.05}, "output": {"lufs": -16.0, "tp": -1.0}
    }
}
def get_preset(name): return PRESETS.get(name, PRESETS["Pop Vocal"])
def list_presets(): return list(PRESETS.keys())