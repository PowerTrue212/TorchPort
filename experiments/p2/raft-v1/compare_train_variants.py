"""Measure official five-run spread and custom-vs-official distance."""
import itertools
import json
from pathlib import Path

import numpy as np

root = Path("/root/autodl-tmp/RAFT-Benchmark-v1/evidence")
official = [root / f"train-variant-official-r{i}" for i in range(1, 12)]
custom = root / "train-variant-custom-r1"
n = len(official)
results = [json.loads((path / "result.json").read_text()) for path in (*official, custom)]
report = {"status": "pilot_only", "loss": [], "states": [], "heldout_epe": {}}
for step in range(3):
    values = np.array([item["steps"][step]["loss"] for item in results])
    pairwise = [abs(values[i] - values[j]) for i, j in itertools.combinations(range(n), 2)]
    report["loss"].append({"step": step, "official_values": values[:n].tolist(),
                            "official_pair_max": max(pairwise),
                            "custom": float(values[n]),
                            "custom_to_official_max": float(max(abs(values[n] - values[:n])))})
    states = [np.load(path / f"state-step-{step}.npz") for path in (*official, custom)]
    assert all(set(state.files) == set(states[0].files) for state in states)
    worst_official = ("", 0.0)
    worst_custom = ("", 0.0)
    beyond = []
    for key in states[0].files:
        arrays = [state[key].astype(np.float64) for state in states]
        spread = max(float(np.max(np.abs(arrays[i] - arrays[j])))
                     for i, j in itertools.combinations(range(n), 2))
        distance = max(float(np.max(np.abs(arrays[n] - arrays[i]))) for i in range(n))
        if spread > worst_official[1]:
            worst_official = (key, spread)
        if distance > worst_custom[1]:
            worst_custom = (key, distance)
        if distance > spread + 1e-12:
            beyond.append({"key": key, "official_spread": spread, "custom_distance": distance})
    for state in states:
        state.close()
    report["states"].append({"step": step, "official_max": worst_official,
                             "custom_max": worst_custom,
                             "tensors_beyond_official_spread": len(beyond),
                             "top_beyond": sorted(beyond, key=lambda x: x["custom_distance"] - x["official_spread"], reverse=True)[:5]})
if all((path / "heldout.json").is_file() for path in (*official, custom)):
    metrics = [json.loads((path / "heldout.json").read_text()) for path in (*official, custom)]
    for sample in metrics[0]["heldout_epe"]:
        values = np.array([item["heldout_epe"][sample] for item in metrics])
        report["heldout_epe"][sample] = {"official_values": values[:n].tolist(),
                                         "official_pair_max": float(max(abs(values[i] - values[j])
                                         for i, j in itertools.combinations(range(n), 2))),
                                         "custom": float(values[n]),
                                         "custom_to_official_max": float(max(abs(values[n] - values[:n])))}
out = root / "train-variant-comparison-r1.json"
out.write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2))
