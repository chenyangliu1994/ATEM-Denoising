from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent

expected = {
    ROOT / "clean_responses" / "clean_responses_ablation_320000x40.dat": (320000, 40),
    ROOT / "measured_noise" / "E1_shielded_16stack.dat": (14171, 80),
    ROOT / "measured_noise" / "E2_shielded_single.dat": (2883, 80),
    ROOT / "measured_noise" / "E3_urban_roadside_16stack.dat": (62539, 80),
    ROOT / "measured_noise" / "E4_urban_roadside_single.dat": (44847, 80),
}

ok = True
for path, shape in expected.items():
    if not path.exists():
        print(f"MISSING  {path.relative_to(ROOT)}")
        ok = False
        continue
    arr = np.loadtxt(path)
    if arr.shape == shape:
        print(f"OK       {path.relative_to(ROOT)}  {arr.shape}")
    else:
        print(f"BAD      {path.relative_to(ROOT)}  got={arr.shape}, expected={shape}")
        ok = False

print("\nPASS" if ok else "\nCHECK FAILED")
