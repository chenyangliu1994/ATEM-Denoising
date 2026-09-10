from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
PY = sys.executable

STEPS = [
    "01_split_measured_noise.py",
    "02_train_wasserstein_generator.py",
    "03_check_generated_noise.py",
    "04_build_ablation_dataset.py",
]

for scenario in ["E1", "E2", "E3", "E4"]:
    print("\n" + "=" * 90)
    print("SCENARIO", scenario)
    print("=" * 90)

    for step in STEPS:
        subprocess.run(
            [PY, str(HERE / step), "--scenario", scenario],
            check=True,
        )

    subprocess.run(
        [PY, str(HERE / "05_train_denoiser.py"), "--scenario", scenario, "--group", "A"],
        check=True,
    )
    subprocess.run(
        [PY, str(HERE / "05_train_denoiser.py"), "--scenario", scenario, "--group", "B"],
        check=True,
    )
    subprocess.run(
        [PY, str(HERE / "06_evaluate_common_test.py"), "--scenario", scenario],
        check=True,
    )

print("\nAll four scenarios finished.")
