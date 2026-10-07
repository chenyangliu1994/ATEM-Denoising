# Data

The public release uses five source data files.

```text
data/
├─ clean_responses/
│  └─ clean_responses_ablation_320000x40.dat
└─ measured_noise/
   ├─ E1_shielded_16stack.dat
   ├─ E2_shielded_single.dat
   ├─ E3_urban_roadside_16stack.dat
   └─ E4_urban_roadside_single.dat
```

The clean-response file contains 320,000 rows and 40 ATEM channels.

Each measured-noise file contains 80 columns:

```text
time_1 ... time_40 noise_1 ... noise_40
```

Expected row counts are:

| File | Rows | Columns |
|---|---:|---:|
| clean_responses_ablation_320000x40.dat | 320000 | 40 |
| E1_shielded_16stack.dat | 14171 | 80 |
| E2_shielded_single.dat | 2883 | 80 |
| E3_urban_roadside_16stack.dat | 62539 | 80 |
| E4_urban_roadside_single.dat | 44847 | 80 |

Keep the measured-noise rows in their original acquisition order.

Generated train/validation/test sets, WGAN-augmented files, prediction files,
model checkpoints, and intermediate arrays are not part of the public source
data. They are regenerated from the files above.

Run:

```bash
python data/verify_public_data.py
```

after placing the data files.
