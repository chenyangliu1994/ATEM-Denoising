# Public data layout

Place the exact data used for the denoising experiments in this directory.

```text
data/
├── clean_responses/
│   └── clean_responses_ablation_320000x40.dat
└── measured_noise/
    ├── E1_shielded_16stack.dat
    ├── E2_shielded_single.dat
    ├── E3_urban_roadside_16stack.dat
    └── E4_urban_roadside_single.dat
```

## Clean responses

`clean_responses_ablation_320000x40.dat`

- 320,000 rows
- 40 ATEM time channels per row
- physical voltage values
- exact clean-response list used for the current controlled A/B ablation

## Measured noise

Each measured-noise file contains 80 columns:

```text
time_1 ... time_40 noise_1 ... noise_40
```

Rows must remain in original acquisition order.

The urban-roadside datasets were collected in the actual local electromagnetic
environment. An operating wireless router was present nearby. This condition
is reported for reproducibility and should not be interpreted as proof that
the router was the dominant noise source.
