# Public data placement

Before committing the repository, place the five source data files below.

```text
ATEM-Denoising/
└─ data/
   ├─ clean_responses/
   │  └─ clean_responses_ablation_320000x40.dat
   └─ measured_noise/
      ├─ E1_shielded_16stack.dat
      ├─ E2_shielded_single.dat
      ├─ E3_urban_roadside_16stack.dat
      └─ E4_urban_roadside_single.dat
```

Use only one clean-response file. If an older local copy is named `dianya.dat`,
rename the intended public copy to:

`clean_responses_ablation_320000x40.dat`

Do not upload generated train/validation/test files, prediction files,
checkpoints, or intermediate arrays. Those are regenerated from the source
data and scripts.

## Expected data dimensions

- clean responses: 320000 x 40
- E1 measured noise: 14171 x 80
- E2 measured noise: 2883 x 80
- E3 measured noise: 62539 x 80
- E4 measured noise: 44847 x 80

The measured-noise files must remain in their original acquisition order.

## Git LFS

The repository already tracks `.dat` files with Git LFS through
`.gitattributes`. Initialize Git LFS in GitHub Desktop before committing the
data files.
