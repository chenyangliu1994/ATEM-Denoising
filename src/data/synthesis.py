import numpy as np


def synthesize_pairs(clean, noise, scale_divisor=20.0, reference_channel_1based=22):
    """
    Exact synthesis rule used in the controlled ablation:

        x_i = s_i + (n_i / scale_divisor) * s_ref
        y_i = s_i

    where the reference channel is 1-based in manuscript notation.
    """
    clean = np.asarray(clean, dtype=np.float64)
    noise = np.asarray(noise, dtype=np.float64)

    if clean.shape != noise.shape:
        raise ValueError(f"clean shape {clean.shape} != noise shape {noise.shape}")
    if clean.ndim != 2 or clean.shape[1] != 40:
        raise ValueError("Expected N x 40 clean/noise matrices.")

    ref_idx = reference_channel_1based - 1
    ref = clean[:, ref_idx:ref_idx + 1]

    scaled_noise = (noise / scale_divisor) * ref
    noisy = clean + scaled_noise
    supervised = np.hstack([noisy, clean])
    return supervised, scaled_noise
