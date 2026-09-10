import numpy as np


def rmse(x):
    return float(np.sqrt(np.mean(np.asarray(x) ** 2)))


def compute_metrics(pred, target, eps_relative_physical=1e-19):
    """
    Physical-domain metrics.
    eps_relative_physical=1e-19 corresponds to eps=1 in the scaled domain
    when scale=1e19.
    """
    pred = np.asarray(pred, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)

    err = pred - target

    overall_rmse = rmse(err)
    mae = float(np.mean(np.abs(err)))

    signal_power = float(np.sum(target ** 2))
    noise_power = float(np.sum(err ** 2))
    snr = np.inf if noise_power == 0 else float(
        10.0 * np.log10(signal_power / noise_power)
    )

    denom = np.where(
        np.abs(target) < eps_relative_physical,
        eps_relative_physical,
        np.abs(target),
    )
    mean_relative = float(np.mean(np.abs(err) / denom))

    return {
        "RMSE": overall_rmse,
        "MAE": mae,
        "SNR_dB": snr,
        "MeanRelative": mean_relative,
        "Early2_RMSE": rmse(err[:, 0:2]),
        "Middle2_29_RMSE": rmse(err[:, 2:30]),
        "Late10_RMSE": rmse(err[:, 30:40]),
        "Zero_Count": int(np.count_nonzero(pred == 0.0)),
    }


def per_channel_rmse(pred, target):
    pred = np.asarray(pred, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    return np.sqrt(np.mean((pred - target) ** 2, axis=0))
