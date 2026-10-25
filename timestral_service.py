"""Local Timestral inference helpers for shadow time-series forecasting.

This module is deliberately separate from the chat-model provider registry. It
returns numerical forecasts only and does not create trade decisions.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_MODEL_ID = "mistralai/Timestral-209M"
DEFAULT_CACHE_DIR = Path(__file__).resolve().parent / "models" / "timestral_cache"


@lru_cache(maxsize=2)
def _load_forecaster(model_id: str, device: str, cache_dir: str):
    try:
        from timestral import TimestralForecaster
    except ImportError as exc:
        raise RuntimeError(
            "Timestral is not installed. Install the project requirements first."
        ) from exc

    return TimestralForecaster.from_pretrained(
        model_id,
        device=device,
        cache_dir=cache_dir,
    )


def forecast_series(
    context: Any,
    prediction_length: int = 20,
    *,
    model_id: str = DEFAULT_MODEL_ID,
    device: str = "cpu",
    cache_dir: str | Path = DEFAULT_CACHE_DIR,
) -> dict:
    """Forecast one or more aligned series and return quantiles as NumPy arrays.

    Input shapes accepted by Timestral are `(T,)`, `(B, T)`, and `(B, V, T)`.
    For multi-asset forecasts, pass aligned log-return series as `(1, V, T)`.
    """
    values = np.asarray(context, dtype=np.float32)
    if values.ndim not in (1, 2, 3):
        raise ValueError("context must have shape (T,), (B,T), or (B,V,T)")
    if prediction_length < 1:
        raise ValueError("prediction_length must be positive")
    if values.shape[-1] < 2:
        raise ValueError("context must contain at least two time steps")
    if np.isinf(values).any():
        raise ValueError("context cannot contain infinite values")

    resolved_cache = str(Path(cache_dir).expanduser().resolve())
    forecaster = _load_forecaster(model_id, device, resolved_cache)
    result = forecaster.forecast(values, prediction_length=prediction_length)
    quantiles = result.quantiles
    if hasattr(quantiles, "detach"):
        quantiles = quantiles.detach().cpu().numpy()

    return {
        "model_id": model_id,
        "device": device,
        "prediction_length": prediction_length,
        "quantile_levels": tuple(float(q) for q in result.quantile_levels),
        "quantiles": np.asarray(quantiles),
    }

