"""v2.3 placement scores; these are not calibrated match predictions."""
from __future__ import annotations

import numpy as np

STRENGTH = 0.50
Z_LIMIT = 2.0


def placement_scores(probabilities: np.ndarray, strength: float = STRENGTH):
    """Adjust D relative to the strongest side; preserve the original H:A ratio.

    The 13-match population itself provides the mean and standard deviation.
    H/D/A ticket counts continue to come exclusively from original probabilities.
    """
    p = np.asarray(probabilities, dtype=float)
    if p.shape != (13, 3) or not np.isfinite(p).all() or (p <= 0).any() or not np.allclose(p.sum(axis=1), 1):
        raise ValueError("Expected 13 positive normalized H/D/A probability rows")
    if not np.isfinite(strength) or strength < 0:
        raise ValueError("strength must be finite and nonnegative")
    relative = np.log(p[:, 1] / np.maximum(p[:, 0], p[:, 2]))
    deviation = float(relative.std())
    z = (relative - relative.mean()) / deviation if deviation > 1e-12 else np.zeros(13)
    z = np.clip(z, -Z_LIMIT, Z_LIMIT)
    logit_d = np.log(p[:, 1] / (1 - p[:, 1]))
    q_d = 1 / (1 + np.exp(-(logit_d + strength * z)))
    q = np.empty_like(p)
    q[:, 1] = q_d
    q[:, [0, 2]] = (1 - q_d)[:, None] * p[:, [0, 2]] / (1 - p[:, 1])[:, None]
    return q, z, relative
