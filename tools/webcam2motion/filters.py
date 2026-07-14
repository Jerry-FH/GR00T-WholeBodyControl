"""Temporal filters for live pose streaming.

OneEuro on the 63 body_pose channels + normalized-lerp lowpass on the root
quaternion, applied BEFORE the FK in smpl_adapter so smpl_joints inherit the
smoothing consistently.
"""

import math

import numpy as np


class OneEuro:
    """One Euro filter (Casiez et al.) over an arbitrary-shape float array."""

    def __init__(self, freq: float = 30.0, min_cutoff: float = 1.0,
                 beta: float = 0.1, d_cutoff: float = 1.0):
        self.freq = freq
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self._x_prev = None
        self._dx_prev = None
        self._t_prev = None

    @staticmethod
    def _alpha(cutoff: float, dt: float) -> float:
        tau = 1.0 / (2 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def __call__(self, x: np.ndarray, t: float) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        if self._x_prev is None:
            self._x_prev = x
            self._dx_prev = np.zeros_like(x)
            self._t_prev = t
            return x
        dt = max(t - self._t_prev, 1e-6)
        self._t_prev = t

        dx = (x - self._x_prev) / dt
        a_d = self._alpha(self.d_cutoff, dt)
        dx_hat = a_d * dx + (1 - a_d) * self._dx_prev

        cutoff = self.min_cutoff + self.beta * np.abs(dx_hat)
        a = 1.0 / (1.0 + (1.0 / (2 * math.pi * cutoff)) / dt)
        x_hat = a * x + (1 - a) * self._x_prev

        self._x_prev = x_hat
        self._dx_prev = dx_hat
        return x_hat

    def reset(self):
        self._x_prev = self._dx_prev = self._t_prev = None


class QuatLowpass:
    """Normalized-lerp lowpass on a (4,) quaternion (hemisphere-corrected)."""

    def __init__(self, alpha: float = 0.4):
        self.alpha = alpha
        self._q = None

    def __call__(self, q: np.ndarray) -> np.ndarray:
        q = np.asarray(q, dtype=np.float64)
        q = q / (np.linalg.norm(q) + 1e-12)
        if self._q is None:
            self._q = q
            return q
        if np.dot(self._q, q) < 0:
            q = -q
        out = (1 - self.alpha) * self._q + self.alpha * q
        out /= np.linalg.norm(out) + 1e-12
        self._q = out
        return out

    def reset(self):
        self._q = None


class DeltaClamp:
    """Per-tick max-delta clamp (rad) as a last-resort spike guard."""

    def __init__(self, max_delta: float = 0.3):
        self.max_delta = max_delta
        self._prev = None

    def __call__(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        if self._prev is None:
            self._prev = x
            return x
        out = self._prev + np.clip(x - self._prev, -self.max_delta, self.max_delta)
        self._prev = out
        return out

    def reset(self):
        self._prev = None
