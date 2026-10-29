"""Elementwise SAMBAR math. Uses all parameters; no subset selection."""

from typing import Any

import jax
import jax.numpy as jnp


def weighted_z(theta_pre, theta_ft, omega_old, omega_new, beta: float, eps: float) -> Any:
    """Selected weighted_taco_ar.py anchor: epsilon weights θ_ft in both parts.

    Where both importance estimates vanish, the anchor equals θ_ft.
    """
    return jax.tree.map(
        lambda tp, tf, oo, on: (oo * tp + (beta * on + eps) * tf) / (oo + beta * on + eps),
        theta_pre,
        theta_ft,
        omega_old,
        omega_new,
    )


def penalty(theta, u, z, rho):
    residuals = jax.tree.map(lambda t, z_leaf: t - z_leaf, theta, z)
    linear = sum(
        jnp.sum(u_leaf * r_leaf) for u_leaf, r_leaf in zip(jax.tree.leaves(u), jax.tree.leaves(residuals), strict=True)
    )
    quad = 0.5 * rho * sum(jnp.sum(r**2) for r in jax.tree.leaves(residuals))
    return linear + quad


def dual_norm(u):
    return jnp.sqrt(sum(jnp.sum(leaf**2) for leaf in jax.tree.leaves(u)))


def update_history(old, current, old_count: int, new_count: int):
    """Task-count-weighted mean of cached stage estimates (not recomputed Fisher)."""
    if old_count < 1 or new_count < 1:
        raise ValueError("Importance task counts must be positive")
    return jax.tree.map(lambda a, b: (old_count * a + new_count * b) / (old_count + new_count), old, current)
