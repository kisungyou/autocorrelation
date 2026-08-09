"""Public routines for Riemannian Moran and Geary diagnostics.

This is the small computational surface used by the public notebooks for
*One Moran, Three Geary Measures*.  It implements the manuscript definitions,
the sphere, affine-invariant SPD, and torus geometries used in the examples,
portable data loaders, shared plotting settings, and deterministic self-checks.

The implementation never silently recenters logarithmic residuals.  A valid
Fréchet center must satisfy its first-order condition up to numerical
tolerance; otherwise tangent-based inference is not run.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import os
from pathlib import Path
import tempfile
from typing import Callable, Iterable

import numpy as np
import pandas as pd


TAU = 2.0 * np.pi
MODULE_DIR = Path(__file__).resolve().parent
DATA_DIR = MODULE_DIR / "data"
TEXT_WIDTH_IN = 6.5

# Keep small numerical examples stable and avoid a large worker pool.
for _name in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_name, "1")
os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / "autocorr-public-matplotlib-cache"),
)

INK = "#1F2933"
MUTED = "#8A8F95"
GRID = "#D9DEE3"
PAPER = "#FFFFFF"
BLUE = "#0072B2"
ORANGE = "#D55E00"
GOLD = "#E69F00"
GREEN = "#009E73"
PURPLE = "#7B3294"
VERMILLION = "#D55E00"
SKY = "#56B4E9"
MORAN_COLORS = {
    "not significant": MUTED,
    "aligned": BLUE,
    "anti-aligned": ORANGE,
}
GEARY_COLORS = {
    "not significant": MUTED,
    "homogeneous": GREEN,
    "transition / outlier": PURPLE,
}

__all__ = [
    "GlobalResult",
    "global_measures",
    "global_randomization",
    "local_diagnostics",
    "validate_weights",
    "center_residual_ratio",
    "pairwise_sqeuclidean",
    "sym_knn",
    "rook_weights",
    "bh_adjust",
    "wrap_angle",
    "intrinsic_circle_mean",
    "intrinsic_torus_mean",
    "sphere_log",
    "sphere_exp",
    "sphere_mean",
    "sphere_components",
    "airm_mean",
    "spd_components",
    "torus_components",
    "euclidean_components",
    "scalar_components",
    "load_chicago",
    "load_california",
    "build_swiss_phases",
    "summarize_local",
    "jaccard",
    "classification_columns",
    "set_notebook_style",
    "panel_label",
    "map_axis",
    "draw_graph",
    "run_self_checks",
]


def wrap_angle(x: np.ndarray | float) -> np.ndarray | float:
    """Map angular differences to [-pi, pi)."""
    return (np.asarray(x) + np.pi) % TAU - np.pi


def pairwise_sqeuclidean(U: np.ndarray) -> np.ndarray:
    q = np.sum(U * U, axis=1)
    D = q[:, None] + q[None, :] - 2.0 * U @ U.T
    return np.maximum(D, 0.0)



def sym_knn(coords: np.ndarray, k: int) -> np.ndarray:
    """Symmetrized binary k-nearest-neighbor weights."""
    from scipy.spatial import cKDTree

    coords = np.asarray(coords, dtype=float)
    n = len(coords)
    tree = cKDTree(coords)
    _, inds = tree.query(coords, k=min(k + 1, n))
    W = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in np.atleast_1d(inds[i])[1:]:
            W[i, int(j)] = 1.0
            W[int(j), i] = 1.0
    np.fill_diagonal(W, 0.0)
    return W

def bh_adjust(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    n = len(p)
    order = np.argsort(p)
    ranked = p[order]
    q = ranked * n / np.arange(1, n + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    q = np.minimum(q, 1.0)
    out = np.empty(n)
    out[order] = q
    return out


def validate_weights(W: np.ndarray, *, require_symmetric: bool = True) -> np.ndarray:
    W = np.asarray(W, dtype=float)
    if W.ndim != 2 or W.shape[0] != W.shape[1]:
        raise ValueError("W must be a square matrix")
    if np.any(~np.isfinite(W)) or np.any(W < 0):
        raise ValueError("W must contain finite, nonnegative weights")
    if not np.allclose(np.diag(W), 0.0):
        raise ValueError("W must have a zero diagonal")
    if require_symmetric and not np.allclose(W, W.T, atol=1e-12, rtol=1e-12):
        raise ValueError("The manuscript's inferential protocol requires symmetric W")
    if W.sum() <= 0:
        raise ValueError("W has no positive spatial links")
    return W


def center_residual_ratio(U: np.ndarray) -> float:
    """Numerical diagnostic ||sum_i u_i|| / sqrt(sum_i ||u_i||^2)."""
    U = np.asarray(U, dtype=float)
    V = float(np.sum(U * U))
    if V <= 0:
        return 0.0
    return float(np.linalg.norm(U.sum(axis=0)) / math.sqrt(V))


def assert_frechet_first_order(U: np.ndarray, *, tol: float = 1e-8) -> float:
    ratio = center_residual_ratio(U)
    if ratio > tol:
        raise ValueError(
            "Logarithmic residuals fail the Fréchet first-order diagnostic: "
            f"||sum u_i||/sqrt(V)={ratio:.3e} > {tol:.1e}. "
            "Do not repair this by post-hoc Euclidean recentering; recompute or "
            "redefine the Fréchet mean."
        )
    return ratio


@dataclass(frozen=True)
class GlobalResult:
    I: float
    C_F: float
    C_P: float
    C_T: float
    kappa: float
    CP_over_CT: float
    L: float
    center_residual_ratio: float
    p_I_upper: float
    p_CP_lower: float
    permutations: int


def global_measures(
    U: np.ndarray,
    D2: np.ndarray,
    W: np.ndarray,
    *,
    V_mu: float | None = None,
    center_tol: float = 1e-8,
) -> dict[str, float]:
    """Compute one Moran and three Geary measures.

    U contains logarithmic residual coordinates in one orthonormal basis of
    T_mu M.  D2 contains squared intrinsic geodesic distances.  Under a valid
    minimizing logarithm, V_mu=sum_i d_g^2(Y_i,mu)=sum_i ||U_i||^2 exactly;
    the explicit argument is retained as a numerical consistency check.
    """
    W = validate_weights(W, require_symmetric=True)
    U = np.asarray(U, dtype=float)
    D2 = np.asarray(D2, dtype=float)
    n = len(U)
    if D2.shape != (n, n):
        raise ValueError("D2 has incompatible shape")
    ratio = assert_frechet_first_order(U, tol=center_tol)
    S0 = float(W.sum())
    V = float(np.sum(U * U))
    if V <= 0:
        raise ValueError("All manifold observations are identical at the chosen center")
    if V_mu is None:
        V_mu = V
    V_mu = float(V_mu)
    if not np.isclose(V_mu, V, rtol=1e-8, atol=1e-10 * max(1.0, V)):
        raise ValueError(
            "Inconsistent mean dispersion: sum d_g^2(Y_i,mu) must equal "
            "sum ||Log_mu(Y_i)||^2 when the same center and minimizing logs are used"
        )
    T2 = pairwise_sqeuclidean(U)
    gram = U @ U.T
    edge_geo = float(np.sum(W * D2))
    edge_tan = float(np.sum(W * T2))
    all_geo = float(np.sum(D2))
    if all_geo <= 0:
        raise ValueError("All intrinsic pairwise distances are zero")
    I = n / S0 * float(np.sum(W * gram)) / V
    C_F = (n - 1) / (2.0 * S0) * edge_geo / V_mu
    C_P = n * (n - 1) / S0 * edge_geo / all_geo
    C_T = (n - 1) / (2.0 * S0) * edge_tan / V
    omega = W.sum(axis=1)
    omega_bar = S0 / n
    L = n / (S0 * V) * float(
        np.sum((omega - omega_bar) * np.sum(U * U, axis=1))
    )
    return {
        "I": I,
        "C_F": C_F,
        "C_P": C_P,
        "C_T": C_T,
        "kappa": C_F / C_P,
        "CP_over_CT": C_P / C_T,
        "L": L,
        "center_residual_ratio": ratio,
    }


def global_randomization(
    U: np.ndarray,
    D2: np.ndarray,
    W: np.ndarray,
    *,
    V_mu: float | None = None,
    permutations: int = 999,
    seed: int = 0,
) -> GlobalResult:
    obs = global_measures(U, D2, W, V_mu=V_mu)
    n = len(U)
    rng = np.random.default_rng(seed)
    I_perm = np.empty(permutations)
    CP_perm = np.empty(permutations)
    for b in range(permutations):
        p = rng.permutation(n)
        m = global_measures(U[p], D2[np.ix_(p, p)], W, V_mu=V_mu)
        I_perm[b] = m["I"]
        CP_perm[b] = m["C_P"]
    pI = (1 + np.sum(I_perm >= obs["I"] - 1e-15)) / (permutations + 1)
    pCP = (1 + np.sum(CP_perm <= obs["C_P"] + 1e-15)) / (permutations + 1)
    return GlobalResult(
        I=obs["I"], C_F=obs["C_F"], C_P=obs["C_P"], C_T=obs["C_T"],
        kappa=obs["kappa"], CP_over_CT=obs["CP_over_CT"], L=obs["L"],
        center_residual_ratio=obs["center_residual_ratio"],
        p_I_upper=float(pI), p_CP_lower=float(pCP), permutations=permutations,
    )


def local_diagnostics(
    U: np.ndarray,
    D2: np.ndarray,
    W: np.ndarray,
    *,
    permutations: int = 499,
    seed: int = 0,
    center_tol: float = 1e-8,
) -> pd.DataFrame:
    """Weighted conditional local Moran and pairwise Geary diagnostics.

    The focal object is fixed.  For each permutation, distinct remaining
    objects are assigned uniformly to the fixed positive-weight neighbor
    positions, preserving the row's possibly unequal weights.
    """
    W = validate_weights(W, require_symmetric=True)
    assert_frechet_first_order(U, tol=center_tol)
    n = len(U)
    V = float(np.sum(U * U))
    rng = np.random.default_rng(seed)
    all_idx = np.arange(n)
    rows: list[tuple] = []
    for i in range(n):
        nbr = np.flatnonzero(W[i] > 0)
        omega_i = float(W[i, nbr].sum())
        if len(nbr) == 0 or omega_i <= 0:
            rows.append((np.nan,) * 9)
            continue
        wrow = W[i, nbr]
        candidates = all_idx[all_idx != i]
        lag_i = np.sum(wrow[:, None] * U[nbr], axis=0)
        local_I = n / V * float(U[i] @ lag_i)
        denom = float(D2[i, candidates].sum())
        if denom <= 0:
            rows.append((local_I, np.nan, "undefined", np.nan, np.nan, "undefined", omega_i, len(nbr), denom))
            continue
        local_G = (n - 1) / omega_i * float(np.dot(wrow, D2[i, nbr])) / denom
        perm_I = np.empty(permutations)
        perm_G = np.empty(permutations)
        for b in range(permutations):
            assigned = rng.choice(candidates, size=len(nbr), replace=False)
            perm_lag = np.sum(wrow[:, None] * U[assigned], axis=0)
            perm_I[b] = n / V * float(U[i] @ perm_lag)
            perm_G[b] = (
                (n - 1) / omega_i * float(np.dot(wrow, D2[i, assigned])) / denom
            )
        if local_I >= 0:
            pI = (1 + np.sum(perm_I >= local_I - 1e-15)) / (permutations + 1)
            iclass = "aligned"
        else:
            pI = (1 + np.sum(perm_I <= local_I + 1e-15)) / (permutations + 1)
            iclass = "anti-aligned"
        if local_G <= 1:
            pG = (1 + np.sum(perm_G <= local_G + 1e-15)) / (permutations + 1)
            gclass = "homogeneous"
        else:
            pG = (1 + np.sum(perm_G >= local_G - 1e-15)) / (permutations + 1)
            gclass = "transition/outlier"
        rows.append((local_I, pI, iclass, local_G, pG, gclass, omega_i, len(nbr), denom))
    df = pd.DataFrame(rows, columns=[
        "local_I", "p_local_I", "I_direction", "local_G", "p_local_G",
        "G_direction", "weight_sum", "neighbor_count", "local_G_denominator",
    ])
    df["q_local_I"] = bh_adjust(df["p_local_I"].fillna(1.0).to_numpy())
    df["q_local_G"] = bh_adjust(df["p_local_G"].fillna(1.0).to_numpy())
    df["I_significant"] = df["q_local_I"] <= 0.05
    df["G_significant"] = df["q_local_G"] <= 0.05
    return df


def intrinsic_circle_mean(
    theta: Iterable[float], *, tie_tol: float = 1e-10
) -> tuple[float, dict[str, float | bool]]:
    """Exact intrinsic Fréchet mean for squared geodesic distance on S^1.

    The objective is piecewise quadratic between antipodal breakpoints.  We
    enumerate those intervals, evaluate every interior quadratic minimizer and
    breakpoint, and reject numerically nonunique minima.
    """
    x = np.mod(np.asarray(list(theta), dtype=float), TAU)
    if x.ndim != 1 or len(x) == 0 or np.any(~np.isfinite(x)):
        raise ValueError("theta must be a nonempty finite one-dimensional array")
    breakpoints = np.unique(np.mod(x + np.pi, TAU))
    candidates: list[float] = []
    if len(breakpoints) == 1:
        breakpoints = np.array([breakpoints[0], breakpoints[0] + TAU])
        intervals = [(breakpoints[0], breakpoints[1])]
    else:
        b = np.sort(breakpoints)
        intervals = [(float(b[k]), float(b[k + 1])) for k in range(len(b) - 1)]
        intervals.append((float(b[-1]), float(b[0] + TAU)))
    for lo, hi in intervals:
        mid = 0.5 * (lo + hi)
        lifted = mid + wrap_angle(x - mid)
        q = float(np.mean(lifted))
        q += TAU * round((mid - q) / TAU)
        if lo - 1e-12 <= q <= hi + 1e-12:
            candidates.append(q % TAU)
        candidates.extend([lo % TAU, hi % TAU])
    # Include the resultant mean only as a candidate, never as the definition.
    z = np.mean(np.exp(1j * x))
    if abs(z) > 1e-14:
        candidates.append(float(np.angle(z) % TAU))
    c = np.unique(np.round(np.mod(candidates, TAU), 14))
    obj = np.array([np.sum(wrap_angle(x - q) ** 2) for q in c], dtype=float)
    order = np.argsort(obj)
    q0 = float(c[order[0]])
    f0 = float(obj[order[0]])
    distinct_gap = math.inf
    nonunique = False
    for idx in order[1:]:
        circ_sep = abs(float(wrap_angle(c[idx] - q0)))
        if circ_sep > 1e-8:
            distinct_gap = float(obj[idx] - f0)
            if distinct_gap <= tie_tol * max(1.0, f0):
                nonunique = True
            break
    if nonunique:
        raise ValueError("Intrinsic circle Fréchet mean is nonunique at numerical tolerance")
    residual_sum = float(np.sum(wrap_angle(x - q0)))
    return q0, {
        "objective": f0,
        "second_minimum_gap": distinct_gap,
        "residual_sum": residual_sum,
        "unique": True,
    }


def intrinsic_torus_mean(Y: np.ndarray) -> tuple[np.ndarray, list[dict[str, float | bool]]]:
    Y = np.asarray(Y, dtype=float)
    if Y.ndim != 2:
        raise ValueError("Y must be n by q")
    means = []
    diagnostics = []
    for j in range(Y.shape[1]):
        q, diag = intrinsic_circle_mean(Y[:, j])
        means.append(q)
        diagnostics.append(diag)
    return np.asarray(means), diagnostics


def rook_weights(n_rows: int, n_columns: int) -> np.ndarray:
    """Return binary rook-contiguity weights for a rectangular lattice."""
    W = np.zeros((n_rows * n_columns, n_rows * n_columns), dtype=float)
    for row in range(n_rows):
        for column in range(n_columns):
            i = row * n_columns + column
            for delta_row, delta_column in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                neighbor_row = row + delta_row
                neighbor_column = column + delta_column
                if 0 <= neighbor_row < n_rows and 0 <= neighbor_column < n_columns:
                    W[i, neighbor_row * n_columns + neighbor_column] = 1.0
    return W


def sphere_log(mu: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Riemannian logarithm on a unit sphere."""
    mu = np.asarray(mu, dtype=float)
    y = np.asarray(y, dtype=float)
    cosine = float(np.clip(mu @ y, -1.0, 1.0))
    angle = math.acos(cosine)
    if angle < 1e-12:
        return np.zeros_like(mu)
    return angle / math.sin(angle) * (y - cosine * mu)


def sphere_exp(mu: np.ndarray, tangent: np.ndarray) -> np.ndarray:
    """Riemannian exponential on a unit sphere."""
    mu = np.asarray(mu, dtype=float)
    tangent = np.asarray(tangent, dtype=float)
    radius = float(np.linalg.norm(tangent))
    if radius < 1e-12:
        return mu.copy()
    return math.cos(radius) * mu + math.sin(radius) / radius * tangent


def sphere_mean(
    observations: np.ndarray, *, tol: float = 1e-12, maxiter: int = 500
) -> np.ndarray:
    """Compute the sample Fréchet mean for observations in one hemisphere."""
    observations = np.asarray(observations, dtype=float)
    mu = observations.mean(axis=0)
    mu /= np.linalg.norm(mu)
    for _ in range(maxiter):
        step = np.mean([sphere_log(mu, y) for y in observations], axis=0)
        if np.linalg.norm(step) < tol:
            break
        mu = sphere_exp(mu, step)
        mu /= np.linalg.norm(mu)
    else:
        raise RuntimeError("Spherical Fréchet mean did not converge")
    return mu


def sphere_components(
    observations: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Return center, logarithms, squared distances, and mean dispersion."""
    observations = np.asarray(observations, dtype=float)
    mu = sphere_mean(observations)
    logarithms = np.vstack([sphere_log(mu, y) for y in observations])
    pairwise_dots = np.clip(observations @ observations.T, -1.0, 1.0)
    distance_squared = np.arccos(pairwise_dots) ** 2
    dispersion = float(
        np.sum(np.arccos(np.clip(observations @ mu, -1.0, 1.0)) ** 2)
    )
    return mu, logarithms, distance_squared, dispersion


def _symfun(matrix: np.ndarray, function: Callable[[np.ndarray], np.ndarray]) -> np.ndarray:
    from scipy.linalg import eigh

    values, vectors = eigh((matrix + matrix.T) / 2.0)
    return (vectors * function(values)) @ vectors.T


def _sqrtm_spd(matrix: np.ndarray) -> np.ndarray:
    return _symfun(matrix, np.sqrt)


def _invsqrtm_spd(matrix: np.ndarray) -> np.ndarray:
    return _symfun(matrix, lambda values: 1.0 / np.sqrt(values))


def _logm_spd(matrix: np.ndarray) -> np.ndarray:
    return _symfun(matrix, np.log)


def _expm_sym(matrix: np.ndarray) -> np.ndarray:
    return _symfun(matrix, np.exp)


def airm_mean(
    matrices: np.ndarray, *, tol: float = 1e-11, maxiter: int = 200
) -> np.ndarray:
    """Affine-invariant Fréchet mean of symmetric positive-definite matrices."""
    matrices = np.asarray(matrices, dtype=float)
    mean = _expm_sym(np.mean([_logm_spd(matrix) for matrix in matrices], axis=0))
    for _ in range(maxiter):
        inverse_root = _invsqrtm_spd(mean)
        step = np.mean(
            [_logm_spd(inverse_root @ matrix @ inverse_root) for matrix in matrices],
            axis=0,
        )
        if np.linalg.norm(step, "fro") < tol:
            break
        root = _sqrtm_spd(mean)
        mean = root @ _expm_sym(step) @ root
        mean = (mean + mean.T) / 2.0
    else:
        raise RuntimeError("Affine-invariant Fréchet mean did not converge")
    return mean


def _spd_vector(matrix: np.ndarray) -> np.ndarray:
    return np.array([matrix[0, 0], math.sqrt(2.0) * matrix[0, 1], matrix[1, 1]])


def spd_components(
    matrices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Affine-invariant center, logarithms, and squared geodesic distances."""
    matrices = np.asarray(matrices, dtype=float)
    mean = airm_mean(matrices)
    inverse_root = _invsqrtm_spd(mean)
    whitened_logs = np.asarray(
        [_logm_spd(inverse_root @ matrix @ inverse_root) for matrix in matrices]
    )
    logarithms = np.vstack([_spd_vector(matrix) for matrix in whitened_logs])
    distance_squared = np.zeros((len(matrices), len(matrices)), dtype=float)
    for i in range(len(matrices)):
        inverse_root_i = _invsqrtm_spd(matrices[i])
        for j in range(i + 1, len(matrices)):
            distance = np.linalg.norm(
                _logm_spd(inverse_root_i @ matrices[j] @ inverse_root_i), "fro"
            )
            distance_squared[i, j] = distance_squared[j, i] = float(distance**2)
    dispersion = float(np.sum(logarithms * logarithms))
    return mean, logarithms, distance_squared, dispersion


def torus_components(
    observations: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Coordinatewise intrinsic center and product-circle geometry."""
    observations = np.asarray(observations, dtype=float)
    mean, _ = intrinsic_torus_mean(observations)
    logarithms = wrap_angle(observations - mean)
    distance_squared = np.zeros((len(observations), len(observations)), dtype=float)
    for column in range(observations.shape[1]):
        differences = wrap_angle(
            observations[:, column][:, None] - observations[:, column][None, :]
        )
        distance_squared += differences * differences
    dispersion = float(np.sum(logarithms * logarithms))
    return mean, logarithms, distance_squared, dispersion


def euclidean_components(
    observations: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Centered residuals and distances for an ordinary Euclidean baseline."""
    observations = np.asarray(observations, dtype=float)
    residuals = observations - observations.mean(axis=0, keepdims=True)
    distance_squared = pairwise_sqeuclidean(observations)
    return residuals, distance_squared, float(np.sum(residuals * residuals))


def scalar_components(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    return euclidean_components(np.asarray(values, dtype=float)[:, None])


def load_chicago(
    data_dir: Path | str = DATA_DIR,
) -> tuple[object, pd.DataFrame, np.ndarray, np.ndarray]:
    """Load Chicago geometry, compositions, shares, and queen weights."""
    import geopandas as gpd
    from libpysal.weights import Queen

    data_dir = Path(data_dir)
    composition = pd.read_csv(data_dir / "chicago_community_area_composition.csv")
    components = [
        "white_nh",
        "hispanic",
        "black_nh",
        "asian_nh",
        "other_multiple_nh",
    ]
    shares = composition[components].to_numpy(dtype=float, copy=True)
    shares /= shares.sum(axis=1, keepdims=True)
    geography = gpd.read_file(data_dir / "Chicago77.shp")
    geography["area_num"] = pd.to_numeric(
        geography["AREA_NUMBE"], errors="raise"
    ).astype(int)
    geography = geography.merge(
        composition[["area_num", "community", "category_total"]],
        on="area_num",
        validate="one_to_one",
    ).sort_values("area_num").reset_index(drop=True)
    queen = Queen.from_dataframe(geography, use_index=True)
    weights = np.zeros((len(geography), len(geography)), dtype=float)
    for i, neighbors in queen.neighbors.items():
        weights[int(i), np.asarray(neighbors, dtype=int)] = 1.0
    weights = np.maximum(weights, weights.T)
    return geography, composition, shares, weights


def load_california(data_dir: Path | str = DATA_DIR) -> dict[str, np.ndarray]:
    """Load the compact, pickle-free California commuting-tensor input."""
    archive = np.load(Path(data_dir) / "california_commuting_input.npz", allow_pickle=False)
    output = {name: archive[name] for name in archive.files}
    if output["tensors"].shape != (58, 2, 2):
        raise ValueError("California tensor archive has an unexpected shape")
    return output


def build_swiss_phases(
    data_dir: Path | str = DATA_DIR,
    *,
    split_year: int = 1990,
    minimum_per_era: int = 8,
) -> pd.DataFrame:
    """Construct the two-era intrinsic bloom-phase table from MeteoSwiss data."""
    source = pd.read_csv(Path(data_dir) / "meteoswiss.csv")
    rows: list[dict[str, float | int | str]] = []
    for location, group in source.groupby("location"):
        early = group[group["year"] < split_year]
        recent = group[group["year"] >= split_year]
        if len(early) < minimum_per_era or len(recent) < minimum_per_era:
            continue
        early_angle = TAU * early["bloom_doy"].to_numpy(float) / 365.25
        recent_angle = TAU * recent["bloom_doy"].to_numpy(float) / 365.25
        early_mean, _ = intrinsic_circle_mean(early_angle)
        recent_mean, _ = intrinsic_circle_mean(recent_angle)
        rows.append(
            {
                "location": str(location).replace("Switzerland/", ""),
                "lat": float(group["lat"].iloc[0]),
                "lon": float(group["long"].iloc[0]),
                "alt_m": float(group["alt"].iloc[0]),
                "n_early": int(len(early)),
                "n_recent": int(len(recent)),
                "theta_early": early_mean,
                "theta_recent": recent_mean,
                "mean_doy_early": early_mean * 365.25 / TAU,
                "mean_doy_recent": recent_mean * 365.25 / TAU,
            }
        )
    result = pd.DataFrame(rows).sort_values("location").reset_index(drop=True)
    result["recent_minus_early_days"] = (
        (result["mean_doy_recent"] - result["mean_doy_early"] + 182.625) % 365.25
    ) - 182.625
    return result


def summarize_local(frame: pd.DataFrame) -> dict[str, int]:
    """Count the four directional discovery classes and their overlap."""
    return {
        "local_moran_aligned": int(
            (frame["I_significant"] & (frame["I_direction"] == "aligned")).sum()
        ),
        "local_moran_anti_aligned": int(
            (frame["I_significant"] & (frame["I_direction"] == "anti-aligned")).sum()
        ),
        "local_geary_homogeneous": int(
            (frame["G_significant"] & (frame["G_direction"] == "homogeneous")).sum()
        ),
        "local_geary_transition_outlier": int(
            (
                frame["G_significant"]
                & (frame["G_direction"] == "transition/outlier")
            ).sum()
        ),
        "local_overlap": int(
            (frame["I_significant"] & frame["G_significant"]).sum()
        ),
    }


def jaccard(first: np.ndarray, second: np.ndarray) -> float:
    first = np.asarray(first, dtype=bool)
    second = np.asarray(second, dtype=bool)
    union = int(np.sum(first | second))
    return float(np.sum(first & second) / union) if union else 1.0


def classification_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Add display classes while keeping significance and direction separate."""
    output = frame.copy()
    output["moran_class"] = "not significant"
    output.loc[
        output["I_significant"] & (output["I_direction"] == "aligned"),
        "moran_class",
    ] = "aligned"
    output.loc[
        output["I_significant"] & (output["I_direction"] == "anti-aligned"),
        "moran_class",
    ] = "anti-aligned"
    output["geary_class"] = "not significant"
    output.loc[
        output["G_significant"] & (output["G_direction"] == "homogeneous"),
        "geary_class",
    ] = "homogeneous"
    output.loc[
        output["G_significant"]
        & (output["G_direction"] == "transition/outlier"),
        "geary_class",
    ] = "transition / outlier"
    return output


def set_notebook_style() -> None:
    """Apply the shared manuscript-aligned notebook display settings."""
    import matplotlib as mpl
    from cycler import cycler

    mpl.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "figure.facecolor": PAPER,
            "axes.facecolor": PAPER,
            "savefig.facecolor": PAPER,
            "font.family": "serif",
            "font.serif": [
                "CMU Serif",
                "STIX Two Text",
                "STIXGeneral",
                "DejaVu Serif",
            ],
            "font.size": 8.5,
            "mathtext.fontset": "cm",
            "axes.titlesize": 9.5,
            "axes.titleweight": "semibold",
            "axes.labelsize": 8.5,
            "axes.edgecolor": INK,
            "axes.linewidth": 0.65,
            "axes.prop_cycle": cycler(
                color=(BLUE, GOLD, GREEN, VERMILLION, PURPLE, SKY)
            ),
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "legend.fontsize": 7.5,
            "legend.frameon": False,
            "lines.linewidth": 1.2,
            "lines.markersize": 4.0,
            "text.color": INK,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def panel_label(ax: object, label: str, *, fontsize: float = 9.5) -> None:
    """Place a lowercase panel label outside the upper-left plotting corner."""
    ax.annotate(
        f"({label})",
        xy=(0.0, 1.0),
        xycoords="axes fraction",
        xytext=(-4.0, 5.0),
        textcoords="offset points",
        ha="right",
        va="bottom",
        fontsize=fontsize,
        fontweight="bold",
        color=INK,
        annotation_clip=False,
        clip_on=False,
        zorder=100,
    )


def map_axis(ax: object) -> None:
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def draw_graph(ax: object, coordinates: np.ndarray, weights: np.ndarray) -> None:
    coordinates = np.asarray(coordinates, dtype=float)
    for i, j in zip(*np.where(np.triu(weights, 1) > 0), strict=False):
        ax.plot(
            coordinates[[i, j], 0],
            coordinates[[i, j], 1],
            color="#B8B8B8",
            linewidth=0.42,
            alpha=0.62,
            zorder=0,
        )


def run_self_checks(seed: int = 99_173) -> dict[str, float]:
    """Run deterministic identities used to audit the public implementation."""
    rng = np.random.default_rng(seed)
    bridge_error = 0.0
    euclidean_error = 0.0
    scale_error = 0.0
    for _ in range(12):
        residuals = rng.normal(size=(15, 3))
        residuals -= residuals.mean(axis=0)
        tangent_distance = pairwise_sqeuclidean(residuals)
        intrinsic_distance = tangent_distance * (
            1.0 + 0.12 * np.tanh(tangent_distance)
        )
        np.fill_diagonal(intrinsic_distance, 0.0)
        upper = np.triu(rng.uniform(size=(15, 15)), 1)
        weights = upper + upper.T
        values = global_measures(
            residuals,
            intrinsic_distance,
            weights,
            V_mu=float(np.sum(residuals * residuals)),
        )
        bridge_error = max(
            bridge_error,
            abs(
                values["C_T"]
                - 14.0 / 15.0 * (1.0 + values["L"] - values["I"])
            ),
            abs(values["C_F"] - values["kappa"] * values["C_P"]),
        )
        euclidean = global_measures(
            residuals,
            tangent_distance,
            weights,
            V_mu=float(np.sum(residuals * residuals)),
        )
        euclidean_error = max(
            euclidean_error,
            abs(euclidean["C_F"] - euclidean["C_P"]),
            abs(euclidean["C_F"] - euclidean["C_T"]),
        )
        scaled = global_measures(
            3.7 * residuals,
            3.7**2 * intrinsic_distance,
            weights,
            V_mu=3.7**2 * float(np.sum(residuals * residuals)),
        )
        scale_error = max(
            scale_error,
            max(abs(scaled[key] - values[key]) for key in ("I", "C_F", "C_P", "C_T")),
        )

    circle = np.array([0.03, 0.15, 2.70, 2.95, 3.10, 5.75])
    circle_mean, _ = intrinsic_circle_mean(circle)
    circle_first_order = abs(float(np.sum(wrap_angle(circle - circle_mean))))
    return {
        "bridge_max_abs_error": float(bridge_error),
        "euclidean_reduction_max_abs_error": float(euclidean_error),
        "scale_invariance_max_abs_error": float(scale_error),
        "circle_first_order_abs_error": float(circle_first_order),
    }


if __name__ == "__main__":
    for _name, _value in run_self_checks().items():
        print(f"{_name}: {_value:.3e}")
