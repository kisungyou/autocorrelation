# Riemannian spatial autocorrelation

This repo is a notebook-first public companion to *Spatial Autocorrelation for
Manifold-Valued Geographic Data*. It
contains the reusable Moran--Geary routines, executable versions of the
controlled spherical study and three geographic applications, compact public
inputs, and archived manuscript-scale summaries.

The release is intentionally smaller than the research workspace. Manuscript
sources, journal files, preparation scripts, duplicate figure formats, caches,
and internal logs are not included. Each checked-in notebook is executed, so
its tables and figures render directly on GitHub.

## Quick start

Use Python 3.12 or newer:

```sh
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[notebooks]"
jupyter lab
```

Open any file under `notebooks/`. Every notebook locates this folder whether
Jupyter starts here or inside `notebooks/`.

## Notebook map

| Notebook | Manuscript study | Main visual checks |
|---|---|---|
| `simulation1_controlled_spherical_behavior.ipynb` | Controlled behavior: weak signal and Geary divergence | null-adjacent spherical sample, four spatial regimes, permutation results, and the three-way Geary split |
| `realdata1_chicago_compositions.ipynb` | Chicago neighborhood composition | composition profiles, global baselines, intrinsic local maps, Euclidean decision differences, and sensitivity checks |
| `realdata2_california_commuting_tensors.ipynb` | California commuting-displacement tensors | affine-invariant tensor geometry, displacement ellipses, spatial graph, and local Moran/Geary classes |
| `realdata3_swiss_phenological_phases.ipynb` | Swiss phenological phases on a torus | two-era phase plane, bloom-date shifts, station graph, and local Moran/Geary classes |

The filenames follow the studies directly rather than using generic names such
as `example` or `quickstart`.

## Fresh and paper-scale calculations

Each notebook has a `PAPER_SCALE` switch near the beginning.

- The checked-in outputs use `PAPER_SCALE=True`, deterministic seeds, and the
  manuscript permutation counts. Every displayed statistic and figure is
  recomputed from the included inputs.
- Setting `PAPER_SCALE=False` uses 499 permutations for an interactive run.
  The exact manuscript-scale values also remain available in
  `data/paper_scale_summary.json` without rerunning the analyses.

All four notebooks run offline. The controlled study generates its own data;
the three geographic studies use the documented files in `data/`.

The release was executed end to end with Python 3.14.5, NumPy 2.3.5, pandas
3.0.3, SciPy 1.17.0, Matplotlib 3.10.8, GeoPandas 1.1.2, libpysal 4.15.0,
pyproj 3.7.2, and Shapely 2.1.2.

## License

The software is distributed under the MIT License; see `LICENSE`. The bundled
data retain their separately documented source and redistribution terms in
`DATA_SOURCES.md` and the files under `data/`.

## Composition

- `autocorr.py` contains the global and local Moran--Geary routines, sphere,
  affine-invariant SPD, and torus implementations, data loaders, plotting
  settings, and deterministic self-checks.
- `notebooks/` contains the four executable studies and keeps
  experiment-specific construction visible.
- `data/` contains the compact public inputs, source registries, distribution
  notices, and archived manuscript-scale summaries.
- `tests/test_autocorr.py` checks mathematical identities, data integrity,
  notebook execution, and public-tree portability.
- `results/` is reserved for user-generated outputs; the notebooks do not
  write into it.
- `LICENSE` and `CITATION.cff` record the MIT terms and sole-author citation
  metadata.

## Numerical scope

The implementation assumes a nonnegative, symmetric, hollow spatial-weight
matrix for the inferential protocol in the paper. It computes exact all-pairs
distance denominators and uses direct Monte Carlo permutations, which is
appropriate for the released examples but not optimized for very large data
sets. Tangent-based statistics are rejected when the supplied logarithms fail
the Fréchet first-order diagnostic.

The sphere routine assumes the observations lie in a region with a unique
Fréchet mean. The affine-invariant routine is written for small SPD matrices,
and the torus routine uses coordinatewise intrinsic circular means. These are
transparent study implementations, not a general manifold-learning library.

