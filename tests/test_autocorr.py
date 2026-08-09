"""Deterministic numerical, data-integrity, and portability checks."""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.dont_write_bytecode = True
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import autocorr


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    checks = autocorr.run_self_checks()
    for name, value in checks.items():
        assert value < 1e-9, f"{name} failed: {value}"

    # Exact conditional-center check for weighted local pairwise Geary. The
    # focal unit is fixed while distinct remaining objects are assigned to
    # three neighbor positions with unequal weights.
    rng = np.random.default_rng(19)
    n = 6
    residuals = rng.normal(size=(n, 2))
    residuals -= residuals.mean(axis=0)
    distance_squared = autocorr.pairwise_sqeuclidean(residuals)
    weights = np.zeros((n, n), dtype=float)
    for neighbor, weight in zip([1, 2, 4], [0.2, 0.8, 1.7], strict=True):
        weights[0, neighbor] = weights[neighbor, 0] = weight
    weights[1, 2] = weights[2, 1] = 0.55
    weights[3, 4] = weights[4, 3] = 1.25
    weights[3, 5] = weights[5, 3] = 0.35
    weights[4, 5] = weights[5, 4] = 0.9
    local = autocorr.local_diagnostics(
        residuals, distance_squared, weights, permutations=199, seed=291
    )
    focal = 0
    neighbors = np.flatnonzero(weights[focal] > 0)
    row_weights = weights[focal, neighbors]
    candidates = [index for index in range(n) if index != focal]
    denominator = distance_squared[focal, candidates].sum()
    exact_values = []
    for assigned in itertools.permutations(candidates, len(neighbors)):
        numerator = float(
            np.dot(row_weights, distance_squared[focal, list(assigned)])
        )
        exact_values.append(
            (n - 1) / row_weights.sum() * numerator / denominator
        )
    assert math.isclose(float(np.mean(exact_values)), 1.0, abs_tol=1e-12)
    observed = (
        (n - 1)
        / row_weights.sum()
        * float(np.dot(row_weights, distance_squared[focal, neighbors]))
        / denominator
    )
    assert math.isclose(local.loc[focal, "local_G"], observed, abs_tol=1e-12)

    data = ROOT / "data"
    expected_hashes = {
        "chicago_community_area_composition.csv":
            "d01a19f09fa3c068a9f061270768023b5de6cd0eea71ec1bb17e51f7df6a0434",
        "Chicago77.shp":
            "7c62188f419911487ac1033a464f55e4b040f1d1c9afcb769e1c352088b98df7",
        "Chicago77.shx":
            "a1eec4184edfadfbc3bd540b6effc2ffac1cd4fe2380f7e712b530b011f4ad79",
        "Chicago77.dbf":
            "f08af37f52cc23a501a29ee1673b07f61a60e1e93cd4b54f206d5b7671cafc55",
        "california_commuting_input.npz":
            "eafa2920b2016d4756332f61c830c9e11df1958dab59a07040854092812a016d",
        "meteoswiss.csv":
            "3c77697278844f7e286a57c6e2d52f5d39806cfb475f993d0b29ba4033dfd310",
    }
    for name, expected in expected_hashes.items():
        assert digest(data / name) == expected, name

    chicago = pd.read_csv(data / "chicago_community_area_composition.csv")
    components = [
        "white_nh",
        "hispanic",
        "black_nh",
        "asian_nh",
        "other_multiple_nh",
    ]
    assert chicago["area_num"].tolist() == list(range(1, 78))
    assert chicago["area_num"].is_unique
    np.testing.assert_array_equal(
        chicago["category_total"].to_numpy(),
        chicago[components].sum(axis=1).to_numpy(),
    )
    expected_rows = {
        30: ("South Lawndale", 4295, 56416, 8292, 666, 215, 69884),
        32: ("The Loop", 21242, 4488, 4661, 10327, 1832, 42550),
        54: ("Riverdale", 35, 92, 7128, 0, 263, 7518),
    }
    indexed = chicago.set_index("area_num")
    columns = ["community", *components, "category_total"]
    for area_num, values in expected_rows.items():
        assert tuple(indexed.loc[area_num, columns]) == values

    california = autocorr.load_california(data)
    assert california["tensors"].shape == (58, 2, 2)
    assert california["geoid"].shape == (58,)
    eigenvalues = np.linalg.eigvalsh(california["tensors"])
    assert np.all(eigenvalues > 0)

    swiss = autocorr.build_swiss_phases(data)
    assert len(swiss) == 104
    assert (swiss[["n_early", "n_recent"]] >= 8).all().all()
    meteo = json.loads((data / "meteoswiss_source.json").read_text())
    assert meteo["dataset"]["required_attribution"] == "Source: MeteoSwiss."
    assert meteo["artifact"]["sha256"] == expected_hashes["meteoswiss.csv"]

    paper = json.loads((data / "paper_scale_summary.json").read_text())
    assert paper["controlled_spherical_behavior"]["global_permutations"] == 999
    assert paper["chicago_compositions"]["global_permutations"] == 1999
    assert paper["chicago_compositions"]["local_permutations"] == 999
    assert paper["california_commuting_tensors"]["n"] == 58
    assert paper["california_commuting_tensors"]["global_permutations"] == 999
    assert paper["california_commuting_tensors"]["local_permutations"] == 499
    assert paper["swiss_phenological_phases"]["n"] == 104
    assert paper["swiss_phenological_phases"]["global_permutations"] == 999
    assert paper["swiss_phenological_phases"]["local_permutations"] == 499

    text_suffixes = {
        "",
        ".cff",
        ".csv",
        ".gitignore",
        ".ipynb",
        ".json",
        ".md",
        ".py",
        ".toml",
        ".txt",
    }
    absolute_path_patterns = (
        re.compile(re.escape("/" + "Users" + "/") + r"[^/\s]+/"),
        re.compile(re.escape("/" + "home" + "/") + r"[^/\s]+/"),
        re.compile(r"[A-Za-z]:\\Users\\[^\\\s]+\\"),
    )
    workspace_fragments = (
        "Dropbox-" + "BaruchCollege",
        "Projects-" + "1Works",
        ".codex/" + "visualizations",
    )
    credential_patterns = (
        re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
        re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
        re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}\b"),
        re.compile(r"\bsk-[A-Za-z0-9]{20,}\b"),
        re.compile(
            r"(?i)(?:api[_-]?key|access[_-]?token|client[_-]?secret|password)"
            r"\s*[:=]\s*[\"'][^\"']{8,}[\"']"
        ),
    )
    disallowed_directories = {
        ".ipynb_checkpoints",
        ".matplotlib",
        ".mypy_cache",
        ".pytest_cache",
        ".venv",
        "__pycache__",
        "local",
        "logs",
        "venv",
    }
    disallowed_names = {
        ".env",
        ".env.local",
        ".rhistory",
        "credentials.json",
        "id_rsa",
        "id_ed25519",
        "secrets.json",
    }
    disallowed_raw_suffixes = {
        ".db",
        ".h5",
        ".hdf5",
        ".key",
        ".p12",
        ".parquet",
        ".pem",
        ".pfx",
        ".rdata",
        ".rds",
        ".sqlite",
    }

    for path in ROOT.rglob("*"):
        relative = path.relative_to(ROOT)
        if ".git" in relative.parts:
            continue
        assert not any(part in disallowed_directories for part in relative.parts), path
        if not path.is_file():
            continue
        assert path.name.lower() not in disallowed_names, path
        assert path.suffix.lower() not in disallowed_raw_suffixes, path
        assert path.stat().st_size <= 8 * 1024 * 1024, path
        if path.suffix.lower() in text_suffixes or path.name == ".gitignore":
            text = path.read_text(encoding="utf-8")
            assert not any(pattern.search(text) for pattern in absolute_path_patterns), path
            assert not any(fragment in text for fragment in workspace_fragments), path
            assert not any(pattern.search(text) for pattern in credential_patterns), path

    expected_data_files = {
        "README.md",
        "Chicago77.dbf",
        "Chicago77.shp",
        "Chicago77.shx",
        "Chicago77_LICENSE.txt",
        "Chicago77_README.md",
        "california_commuting_input.npz",
        "california_source.json",
        "chicago_community_area_composition.csv",
        "chicago_source.json",
        "meteoswiss.csv",
        "meteoswiss_source.json",
        "paper_scale_summary.json",
    }
    assert {path.name for path in data.iterdir()} == expected_data_files
    assert {path.name for path in (ROOT / "results").iterdir()} == {".gitkeep"}

    notebook_paths = sorted((ROOT / "notebooks").glob("*.ipynb"))
    assert len(notebook_paths) == 4
    expected_stems = {
        "simulation1_controlled_spherical_behavior",
        "realdata1_chicago_compositions",
        "realdata2_california_commuting_tensors",
        "realdata3_swiss_phenological_phases",
    }
    assert {path.stem for path in notebook_paths} == expected_stems
    for path in notebook_paths:
        notebook = json.loads(path.read_text())
        code_cells = [
            cell for cell in notebook["cells"] if cell["cell_type"] == "code"
        ]
        assert [cell["execution_count"] for cell in code_cells] == list(
            range(1, len(code_cells) + 1)
        )
        assert not any(
            output.get("output_type") == "error"
            for cell in code_cells
            for output in cell.get("outputs", [])
        )

    print("Autocorr numerical, data-integrity, and public-tree checks passed.")
    for name, value in checks.items():
        print(f"  {name}: {value:.3e}")


if __name__ == "__main__":
    main()
