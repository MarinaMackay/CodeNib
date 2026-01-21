"""
SWE-bench sampling utilities.

This module is designed for programmatic use in the CodeMiner infra. It exposes
configuration-driven APIs and avoids CLI side effects.
"""

from __future__ import annotations

import csv
import json
import subprocess
from collections import defaultdict
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from codeminer.dataset.swebench import SwebenchDataset
from codeminer.log_utils import get_logger

logger = get_logger(__name__)

PYTHON_LANGUAGE = "Python"

DEFAULT_LANGUAGES = {"Rust", "C", "C++", "JavaScript/TypeScript", PYTHON_LANGUAGE}
DEFAULT_MULTILINGUAL_DATASET = "SWE-bench/SWE-bench_Multilingual"
DEFAULT_LITE_DATASET = "princeton-nlp/SWE-bench_Lite"


@dataclass(frozen=True)
class SamplingConfig:
    languages: Set[str] = field(default_factory=lambda: set(DEFAULT_LANGUAGES))
    shallow_clone: bool = True
    min_instances: int = 3
    repos_per_language: int = 5
    instances_per_repo: int = 3
    dataset_split: str = "test"
    filter_instance: str = ".*"
    multilingual_dataset: str = DEFAULT_MULTILINGUAL_DATASET
    lite_dataset: str = DEFAULT_LITE_DATASET
    multilingual_csv_path: Optional[Path] = None
    cache_dir: Optional[Path] = None
    repo_cache_dir: Optional[Path] = None
    output_dir: Optional[Path] = None
    write_outputs: bool = True
    include_plots: bool = True


@dataclass
class SamplingResults:
    selected_repos: Dict[str, List[Dict[str, Any]]]
    selected_instances: List[Dict[str, Any]]
    repo_sizes: List[Dict[str, Any]]
    instance_difficulties: List[Dict[str, Any]]
    output_dir: Optional[Path]
    repo_plot_path: Optional[Path]
    instance_plot_paths: List[Path]


@lru_cache(maxsize=1)
def read_multilingual_csv(csv_path: str) -> Tuple[Set[str], Dict[str, str]]:
    """Read swebench_multilingual.csv and return (valid_languages, repo_to_language)."""
    valid_languages: Set[str] = set()
    repo_to_language: Dict[str, str] = {}
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            lang = row.get("Language")
            repo = row.get("Repository")
            if lang:
                valid_languages.add(lang)
            if repo and lang:
                repo_to_language[repo] = lang
    return valid_languages, repo_to_language


def select_indices_by_percentile(
    n: int,
    target_count: int,
    percentiles: Sequence[float],
    include_ends: bool = True,
) -> List[int]:
    """Select representative indices using percentiles with fallback logic."""
    if n <= target_count:
        return list(range(n))

    indices = {0, n - 1} if include_ends else set()
    for p in percentiles:
        indices.add(round(p * (n - 1)))

    indices = sorted(indices)

    if len(indices) < target_count:
        mid = n // 2
        all_indices = set(indices)
        offset = 1
        while len(all_indices) < target_count and offset < n:
            for c in (mid - offset, mid + offset):
                if 0 <= c < n and c not in all_indices:
                    all_indices.add(c)
                    if len(all_indices) >= target_count:
                        break
            offset += 1
        indices = sorted(all_indices)

    return indices[:target_count]


def select_representative_repos(
    repo_sizes: List[Dict[str, Any]],
    target_count: int = 5,
) -> Dict[str, List[Dict[str, Any]]]:
    """Select representative repos by file_count within each language group."""
    repos_by_group: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for repo_info in repo_sizes:
        repos_by_group[repo_info["language_group"]].append(repo_info)

    selected: Dict[str, List[Dict[str, Any]]] = {}
    for group, repos in repos_by_group.items():
        repos_sorted = sorted(repos, key=lambda x: x["file_count"])
        n = len(repos_sorted)

        indices = select_indices_by_percentile(n, target_count, [0.25, 0.5, 0.75])
        selected_repos = [repos_sorted[i] for i in indices]
        selected[group] = selected_repos

    return selected


def parse_patch_changed_loc(patch: str) -> int:
    """Parse changed_loc from patch content (added lines + deleted lines)."""
    added, deleted = 0, 0
    for line in patch.split("\n"):
        if line.startswith(("---", "+++", "@@", "diff ")):
            continue
        if line.startswith("+"):
            added += 1
        elif line.startswith("-"):
            deleted += 1
    return added + deleted


def calculate_instance_difficulties(
    repo_info: Dict[str, Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """Calculate difficulty (changed_loc) for all instances."""
    difficulties = []
    for repo, info in repo_info.items():
        for instance in info["instances"]:
            difficulties.append(
                {
                    "repo": repo,
                    "language_group": info["language_group"],
                    "instance_id": instance["instance_id"],
                    "changed_loc": parse_patch_changed_loc(instance.get("patch", "")),
                }
            )
    logger.info("Calculated difficulty for %d instances", len(difficulties))
    return difficulties


def select_representative_instances(
    instance_difficulties: List[Dict[str, Any]],
    selected_repos: Dict[str, List[Dict[str, Any]]],
    target_count: int = 3,
) -> List[Dict[str, Any]]:
    """Select representative instances by changed_loc within each selected repo."""
    selected_repo_names = {
        repo_info["repo"] for repos in selected_repos.values() for repo_info in repos
    }

    instances_by_repo: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for inst in instance_difficulties:
        if inst["repo"] in selected_repo_names:
            instances_by_repo[inst["repo"]].append(inst)

    selected_instances = []

    for _repo, instances in instances_by_repo.items():
        instances_sorted = sorted(instances, key=lambda x: x["changed_loc"])
        m = len(instances_sorted)

        if m <= target_count:
            indices = list(range(m))
            levels = ["low", "medium", "high"][:m] if m <= 3 else ["medium"] * m
        else:
            indices = select_indices_by_percentile(
                m, target_count, [0.25, 0.5, 0.75], include_ends=False
            )
            levels = ["low", "medium", "high"]

        for i, idx in enumerate(indices):
            inst = instances_sorted[idx]
            level = levels[i] if i < len(levels) else "medium"

            selected_instances.append(
                {
                    "repo": inst["repo"],
                    "language_group": inst["language_group"],
                    "instance_id": inst["instance_id"],
                    "changed_loc": inst["changed_loc"],
                    "difficulty_level": level,
                    "rank_in_repo": idx + 1,
                    "total_in_repo": m,
                }
            )

    return selected_instances


def group_instances_by_repo(
    multilingual_dataset: Optional[Iterable[Dict[str, Any]]],
    lite_dataset: Optional[Iterable[Dict[str, Any]]],
    repo_to_language: Dict[str, str],
    target_languages: Set[str],
) -> Dict[str, Dict[str, Any]]:
    repo_info: Dict[str, Dict[str, Any]] = {}

    if multilingual_dataset:
        for instance in multilingual_dataset:
            repo = instance.get("repo")
            if not repo:
                continue
            language_group = repo_to_language.get(repo)
            if not language_group or language_group not in target_languages:
                continue
            info = repo_info.setdefault(
                repo, {"language_group": language_group, "instances": []}
            )
            info["instances"].append(instance)

    if lite_dataset and PYTHON_LANGUAGE in target_languages:
        for instance in lite_dataset:
            repo = instance.get("repo")
            if not repo:
                continue
            info = repo_info.setdefault(
                repo, {"language_group": PYTHON_LANGUAGE, "instances": []}
            )
            info["instances"].append(instance)

    return repo_info


def filter_repos_by_instance_count(
    repo_info: Dict[str, Dict[str, Any]],
    min_instances: int = 3,
) -> Dict[str, Dict[str, Any]]:
    return {
        repo: info
        for repo, info in repo_info.items()
        if len(info["instances"]) >= min_instances
    }


def _is_hidden_path(path: Path) -> bool:
    return any(part.startswith(".") for part in path.parts)


def _clone_repo(repo: str, repo_dir: Path, shallow: bool) -> None:
    if repo_dir.exists():
        return
    repo_dir.parent.mkdir(parents=True, exist_ok=True)
    clone_cmd = ["git", "clone"]
    if shallow:
        clone_cmd.extend(["--depth", "1", "--no-tags"])
    clone_cmd.extend([f"https://github.com/{repo}.git", str(repo_dir)])
    logger.info("Cloning %s", repo)
    subprocess.run(
        clone_cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )


def _count_repo_files(repo_dir: Path) -> int:
    count = 0
    for path in repo_dir.rglob("*"):
        if not path.is_file():
            continue
        if _is_hidden_path(path):
            continue
        count += 1
    return count


def calculate_repo_sizes(
    repo_info: Dict[str, Dict[str, Any]],
    cache_dir: Path,
    shallow: bool = True,
) -> List[Dict[str, Any]]:
    repo_sizes = []
    for repo, info in repo_info.items():
        repo_dir = cache_dir / repo.replace("/", "_")
        _clone_repo(repo, repo_dir, shallow=shallow)
        file_count = _count_repo_files(repo_dir)
        repo_sizes.append(
            {
                "repo": repo,
                "language_group": info["language_group"],
                "file_count": file_count,
            }
        )
    return repo_sizes


def plot_repo_sizes(
    repo_sizes: List[Dict[str, Any]],
    selected_repos: Dict[str, List[Dict[str, Any]]],
    output_dir: Path,
) -> Optional[Path]:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib unavailable; skipping repo size plot")
        return None

    output_dir.mkdir(parents=True, exist_ok=True)

    grouped = defaultdict(list)
    for entry in repo_sizes:
        grouped[entry["language_group"]].append(entry)

    plt.figure(figsize=(10, 6))
    for group, entries in grouped.items():
        entries_sorted = sorted(entries, key=lambda x: x["file_count"])
        x_vals = list(range(len(entries_sorted)))
        y_vals = [e["file_count"] for e in entries_sorted]
        plt.scatter(x_vals, y_vals, label=group, alpha=0.7)

        selected = {e["repo"] for e in selected_repos.get(group, [])}
        selected_y = [e["file_count"] for e in entries_sorted if e["repo"] in selected]
        selected_x = [
            idx for idx, e in enumerate(entries_sorted) if e["repo"] in selected
        ]
        if selected_x:
            plt.scatter(selected_x, selected_y, marker="*", s=120)

    plt.title("Repo Size Distribution by Language")
    plt.xlabel("Repo Rank (per language, sorted by size)")
    plt.ylabel("Non-hidden File Count")
    plt.legend(loc="best")
    plt.tight_layout()

    plot_path = output_dir / "repo_sizes.png"
    plt.savefig(plot_path)
    plt.close()
    return plot_path


def plot_instance_difficulties(
    instance_difficulties: List[Dict[str, Any]],
    selected_repos: Dict[str, List[Dict[str, Any]]],
    selected_instances: List[Dict[str, Any]],
    output_dir: Path,
) -> List[Path]:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib unavailable; skipping instance plots")
        return []

    output_dir.mkdir(parents=True, exist_ok=True)

    selected_repo_names = {
        repo_info["repo"] for repos in selected_repos.values() for repo_info in repos
    }

    grouped = defaultdict(list)
    for entry in instance_difficulties:
        if entry["repo"] in selected_repo_names:
            grouped[entry["language_group"]].append(entry)

    selected_by_group = defaultdict(list)
    for entry in selected_instances:
        selected_by_group[entry["language_group"]].append(entry)

    plot_paths = []
    for group, entries in grouped.items():
        values = [e["changed_loc"] for e in entries]
        if not values:
            continue

        plt.figure(figsize=(8, 5))
        plt.hist(values, bins=20, alpha=0.7, color="#4C72B0")
        for selected in selected_by_group.get(group, []):
            plt.axvline(selected["changed_loc"], color="#C44E52", linestyle="--")

        plt.title(f"Instance Difficulty: {group}")
        plt.xlabel("Changed LOC")
        plt.ylabel("Count")
        plt.tight_layout()

        plot_path = output_dir / f"instance_difficulty_{group}.png"
        plt.savefig(plot_path)
        plt.close()
        plot_paths.append(plot_path)

    return plot_paths


def save_json(data: Any, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def save_csv(
    data: Sequence[Dict[str, Any]], path: str, fieldnames: Sequence[str]
) -> None:
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in data:
            writer.writerow(row)


def _resolve_cache_dir(cache_dir: Optional[Path]) -> Path:
    return cache_dir if cache_dir is not None else Path.home() / ".codeminer"


def load_multilingual_dataset(
    dataset_name: str,
    split: str,
    filter_instance: str,
) -> Iterable[Dict[str, Any]]:
    dataset = SwebenchDataset(
        dataset=dataset_name,
        split=split,
        filter_instance=filter_instance,
    )
    return dataset.load()


def load_lite_dataset(
    dataset_name: str,
    split: str,
    filter_instance: str,
) -> Iterable[Dict[str, Any]]:
    dataset = SwebenchDataset(
        dataset=dataset_name,
        split=split,
        filter_instance=filter_instance,
    )
    return dataset.load()


def run_sampling(config: SamplingConfig) -> SamplingResults:
    cache_dir = _resolve_cache_dir(config.cache_dir)
    repo_cache_dir = config.repo_cache_dir or cache_dir

    target_languages = set(config.languages)
    non_python_languages = target_languages - {PYTHON_LANGUAGE}

    repo_to_language: Dict[str, str] = {}
    if non_python_languages:
        if config.multilingual_csv_path is None:
            raise FileNotFoundError(
                "multilingual_csv_path is required when sampling non-Python languages"
            )
        csv_path = config.multilingual_csv_path
        valid_languages, repo_to_language = read_multilingual_csv(str(csv_path))
        invalid_languages = non_python_languages - valid_languages
        if invalid_languages:
            raise ValueError(
                f"Invalid languages: {', '.join(sorted(invalid_languages))}. "
                f"Available: {', '.join(sorted(valid_languages))}"
            )

    multilingual_dataset = None
    lite_dataset = None
    if non_python_languages:
        logger.info("Loading multilingual dataset: %s", config.multilingual_dataset)
        multilingual_dataset = load_multilingual_dataset(
            config.multilingual_dataset, config.dataset_split, config.filter_instance
        )

    if PYTHON_LANGUAGE in target_languages:
        logger.info("Loading Lite dataset: %s", config.lite_dataset)
        lite_dataset = load_lite_dataset(
            config.lite_dataset, config.dataset_split, config.filter_instance
        )

    repo_info = group_instances_by_repo(
        multilingual_dataset, lite_dataset, repo_to_language, target_languages
    )
    repo_info = filter_repos_by_instance_count(
        repo_info, min_instances=config.min_instances
    )

    repo_sizes = calculate_repo_sizes(
        repo_info, repo_cache_dir, shallow=config.shallow_clone
    )
    selected_repos = select_representative_repos(
        repo_sizes, target_count=config.repos_per_language
    )

    instance_difficulties = calculate_instance_difficulties(repo_info)
    selected_instances = select_representative_instances(
        instance_difficulties, selected_repos, target_count=config.instances_per_repo
    )

    output_dir = None
    repo_plot_path = None
    instance_plot_paths: List[Path] = []

    if config.write_outputs:
        output_dir = config.output_dir or (cache_dir / "swebench_sampling")
        output_dir.mkdir(parents=True, exist_ok=True)

        save_json(selected_repos, str(output_dir / "selected_repos.json"))
        save_json(selected_instances, str(output_dir / "selected_instances.json"))
        save_csv(
            selected_instances,
            str(output_dir / "selected_instances.csv"),
            [
                "language_group",
                "repo",
                "instance_id",
                "changed_loc",
                "difficulty_level",
                "rank_in_repo",
                "total_in_repo",
            ],
        )

        if config.include_plots:
            repo_plot_path = plot_repo_sizes(repo_sizes, selected_repos, output_dir)
            instance_plot_paths = plot_instance_difficulties(
                instance_difficulties, selected_repos, selected_instances, output_dir
            )

    return SamplingResults(
        selected_repos=selected_repos,
        selected_instances=selected_instances,
        repo_sizes=repo_sizes,
        instance_difficulties=instance_difficulties,
        output_dir=output_dir,
        repo_plot_path=repo_plot_path,
        instance_plot_paths=instance_plot_paths,
    )
