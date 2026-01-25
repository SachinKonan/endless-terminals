from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from tqdm import tqdm

# Ensure project root is importable
sys.path.insert(0, str(Path().resolve()))

from generator.apptainer_def_gen import iterate_def_template
from generator.completion_test_gen import generate_test_template as generate_final_test_template
from generator.initial_state_test_gen import generate_test_template as generate_initial_test_template
from generator.task_template_gen import generate_template as generate_task_template


@dataclass
class PipelineConfig:
    """Configuration for the task generation pipeline."""

    num_tasks: int
    out_dir: Path
    max_def_retries: int = 5
    max_num_completions: int = 4
    num_solutions: int = 256
    max_actions: int = 20
    model: str = "Qwen/Qwen3-32B"
    max_tokens: int = 1024
    task_temperature: float = 1.0
    test_temperature: float = 0.6
    solution_temperature: float = 1.0
    parallel_jobs: int = 1
    verbose: bool = False


def _safe_write_text(path: Path, content: str) -> None:
    """Write text to a file, creating parent directories as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _build_sif(def_path: Path, sif_path: Path) -> bool:
    """Build a SIF container from a def file."""
    sif_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        rc = subprocess.run(
            ["apptainer", "build", str(sif_path), str(def_path)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        ).returncode
        return rc == 0
    except FileNotFoundError:
        return False
    except subprocess.TimeoutExpired:
        return False


def _format_task_dir(base: Path, idx: int, width: int = 6) -> Path:
    """Generate a unique task directory path."""
    suffix = uuid.uuid4().hex[:8]
    return base / f"task_{idx:0{width}d}_{suffix}"


def _save_task_bundle(
    task_dir: Path,
    task_obj: Dict[str, Any],
    initial_test_code: str,
    def_text: str,
    final_test_code: str,
    summary: Dict[str, Any],
) -> Tuple[Path, Path, Path, Path, Path]:
    """Save all task artifacts to the task directory."""
    task_json = task_dir / "task.json"
    init_py = task_dir / "test_initial_state.py"
    final_py = task_dir / "test_final_state.py"
    def_file = task_dir / "container.def"
    sif_file = task_dir / "container.sif"

    sol_dir = task_dir / "solutions"
    sol_dir.mkdir(parents=True, exist_ok=True)

    _safe_write_text(task_json, json.dumps(task_obj, indent=4))
    _safe_write_text(init_py, initial_test_code)
    _safe_write_text(final_py, final_test_code)
    _safe_write_text(def_file, def_text)
    _safe_write_text(sol_dir / "summary.json", json.dumps(summary, indent=4))

    return task_json, init_py, final_py, def_file, sif_file


def _generate_one_task(idx: int, cfg: PipelineConfig) -> Optional[Path]:
    """Generate a single task through the full pipeline."""
    task_dir = _format_task_dir(cfg.out_dir, idx)
    try:
        # 1) Task description (template)
        if cfg.verbose:
            print(f"[{idx}] Generating task template")
        task_obj = generate_task_template(
            model=cfg.model,
            temperature=cfg.task_temperature,
        )
        if cfg.verbose:
            print(f"[{idx}] Task template generated")

        description = task_obj.get("description", "").strip()
        truth = task_obj.get("truth", "").strip()

        if not description or not truth:
            if cfg.verbose:
                print(f"[{idx}] Invalid task template – missing description/truth; skipping")
            return None

        if cfg.verbose:
            print(f"[{idx}] Description: {description}")
            print(f"[{idx}] Truth: {truth}")

        # 2) Initial tests
        initial_test_code = generate_initial_test_template(
            description,
            truth,
            temperature=cfg.test_temperature,
            model=cfg.model,
            instance=cfg.instance,
            api_version=cfg.api_version,
        )
        if cfg.verbose:
            print(f"[{idx}] Initial test code: {initial_test_code}")

        # 3) Apptainer .def – iterate up to max_def_retries
        def_text: Optional[str] = None
        try:
            def_text = iterate_def_template(
                description,
                truth,
                initial_test_code,
                max_rounds=cfg.max_def_retries,
                num_completions=cfg.max_num_completions,
                model=cfg.model,
                instance=cfg.instance,
                api_version=cfg.api_version,
                temperature=cfg.test_temperature,
                max_tokens=cfg.max_tokens,
            )
        except Exception as e:
            if cfg.verbose:
                print(f"[{idx}] Failed to produce a valid Apptainer def within retries: {e}")
            return None

        if not def_text:
            if cfg.verbose:
                print(f"[{idx}] No def text returned; skipping")
            return None

        if cfg.verbose:
            print(f"[{idx}] Def text: {def_text}")

        # 4) Final completion tests
        final_test_code = generate_final_test_template(
            description,
            truth,
            temperature=cfg.test_temperature,
            model=cfg.model,
            instance=cfg.instance,
            api_version=cfg.api_version,
        )
        if cfg.verbose:
            print(f"[{idx}] Final test code: {final_test_code}")

        # Save task artifacts
        task_obj["name"] = task_dir.name
        _, _, _, def_file, sif_file = _save_task_bundle(
            task_dir, task_obj, initial_test_code, def_text, final_test_code, summary={}
        )

        # Build SIF (if not already present)
        if not sif_file.exists():
            ok = _build_sif(def_file, sif_file)
            if not ok:
                if cfg.verbose:
                    print(f"[{idx}] Failed to build SIF; skipping task")
                shutil.rmtree(task_dir, ignore_errors=True)
                return None

        if cfg.verbose:
            print(f"[{idx}] Task saved at {task_dir}")
        return task_dir

    except Exception as e:
        if cfg.verbose:
            print(f"[{idx}] Unhandled error: {e}")
        if task_dir.exists():
            shutil.rmtree(task_dir, ignore_errors=True)
        return None


def run_pipeline(cfg: PipelineConfig) -> Dict[str, Any]:
    """Run the task generation pipeline."""
    cfg.out_dir.mkdir(parents=True, exist_ok=True)

    results: List[Optional[Path]] = []

    if cfg.parallel_jobs <= 1:
        for i in tqdm(range(1, cfg.num_tasks + 1)):
            res = _generate_one_task(i, cfg)
            results.append(res)
    else:
        with ThreadPoolExecutor(max_workers=cfg.parallel_jobs) as pool:
            futures = {
                pool.submit(_generate_one_task, i, cfg): i
                for i in range(1, cfg.num_tasks + 1)
            }
            for fut in tqdm(as_completed(futures), total=len(futures)):
                results.append(fut.result())

    num_success = sum(1 for r in results if r is not None)
    return {
        "requested": cfg.num_tasks,
        "succeeded": num_success,
        "success_rate": (num_success / cfg.num_tasks) if cfg.num_tasks else 0.0,
        "saved_dirs": [str(r) for r in results if r is not None],
    }


def parse_args(argv: Optional[List[str]] = None) -> PipelineConfig:
    """Parse command line arguments."""
    ap = argparse.ArgumentParser(description="Generate tasks, tests, and containers.")
    ap.add_argument("--num-tasks", type=int, default=100, help="How many tasks to generate")
    ap.add_argument("--out-dir", type=Path, default=Path("tasks"), help="Output directory")
    ap.add_argument(
        "--max-def-retries",
        type=int,
        default=1,
        help="Max retries to synthesize a passing Apptainer def (<=5)",
    )
    ap.add_argument(
        "--max-num-completions",
        type=int,
        default=8,
        help="Max completions to generate for Apptainer def",
    )
    ap.add_argument(
        "--solutions", type=int, default=128, help="Number of solution attempts per task"
    )
    ap.add_argument("--max-actions", type=int, default=16, help="Max shell actions per solution")
    ap.add_argument("--model", type=str, default="Qwen/Qwen3-32B")
    ap.add_argument("--task-temperature", type=float, default=1.0)
    ap.add_argument("--test-temperature", type=float, default=0.6)
    ap.add_argument("--solution-temperature", type=float, default=1.0)
    ap.add_argument("--jobs", type=int, default=8, help="Parallel jobs")
    ap.add_argument("--verbose", action="store_true", help="Verbose logs")
    ap.add_argument("--quiet", action="store_true", help="Silence verbose logs")

    args = ap.parse_args(argv)

    return PipelineConfig(
        num_tasks=args.num_tasks,
        out_dir=args.out_dir,
        max_def_retries=max(1, min(args.max_def_retries, 5)),
        max_num_completions=args.max_num_completions,
        num_solutions=args.solutions,
        max_actions=args.max_actions,
        model=args.model,
        task_temperature=args.task_temperature,
        test_temperature=args.test_temperature,
        solution_temperature=args.solution_temperature,
        parallel_jobs=max(1, args.jobs),
        verbose=args.verbose and not args.quiet,
    )


if __name__ == "__main__":
    cfg = parse_args()
    summary = run_pipeline(cfg)
    print(json.dumps(summary, indent=4))
