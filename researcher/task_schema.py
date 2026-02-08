"""Task definition schema for research agent benchmark.

A task represents a single research challenge: given a paper and its code,
improve upon a specific metric under compute/time constraints.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional
import json


class TaskTier(str, Enum):
    """Difficulty tiers for research tasks."""
    SETUP = "setup"                # Tier 1: get the repo running
    COMPREHENSION = "comprehension"  # Tier 2: answer questions about the code
    IMPLEMENTATION = "implementation"  # Tier 3: implement a component
    INTEGRATION = "integration"    # Tier 4: modify and integrate
    DEBUGGING = "debugging"        # Tier 5: find and fix bugs
    IMPROVE_HPARAM = "improve_hparam"  # Tier 6a: hyperparameter tuning
    IMPROVE_ALGO = "improve_algo"    # Tier 6b: constrained algorithmic improvement
    IMPROVE_ARCH = "improve_arch"    # Tier 6c: architecture modification
    IMPROVE_OPEN = "improve_open"    # Tier 6d: open-ended improvement


class MetricDirection(str, Enum):
    """Whether lower or higher is better for the target metric."""
    MINIMIZE = "minimize"  # e.g., loss, perplexity
    MAXIMIZE = "maximize"  # e.g., accuracy, BLEU


@dataclass
class ClusterBudget:
    """Compute budget available to the agent."""
    num_gpus: int = 2
    gpu_type: str = "L40"
    gpu_hours: float = 1.0
    max_concurrent_jobs: int = 2
    storage_gb: int = 50
    time_budget_hours: float = 2.0  # wall-clock time limit for the agent


@dataclass
class BaselineSpec:
    """Specification of the baseline to beat."""
    command: str  # command to run baseline (from repo root)
    metric_name: str  # name of the primary metric
    metric_path: str  # JSONPath to metric in results (e.g., "eval_loss")
    metric_direction: MetricDirection = MetricDirection.MINIMIZE
    metric_value: Optional[float] = None  # filled after baseline run
    num_seeds: int = 3  # baseline averaged over N seeds
    config_description: str = ""  # human-readable config (e.g., "125m, 200 steps")


@dataclass
class ImprovementTarget:
    """What the agent must achieve."""
    min_improvement_pct: float = 5.0  # must beat baseline by at least this %
    bronze_pct: float = 5.0
    silver_pct: float = 10.0
    gold_pct: float = 15.0

    def threshold(self, baseline_value: float, direction: MetricDirection) -> Dict[str, float]:
        """Compute absolute thresholds from baseline and improvement percentages."""
        thresholds = {}
        for level, pct in [("bronze", self.bronze_pct), ("silver", self.silver_pct), ("gold", self.gold_pct)]:
            if direction == MetricDirection.MINIMIZE:
                thresholds[level] = baseline_value * (1 - pct / 100)
            else:
                thresholds[level] = baseline_value * (1 + pct / 100)
        return thresholds


@dataclass
class Constraint:
    """A constraint on what the agent may or may not modify."""
    description: str
    # Files/dirs the agent MAY modify (glob patterns). Empty = everything allowed.
    allowed_paths: List[str] = field(default_factory=list)
    # Files/dirs the agent must NOT modify (glob patterns). Checked via hash.
    locked_paths: List[str] = field(default_factory=list)
    # Max parameter count (prevent using a bigger model)
    max_params: Optional[int] = None
    # Must use same eval data
    lock_eval_data: bool = True


@dataclass
class ResearchTask:
    """A complete research task definition.

    This is the top-level object that defines everything needed to:
    1. Build the container environment
    2. Present the task to the agent
    3. Evaluate the agent's result
    """
    # Identity
    task_id: str
    tier: TaskTier

    # Paper + code
    paper_url: str  # URL to paper PDF/markdown
    paper_md: str  # paper content as markdown
    repo_url: str  # GitHub URL
    repo_branch: str = "main"

    # Task description (what the agent sees)
    description: str = ""
    # Ground truth (what the tests use, agent doesn't see)
    truth: str = ""

    # Baseline and target
    baseline: BaselineSpec = field(default_factory=BaselineSpec)
    target: ImprovementTarget = field(default_factory=ImprovementTarget)

    # Compute budget
    budget: ClusterBudget = field(default_factory=ClusterBudget)

    # Constraints
    constraint: Constraint = field(default_factory=Constraint)

    # Container setup
    extra_pip_packages: List[str] = field(default_factory=list)
    setup_commands: List[str] = field(default_factory=list)  # run during container build
    pre_baseline_commands: List[str] = field(default_factory=list)  # run before baseline

    # Metadata
    tags: List[str] = field(default_factory=list)
    difficulty_estimate: str = ""  # "easy", "medium", "hard", "very_hard"

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to JSON-compatible dict."""
        d = {}
        for k, v in self.__dict__.items():
            if isinstance(v, Enum):
                d[k] = v.value
            elif hasattr(v, '__dict__') and not isinstance(v, str):
                d[k] = {kk: (vv.value if isinstance(vv, Enum) else vv)
                         for kk, vv in v.__dict__.items()}
            else:
                d[k] = v
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ResearchTask":
        """Deserialize from dict."""
        # Reconstruct nested dataclasses
        if "baseline" in d and isinstance(d["baseline"], dict):
            bl = d["baseline"]
            if "metric_direction" in bl:
                bl["metric_direction"] = MetricDirection(bl["metric_direction"])
            d["baseline"] = BaselineSpec(**bl)
        if "target" in d and isinstance(d["target"], dict):
            d["target"] = ImprovementTarget(**d["target"])
        if "budget" in d and isinstance(d["budget"], dict):
            d["budget"] = ClusterBudget(**d["budget"])
        if "constraint" in d and isinstance(d["constraint"], dict):
            d["constraint"] = Constraint(**d["constraint"])
        if "tier" in d:
            d["tier"] = TaskTier(d["tier"])
        return cls(**d)

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load(cls, path: Path) -> "ResearchTask":
        return cls.from_dict(json.loads(path.read_text()))

    def agent_prompt(self) -> str:
        """Generate the prompt shown to the agent (no ground truth)."""
        budget = self.budget
        lines = [
            self.description,
            "",
            "## Compute Budget",
            f"- GPUs: {budget.num_gpus}x {budget.gpu_type}",
            f"- GPU-hours: {budget.gpu_hours}",
            f"- Max concurrent jobs: {budget.max_concurrent_jobs}",
            f"- Storage: {budget.storage_gb} GB",
            f"- Wall-clock time limit: {budget.time_budget_hours} hours",
            "",
            "## Baseline",
            f"- Config: {self.baseline.config_description}",
            f"- Metric: {self.baseline.metric_name} = {self.baseline.metric_value}",
            f"- Direction: {'lower is better' if self.baseline.metric_direction == MetricDirection.MINIMIZE else 'higher is better'}",
            "",
            "## Your Goal",
        ]
        if self.baseline.metric_value is not None:
            thresholds = self.target.threshold(self.baseline.metric_value, self.baseline.metric_direction)
            lines.extend([
                f"- Bronze: {self.baseline.metric_name} {'<' if self.baseline.metric_direction == MetricDirection.MINIMIZE else '>'} {thresholds['bronze']:.4f}",
                f"- Silver: {self.baseline.metric_name} {'<' if self.baseline.metric_direction == MetricDirection.MINIMIZE else '>'} {thresholds['silver']:.4f}",
                f"- Gold:   {self.baseline.metric_name} {'<' if self.baseline.metric_direction == MetricDirection.MINIMIZE else '>'} {thresholds['gold']:.4f}",
            ])
        if self.constraint.allowed_paths:
            lines.extend([
                "",
                "## Constraints",
                "You may only modify files matching:",
            ])
            for p in self.constraint.allowed_paths:
                lines.append(f"  - {p}")
        if self.constraint.locked_paths:
            lines.extend([
                "You must NOT modify:",
            ])
            for p in self.constraint.locked_paths:
                lines.append(f"  - {p}")
        lines.extend([
            "",
            "## Instructions",
            "- The paper and code are in /home/user/workspace/",
            "- Submit experiments with: ray job submit --working-dir <dir> -- <command>",
            "- Check budget with: cat /home/user/.budget.json",
            "- Write your final result to /home/user/results/final_result.json",
            f'  Format: {{"job_id": "<id>", "{self.baseline.metric_name}": <value>, "description": "<what you changed>"}}',
        ])
        return "\n".join(lines)
