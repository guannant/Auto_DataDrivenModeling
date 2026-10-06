from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np


@dataclass
class RunConfig:
    """
    Settings for one Auto-DDM run. Each example supplies its own defaults
    through Problem.default_config(). main.py can override some of them.
    """
    max_generations: int = 50
    n_init: int = 20               # size of the initial Gaussian population
    pool_size: int = 20            # parent pool size kept every generation
    lower_bound: float = 1e-2
    upper_bound: float = 1000.0
    init_std: float = 0.8          # relative std of the initial Gaussian pool
    budget: int = 2                # max parameters an agent may edit per set
    most_recent: int = 50          # window for the diversity agent statistics
    initial_epsilon: float = 0.0   # fixed epsilon, or the start value if adaptive
    adaptive_epsilon: bool = False # True: survivor selection uses the LLM-proposed epsilon
    baseline: bool = False         # True: NSGA-II without the LLM agents
    start_with_repair: bool = True # first graph node: repair agent (True) or variation (False)
    diversity_every: int = 5       # run the diversity agent every N generations
    recursion_limit: int = 10_000  # LangGraph recursion limit

    def to_dict(self):
        return asdict(self)


class Problem:
    """
    Interface between the Auto-DDM framework and one optimization task.

    To add a new example, subclass Problem, implement the methods below,
    and register the class in examples/__init__.py.
    """

    name = "problem"
    n_var = None
    n_obj = None

    def default_config(self) -> RunConfig:
        return RunConfig()

    def prepare(self, output_dir: Path) -> None:
        """Optional hook. Runs once before the optimization starts."""

    def initial_guess(self, rng: np.random.Generator) -> np.ndarray:
        """Return the center point (shape (n_var,)) of the initial Gaussian pool."""
        raise NotImplementedError

    def evaluate(self, params: np.ndarray, generation: int) -> np.ndarray:
        """Evaluate a batch of parameter vectors (N, n_var). Return objectives (N, n_obj)."""
        raise NotImplementedError

    # ---- Repair agent prompt text (problem-specific parts) ----

    # Extra lines for the epsilon-adjustment prompt, inserted after the epsilon definition.
    epsilon_prompt_hint = ""
    # Show the current epsilon value to the LLM in the epsilon-adjustment prompt.
    show_current_epsilon = True
    # If the repair agent fails: False = keep the parent pool unchanged,
    # True = keep the same candidates, with the Pareto members first.
    repair_failure_pareto_first = False

    def repair_prompt_header(self, n_vars: int, n_objs: int) -> str:
        """System prompt text before the 'What you will be given' list."""
        raise NotImplementedError

    def repair_prompt_footer(self, n_bad: int, n_vars: int) -> str:
        """System prompt text after the 'What you will be given' list (guidelines and output format)."""
        raise NotImplementedError

    def describe_bad_set(self, idx: int, params: np.ndarray, objs: np.ndarray) -> str:
        """One line that describes a non-Pareto (bad) candidate to the repair agent."""
        raise NotImplementedError
