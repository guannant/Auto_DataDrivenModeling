import json
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np

from utils.NSGA_related import (
    get_pareto_front_indices,
    initialize_gaussian_pool,
    select_parent_indices,
    get_bounds_and_constraints,
    get_index_mapping_note,
)
from optimizer.network import build_ea_langgraph_merged
from agents.agent_log import AgentLogger
from analysis.mpd import plot_mpd


class _Tee:
    """Write to the terminal and to a log file."""

    def __init__(self, stream, log_file):
        self.stream, self.log_file = stream, log_file

    def write(self, text):
        self.stream.write(text)
        self.log_file.write(text)
        return len(text)

    def flush(self):
        self.stream.flush()
        self.log_file.flush()

    def __getattr__(self, name):
        return getattr(self.stream, name)


@contextmanager
def log_console(path):
    """Copy stdout and stderr of this Python process to `path` (append)."""
    with open(path, "a") as log_file:
        log_file.write(f"\n===== Run started {time.strftime('%Y-%m-%d %H:%M:%S')} =====\n")
        old_out, old_err = sys.stdout, sys.stderr
        sys.stdout, sys.stderr = _Tee(old_out, log_file), _Tee(old_err, log_file)
        try:
            yield
        finally:
            sys.stdout, sys.stderr = old_out, old_err


def run_optimization(problem, config, llm, seed=0, output_dir="results", run_info=None):
    """
    Run the LLM-agentic evolutionary optimization (Auto-DDM) on one problem.

    Parameters
    ----------
    problem : optimizer.problem.Problem
        The task to optimize (for example, CALPHAD or the image toy).
    config : optimizer.problem.RunConfig
        Population sizes, bounds, budget, and epsilon settings.
    llm : callable
        Chat function: takes a list of messages and returns the reply text.
    seed : int
        Random seed for the evolutionary operators.
    output_dir : str or Path
        Folder for the run results. The run writes run.log (console output),
        agent_log.jsonl (all agent prompts and replies), the trace .npy files,
        and mpd.csv / mpd.png.
    run_info : dict, optional
        Extra values (for example, the LLM model) for run_config.json and runs.csv.

    Returns
    -------
    dict
        The final workflow state.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    run_info = dict(run_info or {})
    start = time.strftime("%Y-%m-%d %H:%M:%S")
    with log_console(output_dir / "run.log"):
        final_state = _run(problem, config, llm, seed, output_dir, run_info)
        print("Computing MPD metrics...")
        _, finals = plot_mpd([output_dir], labels=[f"{problem.name} seed {seed}"],
                             reference=problem.mpd_reference)
        append_run_summary(
            output_dir.parent / "runs.csv",
            {"run_dir": output_dir.name, "example": problem.name, "seed": seed,
             "start": start, "end": time.strftime("%Y-%m-%d %H:%M:%S"),
             "generations": final_state["generation"],
             "evaluations": len(final_state["all_para"]),
             **run_info, **finals[0]},
        )
        print(f"Optimization finished. Results are in {output_dir}")
    return final_state


def append_run_summary(path, row):
    """Add one row per finished run to <example results folder>/runs.csv."""
    import csv
    path = Path(path)
    new_file = not path.exists()
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row))
        if new_file:
            writer.writeheader()
        writer.writerow(row)


def _run(problem, config, llm, seed, output_dir, run_info):
    problem.prepare(output_dir)

    rng = np.random.default_rng(seed=seed)
    n_var = problem.n_var

    # Bounds
    lower = np.full(n_var, config.lower_bound, dtype=float)
    upper = np.full(n_var, config.upper_bound, dtype=float)
    bounds = (lower, upper)

    # Initialize population around the initial guess
    init_params = problem.initial_guess(rng)
    param_pool = initialize_gaussian_pool(
        rng=rng,
        center_point=init_params,
        n_samples=config.n_init,
        std=config.init_std,
        bounds=bounds,
    )

    # Initial objective evaluation
    objectives = problem.evaluate(param_pool, 0)

    # Select parents
    selected_idx = select_parent_indices(rng, objectives, config.pool_size)
    parent_pool = param_pool[selected_idx]
    parent_objectives = objectives[selected_idx]

    history = [
        {
            "objectives": parent_objectives,
            "parent_pool": parent_pool,
        }
    ]

    # Initial state for the LangGraph workflow
    init_state = {
        "problem": problem,
        "parent_pool": parent_pool,
        "parent_objectives": parent_objectives,
        "llm": llm,
        "bounds_and_constraints": get_bounds_and_constraints(bounds),
        "index_mapping_note": get_index_mapping_note(n_var),
        "history": history,
        "bounds": bounds,
        "pool_size": config.pool_size,
        "generation": 0,
        "budget": config.budget,
        "max_generations": config.max_generations,
        "all_para": parent_pool,
        "all_obj": parent_objectives,
        "rng": rng,
        "eps_vals": config.initial_epsilon,
        "adaptive_epsilon": config.adaptive_epsilon,
        "diversity_every": config.diversity_every,
        "most_recent": config.most_recent,
        "agent_logger": AgentLogger(output_dir / "agent_log.jsonl"),
    }

    workflow = build_ea_langgraph_merged(start_with_repair=config.start_with_repair)

    print(f"Starting optimization: example={problem.name}, seed={seed}, "
          f"generations={config.max_generations}")
    final_state = init_state
    epsilons = []

    # Stream over the workflow execution
    for event in workflow.stream(init_state, config={"recursion_limit": config.recursion_limit}):
        # Each event is a dict {node_name: state}
        node_name, node_state = next(iter(event.items()))

        if node_name != "EvalAndSurvivor":
            continue

        final_state = node_state
        eps = node_state["epsilon_used"]
        epsilons.append(eps)
        all_obj = node_state["all_obj"]

        pareto_idx = get_pareto_front_indices(all_obj, epsilon=eps)
        pareto_objs = all_obj[pareto_idx]

        # Average error per candidate, then mean & std across the Pareto set
        candidate_means = np.mean(pareto_objs, axis=1)

        print(f"----- Generation {node_state['generation']} -----")
        print(f"  Epsilon: {eps:.4f}")
        print(f"  Pareto front size: {len(pareto_idx)}")
        print(f"  Avg Pareto error (mean over objectives): {float(np.mean(candidate_means)):.4f}")
        print(f"  Std Pareto error (mean over objectives): {float(np.std(candidate_means)):.4f}")

        save_results(final_state, epsilons, config, seed, output_dir, run_info)

    return final_state


def save_results(state, epsilons, config, seed, output_dir, run_info=None):
    """Save the run trace. history_objectives has shape (generations + 1, pool_size, n_obj)."""
    output_dir = Path(output_dir)
    np.save(output_dir / "history_objectives.npy",
            np.array([h["objectives"] for h in state["history"]]))
    np.save(output_dir / "history_parent_pool.npy",
            np.array([h["parent_pool"] for h in state["history"]]))
    np.save(output_dir / "all_params.npy", state["all_para"])
    np.save(output_dir / "all_objectives.npy", state["all_obj"])
    np.save(output_dir / "epsilon_per_generation.npy", np.array(epsilons, dtype=float))
    with open(output_dir / "run_config.json", "w") as f:
        json.dump({"example": state["problem"].name, "seed": seed, **(run_info or {}),
                   **config.to_dict()}, f, indent=2)
