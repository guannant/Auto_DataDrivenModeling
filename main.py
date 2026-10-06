"""
Auto-DDM entry point.

Examples:
    python main.py --example image_toy --seed 0
    python main.py --example calphad --adaptive-epsilon
    python main.py --example image_toy --baseline
    python main.py --list
"""
import argparse
import os
import time
from dataclasses import replace
from pathlib import Path

from examples import EXAMPLES, load_problem

REPO_ROOT = Path(__file__).resolve().parent


def load_env_file(path):
    """Read KEY=VALUE lines from a .env file. Variables already set in the shell win."""
    path = Path(path)
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        value = value.strip().strip('"').strip("'")
        if value:  # an empty value means "use the default"
            os.environ.setdefault(key.strip(), value)


def parse_args():
    parser = argparse.ArgumentParser(description="Run an Auto-DDM example.")
    parser.add_argument("--example", choices=sorted(EXAMPLES),
                        help="Example to run.")
    parser.add_argument("--list", action="store_true",
                        help="List the available examples and exit.")
    parser.add_argument("--seed", type=int, default=0,
                        help="Random seed (default: 0). The paper uses 0, 42 and 147 for image_toy.")
    parser.add_argument("--generations", type=int, default=None,
                        help="Number of generations (default: the example default).")
    parser.add_argument("--baseline", action="store_true",
                        help="Run the NSGA-II baseline without the LLM agents. "
                             "No OpenAI API key is necessary.")
    parser.add_argument("--adaptive-epsilon", action="store_true",
                        help="Let the repair agent adjust epsilon for survivor selection. "
                             "Without this option, epsilon stays at the --epsilon value.")
    parser.add_argument("--epsilon", type=float, default=None,
                        help="Fixed epsilon for the epsilon-dominance sorting, or the start "
                             "value with --adaptive-epsilon (default: 0).")
    parser.add_argument("--output-dir", default=None,
                        help="Results folder (default: a new folder "
                             "results/<example>/run_<date>-<time>_seed<seed>). "
                             "Give the folder of a stopped run to continue it.")
    parser.add_argument("--env-file", default=str(REPO_ROOT / ".env"),
                        help="Path to the .env file with the OpenAI settings.")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.list or args.example is None:
        print("Available examples:")
        for name in sorted(EXAMPLES):
            print(f"  {name}")
        print("\nRun one with: python main.py --example <name>")
        return

    if args.baseline and args.adaptive_epsilon:
        raise SystemExit("--adaptive-epsilon needs the repair agent. Do not use it with --baseline.")

    load_env_file(args.env_file)
    # Import after the .env file is loaded
    from agents.chatbox import openai_chat_completion, get_client, get_model
    from optimizer.runner import run_optimization

    if args.baseline:
        llm, model = None, "none"
    else:
        get_client()  # stop early if the API key is missing
        llm, model = openai_chat_completion, get_model()

    problem = load_problem(args.example)
    config = problem.default_config()
    if args.generations is not None:
        config = replace(config, max_generations=args.generations)
    if args.adaptive_epsilon:
        config = replace(config, adaptive_epsilon=True)
    if args.epsilon is not None:
        config = replace(config, initial_epsilon=args.epsilon)
    if args.baseline:
        config = replace(config, baseline=True)

    run_name = f"run_{time.strftime('%Y%m%d-%H%M%S')}_seed{args.seed}"
    if args.baseline:
        run_name += "_baseline"
    output_dir = Path(args.output_dir or REPO_ROOT / "results" / args.example / run_name)

    run_optimization(
        problem=problem,
        config=config,
        llm=llm,
        seed=args.seed,
        output_dir=output_dir,
        run_info={"model": model},
    )


if __name__ == "__main__":
    main()
