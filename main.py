"""
Auto-DDM entry point.

Examples:
    python main.py --example image_toy --seed 0
    python main.py --example calphad
    python main.py --list
"""
import argparse
import os
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
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


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
    parser.add_argument("--output-dir", default=None,
                        help="Results folder (default: results/<example>/seed_<seed>).")
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

    load_env_file(args.env_file)
    # Import after the .env file is loaded
    from agents.chatbox import openai_chat_completion, get_client
    from optimizer.runner import run_optimization

    get_client()  # stop early if the API key is missing

    problem = load_problem(args.example)
    config = problem.default_config()
    if args.generations is not None:
        config = replace(config, max_generations=args.generations)

    output_dir = Path(args.output_dir or REPO_ROOT / "results" / args.example / f"seed_{args.seed}")

    run_optimization(
        problem=problem,
        config=config,
        llm=openai_chat_completion,
        seed=args.seed,
        output_dir=output_dir,
    )


if __name__ == "__main__":
    main()
