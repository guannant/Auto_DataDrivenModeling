# Auto-DDM: An LLM-Guided Evolutionary Framework for Data-Driven Materials Modeling

This repository contains the code and data for the paper:

> G. Tang, N. Paulson. *An LLM-Guided Evolutionary Framework for Data-Driven Materials Modeling.*
> Preprint: https://www.researchsquare.com/article/rs-8574739/v2
> Archived code: https://doi.org/10.5281/zenodo.23147008

## Overview

Auto-DDM (Autonomous Data-Driven Modeling) finds dataset weights for model calibration. It treats each dataset error as one objective and runs a multi-objective evolutionary algorithm (NSGA-II with ε-dominance). Two LLM agents take part in the loop:

- **Repair agent.** It edits poorly performing candidates. It uses parameter–parameter correlations, parameter–objective correlations, PCA, and the spread of each parameter. If all candidates are on the Pareto front, it proposes a new ε instead.
- **Diversity agent.** Every 5 generations, it perturbs the parent pool to stop an early collapse of the search.

The repository contains the two case studies from the paper:

| Example | Parameters | Objectives | Description |
|---|---|---|---|
| `image_toy` | 3 | 3 | Reconstruct a synthetic RGB image from three noisy channel mixes |
| `calphad` | 22 | 22 | Weight 22 datasets in a Cu–Mg CALPHAD assessment (ESPEI + MLP surrogate) |

## Requirements

- Python 3.12. We tested the code on macOS with an Apple M4 chip.
- An OpenAI API key. The paper uses the model `gpt-5-mini`.
- For the CALPHAD example: the Auto-DDM fork of ESPEI, [guannant/llm_espei](https://github.com/guannant/llm_espei). `requirements.txt` installs the required commit.

## Installation

```bash
git clone https://github.com/guannant/Auto_DataDrivenModeling.git
cd Auto_DataDrivenModeling

conda create -n autoddm python=3.12
conda activate autoddm
pip install -r requirements.txt
```

`requirements.txt` pins the package versions that we used. The image toy example does not use ESPEI, PyCalphad, or PyTorch. To run only the image toy example, you can install only the "Core framework" lines.

## OpenAI settings

1. Copy the template: `cp .env.example .env`
2. Open `.env` and replace `your-openai-api-key-here` with your OpenAI API key.

| Variable | Default | Description |
|---|---|---|
| `OPENAI_API_KEY` | none (required) | Your API key |
| `OPENAI_MODEL` | `gpt-5-mini` | Model for both agents |
| `OPENAI_BASE_URL` | empty | URL of an OpenAI-compatible server. Empty means api.openai.com. |
| `OPENAI_TEMPERATURE` | `0.3` | Sampling temperature of the paper runs. If the model accepts only its default temperature, the code stops sending this value and prints a warning. |

Git does not track `.env`. Do not commit your key.

## Quick start

```bash
python main.py --list                                   # show the examples
python main.py --example image_toy --generations 5      # short test run
```

## Reproduce the paper results

The LLM replies are not deterministic. Two runs with the same seed can give different numbers. The trends in the paper (convergence of MPD-100 and MPD-20) should repeat. The exact values do not repeat.

### Image toy (paper Fig. 6)

Run the three seeds from the paper. Then plot the MPD metrics:

```bash
python main.py --example image_toy --seed 0
python main.py --example image_toy --seed 42
python main.py --example image_toy --seed 147

python analysis/mpd.py results/image_toy/seed_0 results/image_toy/seed_42 results/image_toy/seed_147 \
    --labels "seed 0" "seed 42" "seed 147" --reference 0.4498 --out results/image_toy/mpd.png
```

Each run has 100 generations. The reconstruction takes approximately 2 s per generation on a laptop CPU, plus the time for the LLM calls. The theoretical best distance is approximately 0.45 (dashed line). Expect MPD-20 to converge near this line.

### Cu–Mg CALPHAD assessment (paper Fig. 8)

```bash
python main.py --example calphad --seed 0
python analysis/mpd.py results/calphad/seed_0 --out results/calphad/mpd.png
```

Each candidate runs one ESPEI MCMC calibration (800 iterations) with the MLP surrogate. This takes approximately 35 s on an Apple M4. The run evaluates 20 initial candidates and 10 offspring per generation, so 50 generations take approximately 5 to 6 hours. For a short test, add `--generations 2`.

The run writes one calibrated TDB file per candidate (`LLM_agent_<generation>_<index>.tdb`) to the results folder. If a TDB file already exists, the run uses it again and does not run ESPEI for that candidate. Thus you can continue a stopped run with the same command. To start a new run, delete the folder or use `--output-dir`.

### Output files

Each run writes to `results/<example>/seed_<seed>/` (change it with `--output-dir`):

| File | Content |
|---|---|
| `history_objectives.npy` | Parent-pool objectives per generation, shape (generations + 1, pool size, objectives). `analysis/mpd.py` reads this file. |
| `history_parent_pool.npy` | Parent-pool parameters (dataset weights) per generation |
| `all_params.npy`, `all_objectives.npy` | All evaluated candidates |
| `epsilon_per_generation.npy` | ε used for survivor selection in each generation |
| `run_config.json` | Example, seed, and run settings |
| `mpd.csv`, `mpd.png` | MPD metrics per generation (written by `analysis/mpd.py`) |
| `datasets.png` | Image toy only: ground truth and the three datasets |

### Run settings

The values below are the defaults in `default_config()` of each example. They are the settings of the paper runs.

| Setting | image_toy | calphad |
|---|---|---|
| Generations | 100 | 50 |
| Initial population / parent pool / offspring | 20 / 20 / 10 | 20 / 20 / 10 |
| Parameter bounds | [1e-9, 1] | [0.01, 1000] |
| Initial guess | random, from the seed | `examples/CALPHAD/initial_weights.json` |
| Initial Gaussian spread (relative std) | 0.8 | 0.8 |
| Agent edit budget (parameters per candidate) | 3 | 22 |
| ε in survivor selection | always 0 | adaptive (proposed by the repair agent) |
| First step | variation | repair agent |
| Diversity agent | every 5 generations | every 5 generations |
| Diversity statistics window | 50 most recent candidates | 50 most recent candidates |

## How the CALPHAD example works

For each candidate weight vector, `examples/CALPHAD/calphad_problem.py`:

1. Writes the weights to `weights.json` in the surrogate folder (`examples/CALPHAD/Pytorch_MLP_CV/`).
2. Writes an ESPEI input file and runs `espei`. The environment variable `AUTODDM_SURROGATE_DIR` tells ESPEI where to find the surrogate model (`final_model_full_data.pt`, `scalerX.pkl`, `scalerY.pkl`) and the weights.
3. Evaluates the calibrated TDB against the 22 datasets in `examples/CALPHAD/input-data_entropyIncl/` and returns 22 errors.

To use a different surrogate folder, set `AUTODDM_SURROGATE_DIR` before you run `main.py`.

The repository contains the trained surrogate. It does not contain the surrogate training step (the full ESPEI MCMC run on an HPC cluster, see the paper Methods section).

## Repository layout

```
main.py                     Entry point. Selects and runs one example.
.env.example                Template for the OpenAI settings
requirements.txt            Pinned package versions
agents/
  chatbox.py                OpenAI client (reads the .env settings)
  repair.py                 Repair agent (parameter edits and ε adjustment)
  diversity.py              Diversity agent
optimizer/
  problem.py                Problem interface and RunConfig (run settings)
  network.py                LangGraph workflow (repair → variation → evaluation → survival)
  runner.py                 Runs one optimization and saves the results
utils/
  NSGA_related.py           NSGA-II operators, ε-dominance, population statistics
analysis/
  mpd.py                    MPD-100 and MPD-20 metrics and plot
examples/
  __init__.py               Example registry
  image_toy/                Synthetic image toy
  CALPHAD/                  Cu–Mg data, initial TDB, phase models, ESPEI settings, MLP surrogate
```

The agent prompts are in `agents/repair.py` and `agents/diversity.py`. The problem-specific parts of the repair prompt are in each example (`repair_prompt_header`, `repair_prompt_footer`, `describe_bad_set`).

## Add an example

1. Make a folder in `examples/`.
2. Subclass `optimizer.problem.Problem`. Implement `default_config`, `initial_guess`, `evaluate`, and the three repair-prompt methods.
3. Add the class to `EXAMPLES` in `examples/__init__.py`.
4. Run it with `python main.py --example <name>`.

## License

MIT. See `LICENSE`.
