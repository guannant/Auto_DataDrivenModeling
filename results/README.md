# Results

`main.py` writes each run to a new folder here:

```
results/
  image_toy/
    runs.csv                          one summary row per finished run
    run_20261005-143000_seed0/        one folder per run
    run_20261005-151200_seed42/
    run_20261005-160000_seed0_baseline/   NSGA-II baseline (--baseline)
  calphad/
    runs.csv
    run_20261006-090000_seed0/
```

The folder name is `run_<date>-<time>_seed<seed>`. To continue a stopped run, give its folder with `--output-dir`.

## runs.csv

| Column | Content |
|---|---|
| `run_dir` | Run folder name |
| `example`, `method`, `seed`, `model` | Example, `auto-ddm` or `baseline`, random seed, LLM model (`none` for the baseline) |
| `start`, `end` | Start and end time of the run |
| `generations`, `evaluations` | Completed generations, evaluated candidates |

## Files in a run folder

| File | Content |
|---|---|
| `run.log` | Console output of the run |
| `agent_log.jsonl` | All agent prompts, LLM replies, and proposed edits (see the main README) |
| `history_objectives.npy` | Parent-pool objectives per generation, shape (generations + 1, pool size, objectives) |
| `history_parent_pool.npy` | Parent-pool parameters per generation |
| `all_params.npy`, `all_objectives.npy` | All evaluated candidates |
| `epsilon_per_generation.npy` | ε used for survivor selection in each generation |
| `run_config.json` | Example, seed, model, and run settings |
| `datasets.png` | Image toy only: ground truth and the three datasets |
| `LLM_agent_<generation>_<index>.tdb` | CALPHAD only: calibrated TDB per candidate, with MCMC trace and probability files |
| `espei_log.txt`, `run_mcmc.yaml` | CALPHAD only: ESPEI log and input file |

Git does not track the run folders or `runs.csv`.
