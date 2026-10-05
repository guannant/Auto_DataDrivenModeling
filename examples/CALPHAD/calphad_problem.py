import os
import re
import json
import shutil
import subprocess
from collections import defaultdict
from pathlib import Path

import yaml
import numpy as np

from pycalphad import Database
from espei.datasets import load_datasets, recursive_glob
from espei.utils import unpack_piecewise, database_symbols_to_fit
from espei.optimizers.opt_generate_samples import DataGenerator

from optimizer.problem import Problem, RunConfig

EXAMPLE_DIR = Path(__file__).resolve().parent


def group_by_comps(input_dict):
    groups = defaultdict(list)
    for key, val_list in input_dict.items():
        # Extract the comps part (the content within parentheses after 'comps:')
        match = re.search(r"comps:\s*(\([^)]+\))", key)
        if not match:
            print("find no match")
            continue  # skip keys that do not contain the comps part
        comps_str = match.group(1)
        
        # Remove the colon and anything inside curly braces.
        # For example, transform "LAVES_C15: {X_MG: 0.31592}" to "LAVES_C15"
        cleaned = re.sub(r":\s*\{[^}]*\}", "", comps_str)
        # Remove extra spaces around commas and parentheses:
        cleaned = re.sub(r"\s*,\s*", ",", cleaned)
        cleaned = re.sub(r"\(\s*", "(", cleaned)
        cleaned = re.sub(r"\s*\)", ")", cleaned)
        
        # Use the cleaned comps string as the group key
        groups[cleaned].extend(val_list)
    
    # Compute the average for each group
    result = {}
    for comps, values in groups.items():
        avg = sum(values) / len(values) if values else None
        result[comps] = avg
        
    return result

def plot_grouped_bar(data_list):
    # Group values by key
    grouped_data = {}
    for d in data_list:
        for key, value in d.items():
            grouped_data.setdefault(key, []).append(abs(value))
    
    # Sort keys for consistent x-axis ordering
    keys = sorted(grouped_data.keys())    
    feed_for_gpt = {}
    # Plot each group's bars
    for i, key in enumerate(keys):
        values = grouped_data[key]
        feed_for_gpt[key] = np.average(values)
    return feed_for_gpt

def plot_bar_dict(data_dict):
    # Extract keys and values (assuming each value list has one element)
    keys = list(data_dict.keys())
    abe = [abs((data_dict[k][0][1] - data_dict[k][0][0])) * 100 for k in keys]
    feed_for_gpt ={k:v/100 for k, v in zip(keys, abe)}
    return feed_for_gpt
# Objective order. Index k of a parameter vector and of an objective vector is dataset k.
title ={
    "CUMG2_HMR": 20.42921352219551,
    "CUMG2_SMR": 0.528,
    "FCC_A1_HM_MIX": 13.410089581522925,
    "FCC_A1_SM_MIX": 0.528,
    "HCP_A3_HM_MIX": 8.467937527171069,
    "LAVES_C15_HMR": 26.362360143472927,
    "LAVES_C15_SMR": 0.528,
    "LAVES_C15_HM_MIX": 12.247749925325591,
    "LIQUID_HM_MIX": 12.593890860385642,
    "LIQUID_SM_MIX": 0.528,
    "(LAVES_C15)": 489.24809999999997,
    "(LAVES_C15,FCC_A1)": 43.486763079751725,
    "(LAVES_C15,LIQUID)": 59.47870989563999,
    "(LAVES_C15,CUMG2)": 5.489296573964274,
    "(LIQUID,LAVES_C15)": 0.6808378299754795,
    "(HCP_A3,CUMG2)": 17.7269585724,
    "(LIQUID,HCP_A3)": 478.8,
    "(LIQUID,CUMG2)": 48.083841263001595,
    "(LIQUID,FCC_A1)": 175.2315980625,
    "(FCC_A1,LAVES_C15)": 82.35009441792002,
    "(FCC_A1,LIQUID)": 12.411224504756625,
    "(FCC_A1)": 197.27386015759902
}

def process_output(out):
    zpf_errors = [] # by each dataset
    for item in out[1]:
        zpf_errors.append(group_by_comps(item))
    zpf_value = plot_grouped_bar(zpf_errors)
    thermoc_val = plot_bar_dict(out[0])
    merged_dict = {**thermoc_val,**zpf_value}
    ordered_keys = list(title.keys())
    ordered_out = [merged_dict[k] for k in ordered_keys]
    return  ordered_out


class CalphadProblem(Problem):
    """
    Cu-Mg CALPHAD assessment (22 dataset weights, 22 objectives).

    For each candidate weight vector, the problem:
    1. writes the weights to <surrogate_dir>/weights.json,
    2. runs the ESPEI surrogate MCMC (`espei --input <yaml>`) to get a calibrated TDB,
    3. evaluates the TDB against the 22 datasets and returns the 22 errors.
    """

    name = "calphad"
    n_var = len(title)
    n_obj = len(title)

    def __init__(self, initial_weights_path=None, surrogate_dir=None):
        self.initial_weights_path = Path(initial_weights_path or EXAMPLE_DIR / "initial_weights.json")
        # The ESPEI fork reads weights.json and the MLP surrogate from this folder.
        self.surrogate_dir = Path(
            surrogate_dir or os.environ.get("AUTODDM_SURROGATE_DIR") or EXAMPLE_DIR / "Pytorch_MLP_CV"
        )
        self.yaml_template_path = EXAMPLE_DIR / "run_mcmc.yaml"
        self.dataset_path = EXAMPLE_DIR / "input-data_entropyIncl"
        self.phase_models_path = EXAMPLE_DIR / "phase_models.json"
        self.tdb_initial_path = EXAMPLE_DIR / "Cu-Mg-generated.tdb"
        self.output_dir = EXAMPLE_DIR / "out"

        with open(self.phase_models_path) as f:
            phase_models = json.load(f)
        dbf = Database(str(self.tdb_initial_path))
        datasets = load_datasets(sorted(recursive_glob(str(self.dataset_path), '*.json')))
        self.data_gen = DataGenerator(dbf, datasets, phase_models)

    def default_config(self) -> RunConfig:
        return RunConfig(
            max_generations=50,
            n_init=20,
            pool_size=20,
            lower_bound=1e-2,
            upper_bound=1000.0,
            init_std=0.8,
            budget=self.n_obj,
            most_recent=50,
            initial_epsilon=0.0,
            adaptive_epsilon=False,   # main.py --adaptive-epsilon turns it on (paper setting)
            start_with_repair=True,
        )

    def prepare(self, output_dir):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        if shutil.which("espei") is None:
            raise RuntimeError(
                "The 'espei' command was not found. Install the Auto-DDM ESPEI fork "
                "(see README.md, section 'CALPHAD example')."
            )

    def initial_guess(self, rng):
        with open(self.initial_weights_path) as f:
            init_params_dict = json.load(f)
        return np.array([init_params_dict[k] for k in title], dtype=float)

    def evaluate(self, params, generation):
        results = []
        for i, p in enumerate(params):
            results.append(self.objective_fn(p, iteration=generation, post_fix=i))
        return np.array(results)

    def eval_output(self, tdb_file_path):
        dbf = Database(str(tdb_file_path))
        symbols_to_fit = database_symbols_to_fit(dbf)
        point = np.array([unpack_piecewise(dbf.symbols[s]) for s in symbols_to_fit])
        output = self.data_gen.eval_grouped_error(point)
        return process_output(output)

    def objective_fn(self, params, iteration, post_fix):
        # 1. Write the candidate weights where the ESPEI surrogate reads them
        params_dict = dict(zip(title.keys(), params))
        self.surrogate_dir.mkdir(parents=True, exist_ok=True)
        with open(self.surrogate_dir / "weights.json", "w") as wf:
            json.dump(params_dict, wf, indent=4)

        # 2. Write the ESPEI input file for this candidate
        base_name = f"LLM_agent_{iteration}_{post_fix}"
        db_path = self.output_dir / f"{base_name}.tdb"
        with open(self.yaml_template_path) as f:
            data = yaml.safe_load(f)
        data["system"]["datasets"] = str(self.dataset_path)
        data["system"]["phase_models"] = str(self.phase_models_path)
        data["mcmc"]["input_db"] = str(self.tdb_initial_path)
        data["output"]["logfile"] = str(self.output_dir / "espei_log.txt")
        data["output"]["output_db"] = str(db_path)
        data["output"]["tracefile"] = str(self.output_dir / f"{base_name}_trace.npy")
        data["output"]["probfile"] = str(self.output_dir / f"{base_name}_lnprob.npy")
        yaml_path = self.output_dir / "run_mcmc.yaml"
        with open(yaml_path, "w") as f:
            yaml.safe_dump(data, f)

        # 3. Run ESPEI. A TDB that already exists in this output folder is reused (resume).
        if db_path.exists():
            print(f"Database {db_path} already exists, skipping ESPEI run.")
        else:
            env = {**os.environ, "AUTODDM_SURROGATE_DIR": str(self.surrogate_dir)}
            try:
                subprocess.run(["espei", "--input", str(yaml_path)], check=True,
                               cwd=self.output_dir, env=env)
            except subprocess.CalledProcessError as e:
                raise RuntimeError(f"ESPEI run failed for {base_name}: {e}") from e

        return self.eval_output(db_path)

    # ---- Repair agent prompt text ----

    def repair_prompt_header(self, n_vars, n_objs):
        return (
            "System: You are an optimization agent tuning parameters (σ array) for a multi-objective evolutionary algorithm.\n\n"
            "Problem summary:\n"
            f"- There are {n_vars} datasets, each with its own RMS error objective e_k (lower is better).\n"
            f"- Each candidate parameter vector has length {n_vars}: per-dataset scale parameters σ_k.\n"
            f"- Each reconstruction yields an objective vector of length {n_objs}: RMS residuals e_k for each dataset (lower is better).\n\n"
            f"- The overall goal is to minimize all RMS objectives as much as possible (multi-objective minimization) by repairing the bad parameter sets (adjust the σ values to proper values).\n"
            "How parameters drive objectives:\n"
            "- The parameters σ_k act as scaling factors in the minimization process.\n"
            "- Smaller σ_k → dataset k has more influence, which will reduce its error but risks overfitting its noise and hurting other datasets.\n"
            "- Larger σ_k → dataset k has less influence, which will prevent overfitting but can leave its error high.\n"
            "- Your job:find σ values that reduce all RMS objectives without collapsing into overfitting on one dataset or ignoring others.\n\n"
        )

    def repair_prompt_footer(self, n_bad, n_vars):
        return (
            "**Guidelines:**\n"
            "- You should aim for all objectives to be as low as possible (<20).\n"
            "- Use these facts to identify which parameters to change, by how much, and why\n"
            "- Learn from the correlations, PCA, and diversity to make meaningful edits (either big or small) to the parameters.\n"
            "- Focus on reducing errors for bad sets while maintaining balance.\n"
            "- Do not collapse all σ to extremes (0 or max).\n"
            "Output format (STRICT):\n"
            f"- Return a valid Python list of {n_bad} dicts.\n"
            "- the current pareto sets might be a temporary solution during the exploration and don't trust their objectives in terms of what is optimal.\n"
            f"- Each dict must have 'values' (a list of {n_vars} floats) and 'rationale' (short text).\n"
            "- The FIRST LINE of your reply must be ONLY that Python list—no extra text."
        )

    def describe_bad_set(self, idx, params, objs):
        bad_dims = np.where(objs < 20)[0].tolist()
        return (
            f"- Row {idx}: has objectives more than threshold in those index {bad_dims}; "
            f"params={np.array2string(params, precision=3, separator=', ')}, "
            f"objs={np.array2string(objs, precision=3, separator=', ')}"
        )
