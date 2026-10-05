import numpy as np
import ast
from utils.NSGA_related import arr2str
from agents.agent_log import log_agent


def create_llm_diversity_agent(llm, max_retries=10):
    """
    Diversity agent for multi-objective EA with ImageToy.
    Goal: Propose new candidate sets that expand parameter diversity
          while staying within bounds and not collapsing objectives.
    Uses full history (all_para, all_obj) for statistics.
    """

    def diversity_agent_node(state):
        parent_pool = state["parent_pool"]                 # (N, n_vars)
        parent_objectives = state["parent_objectives"]     # (N, n_objs)
        most_recent = state.get("most_recent", 100)  # (k, n_vars)
        all_para = state.get("all_para", parent_pool)[-most_recent:]      # global pool
        n_vars = parent_pool.shape[1]
        n_objs = parent_objectives.shape[1]
        bounds = state.get("bounds", None)
        budget = state.get("budget", 2)

        if bounds is not None:
            lower, upper = bounds
            bounds_str = f"[{float(np.min(lower))}, {float(np.max(upper))}]"
        else:
            bounds_str = "[0, 1]"

        pool_size = parent_pool.shape[0]

        # --- Diversity scores from the ENTIRE pool ---
        param_diversity = np.std(all_para, axis=0)

        # --- Step 1: Summaries ---
        param_param_corr = state["summary"]["param_param_corr"]
        param_obj_corr = state["summary"]["param_obj_corr"]
        pca_loadings = state["summary"]["pca_loadings"]
        pca_explained_variance = state["summary"]["pca_explained_variance"]

        # --- Step 2: System prompt ---
        system_message = (
            "System: You are a diversity agent for a multi-objective evolutionary algorithm.\n\n"
            "Your task:\n"
            "- make edits to the current sets in the pool that increase exploration while respecting bounds.\n"
            "- Focus on spreading values in parameters with low diversity and stabilizing those with very high diversity.\n"
            "- Avoid collapsing parameters to extremes (0 or max).\n\n"
            "Problem summary:\n"
            f"- Each candidate has {n_vars} parameters (σ_k per dataset).\n"
            f"- Each solution yields {n_objs} objectives (RMS errors e_k, lower is better).\n\n"
            "Provided data you can use:\n"
            "1) Full parent pool and objectives (current generation).\n"
            f"2) The statistics for the most recent {most_recent} sets.\n"
            "   • Diversity scores per parameter computed from ALL candidates.\n"
            "   • Array of length n_vars (index i → σ_i).\n"
            "   • Each score measures spread of σ values across the full pool.\n"
            "   • Low score = values clustered → encourage exploration.\n"
            "   • High score = values spread → encourage refinement/stabilization.\n"
            "   • Use these scores to decide which σ to perturb and by how much.\n"
            "3) Parameter–parameter correlation (matrix) of the entire history pool.\n"
            "   • Positive correlation: σ_i and σ_j move together.\n"
            "   • Negative correlation: σ_i and σ_j trade off.\n"
            "   • Use to design consistent edits across correlated parameters.\n"
            "4) Parameter–objective correlation of the entire history pool.\n"
            "   • How each σ dimension influences each objective.\n"
            "5) PCA loadings + explained variance.\n"
            "   • Use early PCs (high variance) to guide exploration directions.\n"
            f"6) Bounds reminder: all σ must remain inside {bounds_str}.\n"
            f"7) Edit budget: at most {budget} parameters may be shifted in each new set.\n\n"
            "**Guidelines:**\n"
            "- Inject diversity by perturbing clustered parameters\n"
            "- Spread out solutions across unexplored parameter space.\n"
            "- Prioritize exploration over exploitation.\n\n"
            "Output format (STRICT):\n"
            f"- Return a valid Python list of {pool_size} dicts.\n"
            f"- Each dict must have 'values' (a list of {n_vars} floats) and 'rationale' (short text).\n"
            "- The FIRST LINE of your reply must be ONLY that Python list—no extra text."
        )

        # --- Step 3: User message ---
        user_msg = (
            "\n==== Indexing & Semantics ====\n"
            f"• Parameters: 0..{n_vars-1} (σ_k per dataset).\n"
            f"• Objectives: 0..{n_objs-1} (RMS error, lower = better).\n\n"
            "==== Current Parent Pool (parameters) ====\n"
            + arr2str(parent_pool, decimals=3, max_rows=20)
            + "\n\n==== Current Parent Objectives ====\n"
            + arr2str(parent_objectives, decimals=3, max_rows=20)
            + "\n\n==== Global Diversity per Parameter (from most recent candidates) ====\n"
            + arr2str(param_diversity)
            + "\n\n==== Param–Param Correlation ====\n"
            + arr2str(param_param_corr)
            + "\n\n==== Param–Objective Correlation ====\n"
            + arr2str(param_obj_corr)
            + "\n\n==== PCA Loadings + Explained Variance ====\n"
            + arr2str(pca_loadings) + "\n"
            + arr2str(pca_explained_variance)
            + "\n\nInstructions:\n"
            "- Expand diversity in globally clustered parameters.\n"
            "- Stabilize extreme variation in globally high-diversity parameters.\n"
            "- Keep all σ inside bounds.\n"
            f"- Adjust at most {budget} parameters per set.\n"
            "- Output must cover all pool size (each dict corresponds to one candidate).\n"
        )

        # --- Step 4: Retry loop ---
        tries = 0
        sets = None
        report_raw = ""
        while sets is None and tries < max_retries:
            prompt = [
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_msg}
            ]
            result = llm(prompt)
            report_raw = result
            try:
                sets = ast.literal_eval(result.strip())
                valid = (
                    isinstance(sets, list)
                    and all(
                        isinstance(item, dict)
                        and 'values' in item
                        and 'rationale' in item
                        and isinstance(item['values'], (list, tuple, np.ndarray))
                        and len(item['values']) == n_vars
                        for item in sets
                    )
                    and len(sets) == pool_size
                )
                if not valid:
                    sets = None
            except Exception:
                sets = None

            log_agent(state, event="llm_call", agent="diversity", mode="edit", attempt=tries + 1,
                      messages=prompt, reply=result, valid=sets is not None)
            if sets is None:
                system_message += (
                    f"\nWARNING: Your previous output was NOT a valid Python list of {pool_size} dicts "
                    f"with 'values' (length {n_vars}) and 'rationale'. "
                    "The first line must be ONLY that Python list. Try again."
                )
            tries += 1

        if sets is None:
            print("⚠️ LLM Diversity agent failed, returning unchanged pool.")
            log_agent(state, event="result", agent="diversity", mode="edit", failed=True)
            return {
                **state,
                "diverse_pool": parent_pool,
                "rationales": [],
                "diversity_report_raw": report_raw
            }

        arrs = [np.array(item['values'], dtype=float) for item in sets]
        rationales = [str(item['rationale']) for item in sets]
        log_agent(state, event="result", agent="diversity", mode="edit", failed=False,
                  old_values=parent_pool, old_objectives=parent_objectives,
                  new_values=arrs, rationales=rationales)

        return {
            **state,
            "diverse_pool": np.vstack(arrs),
        }

    return diversity_agent_node
