# ============================
# ColorMix Image Toy (with error normalized by obs_sigmas)
# ============================
from pathlib import Path

import numpy as np

from optimizer.problem import Problem, RunConfig


# ---------- Utilities ----------
def norm01(a):
    a = a.astype(float)
    return (a - a.min())/(a.max() - a.min() + 1e-12)

# ---------- Operators ----------
class OperatorsColorMix:
    """
    Defines N mixed-channel linear operators.
    Each dataset is a linear mix of RGB channels.
    """
    def __init__(self,H,W,mix_matrix):
        self.H,self.W=H,W
        self.mix_matrix=np.array(mix_matrix,float)  # shape (n_ops,3)
        self.n_ops=self.mix_matrix.shape[0]

    def apply_H(self,X,k):
        # X shape (H,W,3), mix_matrix[k] is length-3
        return np.tensordot(X,self.mix_matrix[k],axes=([2],[0]))

    def apply_HT(self,r,k):
        # adjoint: replicate residual across channels, weighted by mix row
        return r[...,None]*self.mix_matrix[k][None,None,:]

# ---------- Data Simulation ----------
def simulate_data(X,ops,rng,obs_sigmas):
    y=[]
    for k,s in enumerate(obs_sigmas):
        noise = s * rng.standard_normal(X.shape[:2])
        yk=ops.apply_H(X,k)+noise
        y.append(yk)
    return tuple(y)

# ---------- Reconstruction (Adam) ----------
def reconstruct_mle_adam(ops,y,
                         pred_sigmas=(0.1,0.1,0.1),
                         obs_sigmas=(0.1,0.1,0.1),
                         iters=800,eta=0.05,
                         beta1=0.9,beta2=0.999,eps=1e-8,
                         verbose=False):
    pred_sigmas = np.array(pred_sigmas,float)
    obs_sigmas  = np.array(obs_sigmas,float)

    H,W=ops.H,ops.W
    X=np.zeros((H,W,3))
    m=np.zeros_like(X); v=np.zeros_like(X)

    def add_term(X,Hf,Ht,y,s_pred,k,g):
        r=(Hf(X,k)-y)
        g+=Ht(r,k)/(s_pred**2)
        return g,r

    last_errs=None
    for t in range(1,iters+1):
        g=np.zeros_like(X)
        residuals=[]

        for k in range(ops.n_ops):
            g,r = add_term(X,ops.apply_H,ops.apply_HT,y[k],pred_sigmas[k],k,g)
            residuals.append(r)

        # Adam update
        m=beta1*m+(1-beta1)*g
        v=beta2*v+(1-beta2)*(g*g)
        mhat=m/(1-beta1**t)
        vhat=v/(1-beta2**t)
        X=X-eta*mhat/(np.sqrt(vhat)+eps)
        X=np.clip(X,0,1)

        # compute normalized errors (RMS / obs_sigma)
        errs=[]
        for k,r in enumerate(residuals):
            rms=np.sqrt(np.mean(r**2))
            errs.append(rms/obs_sigmas[k])
        last_errs=tuple(errs)

    if verbose:
        print("Final normalized RMS:",last_errs)
    return X,last_errs

# ---------- Wrapper ----------
class ImageToyProblem(Problem):
    """
    Synthetic image toy (3 dataset weights, 3 objectives).

    A hidden RGB image is the ground truth. Three noisy datasets are linear
    mixes of its channels (rows of mix_matrix). A candidate parameter vector holds
    the per-dataset sigmas of a weighted regression. The objectives are the
    normalized RMS errors between each dataset and its reconstruction.

    The ground truth and the noise use their own fixed seed (data_seed),
    so the data is the same for every optimization seed.
    """

    name = "image_toy"
    n_var = 3
    n_obj = 3

    def __init__(self,H=64,W=64,data_seed=0,
                 obs_sigmas=(1e-3,1e-3,1e-3),
                 mix_matrix=None,
                 eval_iters=1000,eval_eta=0.1):
        self.H,self.W=H,W
        self.rng=np.random.default_rng(data_seed)
        self.eval_iters,self.eval_eta=eval_iters,eval_eta

        # make RGB ground truth
        self.X_star=self.make_ground_truth(H,W,data_seed)

        if mix_matrix is None:
            mix_matrix=[
                [1,0,1],  # R+B
                [0,1,-1],  # G+B
                [1,-1,0]   # R+G
            ]
        self.ops=OperatorsColorMix(H,W,mix_matrix)
        self.obs_sigmas=obs_sigmas
        self.y=simulate_data(self.X_star,self.ops,self.rng,obs_sigmas)

        # compute and print effective dataset contributions
        contrib=np.sum(np.abs(self.ops.mix_matrix),axis=1)
        contrib=contrib/np.sum(contrib)
        print("Effective dataset contributions:",contrib)

    def make_ground_truth(self,h,w,seed=0):
        rng=np.random.default_rng(seed)
        X=np.zeros((h,w,3))
        yy,xx=np.mgrid[0:h,0:w]
        cx,cy=w/2,h/2
        r=np.sqrt((xx-cx)**2+(yy-cy)**2)

        # --- Background gradient ---
        base = (1 - r/np.max(r))
        for c in range(3):
            X[:,:,c] = 0.2*base

        # --- Red circle ---
        mask_circ = (xx-cx)**2 + (yy-cy)**2 < (0.25*min(h,w))**2
        X[mask_circ,0] = 1.0

        # --- Green square ---
        X[int(0.15*h):int(0.35*h),int(0.15*w):int(0.35*w),1] = 1.0

        # --- Blue triangle ---
        tri_mask = (yy > 0.6*h) & (xx > 0.6*w) & (yy-0.6*h < xx-0.6*w)
        X[tri_mask,2] = 1.0

        # --- Yellow band ---
        band = np.abs(yy-xx) < 8
        X[band,0:2] = 1.0

        # --- Checkerboard patch ---
        patch_size=8
        for i in range(0,h,patch_size):
            for j in range(0,w,patch_size):
                if (i//patch_size + j//patch_size) % 2 == 0:
                    X[i:i+patch_size,j:j+patch_size,1] = 0.8
                else:
                    X[i:i+patch_size,j:j+patch_size,2] = 0.8

        X += 0.05 * rng.standard_normal((h,w,3))
        return np.clip(X,0,1)

    def plot_datasets(self,path):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig,axes=plt.subplots(1,self.ops.n_ops+1,figsize=(16,4))
        axes[0].imshow(self.X_star); axes[0].set_title("Ground Truth")
        for k in range(self.ops.n_ops):
            axes[k+1].imshow(norm01(self.ops.apply_H(self.X_star,k)),cmap="gray")
            axes[k+1].set_title(f"H{k+1}")
        for ax in axes: ax.axis("off")
        plt.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)

    def predict(self,pred_sigmas=(0.1,0.1,0.1),iters=500,eta=0.05,path=None):
        """Reconstruct with one sigma vector. Save the comparison figure to `path` if given."""
        X_recon,errs=reconstruct_mle_adam(self.ops,self.y,
                                          pred_sigmas=pred_sigmas,
                                          obs_sigmas=self.obs_sigmas,
                                          iters=iters,eta=eta,verbose=True)
        if path is not None:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            fig,axes=plt.subplots(1,4,figsize=(14,4))
            axes[0].imshow(self.X_star); axes[0].set_title("Ground Truth"); axes[0].axis("off")
            axes[1].imshow(np.clip(X_recon,0,1)); axes[1].set_title("Reconstruction"); axes[1].axis("off")
            diff=np.abs(X_recon-self.X_star).mean(axis=2)
            axes[2].imshow(norm01(diff),cmap="magma"); axes[2].set_title("Abs Error"); axes[2].axis("off")
            axes[3].bar([f"H{i+1}" for i in range(self.ops.n_ops)],errs,color="skyblue")
            axes[3].set_title("Normalized RMS Errors")
            plt.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)
        return errs

    # ---- Problem interface ----

    def default_config(self) -> RunConfig:
        return RunConfig(
            max_generations=100,
            n_init=20,
            pool_size=20,
            lower_bound=1e-9,
            upper_bound=1.0,
            init_std=0.8,
            budget=3,
            most_recent=50,
            initial_epsilon=0.0,
            adaptive_epsilon=False,   # fixed epsilon (paper setting for this example)
            start_with_repair=False,  # first generation starts with variation
        )

    def prepare(self, output_dir):
        self.plot_datasets(Path(output_dir) / "datasets.png")

    def initial_guess(self, rng):
        # Arbitrary initial guess
        return np.array([0.5, 0.7, 0.2])

    def evaluate(self, params, generation):
        results=[]
        for sigmas in params:
            _,errs=reconstruct_mle_adam(self.ops,self.y,
                                        pred_sigmas=tuple(sigmas),
                                        obs_sigmas=self.obs_sigmas,
                                        iters=self.eval_iters,eta=self.eval_eta,verbose=False)
            results.append(errs)
        return np.array(results)

    # ---- Repair agent prompt text ----

    epsilon_prompt_hint = (
        f"- Increasing epsilon makes dominance stricter (fewer points survive on the Pareto front), "
        f"while decreasing epsilon makes it looser (more points survive).\n"
    )
    show_current_epsilon = False
    # If the repair agent fails, keep the pool with the Pareto members first
    repair_failure_pareto_first = True

    def repair_prompt_header(self, n_vars, n_objs):
        return (
            "System: You are an optimization agent tuning hyperparameters for a multi-dataset image reconstruction problem.\n\n"
            "Problem summary:\n"
            f"- Each candidate parameter vector has length {n_vars}: per-dataset scale parameters σ_k.\n"
            f"- Each reconstruction yields an objective vector of length {n_objs}: RMS residuals e_k for each dataset (lower is better).\n\n"
            "How parameters drive objectives:\n"
            "- The parameters σ_k act as scaling factors in the minimization process.\n"
            "- Smaller σ_k → dataset k has more influence, which may reduce its error but risks overfitting its noise and hurting other datasets.\n"
            "- Larger σ_k → dataset k has less influence, which may prevent overfitting but can leave its error high.\n"
            "- Your job:find σ values that reduce all RMS objectives without collapsing into overfitting on one dataset or ignoring others.\n\n"
        )

    def repair_prompt_footer(self, n_bad, n_vars):
        return (
            "**Guidelines:**\n"
            "- Learn from the correlations, PCA, and diversity to make small but meaningful edits.\n"
            "- Focus on reducing errors for bad sets while maintaining balance.\n"
            "- Do not collapse all σ to extremes (0 or max).\n"
            "- Keep Pareto front members unchanged.\n\n"
            "Output format (STRICT):\n"
            f"- Return a valid Python list of {n_bad} dicts.\n"
            f"- Each dict must have 'values' (a list of {n_vars} floats) and 'rationale' (short text).\n"
            "- The FIRST LINE of your reply must be ONLY that Python list—no extra text."
        )

    def describe_bad_set(self, idx, params, objs):
        bad_dims = np.where(objs == np.max(objs))[0].tolist()
        return (
            f"- Row {idx}: worst errors at objectives {bad_dims}; "
            f"params={np.array2string(params, precision=3, separator=', ')}, "
            f"objs={np.array2string(objs, precision=3, separator=', ')}"
        )
