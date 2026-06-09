import torch
from botorch.models.gp_regression_fidelity import SingleTaskMultiFidelityGP
from botorch.models.transforms.outcome import Standardize
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood
from botorch import fit_gpytorch_mll
from botorch.models.cost import AffineFidelityCostModel
from botorch.acquisition.cost_aware import InverseCostWeightedUtility
from botorch.acquisition import PosteriorMean
from botorch.acquisition.knowledge_gradient import qMultiFidelityKnowledgeGradient
from botorch.acquisition.fixed_feature import FixedFeatureAcquisitionFunction
from botorch.optim.optimize import optimize_acqf, optimize_acqf_mixed
from botorch.acquisition.utils import project_to_target_fidelity


class MultiFidelityBO_Wu2019KG:
    """Multi-fidelity BO with qMFKG over two discrete fidelities {0.0, 1.0}.

    This object owns the surrogate model and the dataset and performs all BO
    logic EXCEPT scoring. Usage is ask/tell:

        suggest()               -> returns candidates (and their cost) to score
        register_observations() -> hand back the scores; the model is refit

    Data layout: train_x is (n, PROBLEM_DIM + 1); the first PROBLEM_DIM columns
    are the design and the last column is the fidelity (0.0 = low, 1.0 = high =
    target). train_obj is (n, 1) and is MAXIMIZED. Candidates returned by
    suggest() carry the chosen fidelity in their last column, so the caller can
    dispatch to the correct scoring function.
    """

    LOW_FIDELITY  = 0.0
    HIGH_FIDELITY = 1.0

    def __init__(self, train_x, train_obj, params=None):
        params = params or {}
        self.smoke_test = params.get("SMOKE_TEST", False)

        # --- dimensions -----------------------------------------------------
        self.problem_dim = params.get("PROBLEM_DIM", 5)   # design dims only
        self.dim = self.problem_dim + 1                   # design + fidelity
        self.fidelity_col = self.problem_dim              # fidelity column index (== 5)

        # --- acqf-optimization budgets -------------------------------------
        self.batch_size = params.get("BATCH_SIZE", 4)
        self.num_restarts = params.get("NUM_RESTARTS", 5 if not self.smoke_test else 2)
        self.raw_samples = params.get("RAW_SAMPLES", 128 if not self.smoke_test else 4)
        self.num_restarts_cv = 10 if not self.smoke_test else 2
        self.raw_samples_cv = 1024 if not self.smoke_test else 4
        self.num_fantasies = 128 if not self.smoke_test else 2

        self.tkwargs = {
            "dtype": torch.double,
            "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        }

        # --- full-space bounds (design + fidelity), shape (2, dim) ----------
        self.bounds = torch.tensor(
            [[0.0] * self.dim, [1.0] * self.dim], **self.tkwargs
        )
        self.target_fidelities = {self.fidelity_col: 1.0}

        # cost(s) = fixed_cost + weight * s  ->  cost(0)=1, cost(1)=10
        self.cost_model = AffineFidelityCostModel(
            fidelity_weights={self.fidelity_col: 9.0}, fixed_cost=1.0
        )
        self.cost_aware_utility = InverseCostWeightedUtility(cost_model=self.cost_model)

        # dataset is owned by this object
        self.train_x = train_x.to(**self.tkwargs)
        self.train_obj = train_obj.to(**self.tkwargs)
        self._fit_model()

    # ------------------------------------------------------------------ model
    def _fit_model(self):
        """(Re)build and fit the surrogate on the current dataset."""
        self.model = SingleTaskMultiFidelityGP(
            self.train_x,
            self.train_obj,
            outcome_transform=Standardize(m=1),
            data_fidelities=[self.fidelity_col],
        )
        self.mll = ExactMarginalLogLikelihood(self.model.likelihood, self.model)
        fit_gpytorch_mll(self.mll)

    def project(self, X):
        """Project X onto the target (highest) fidelity."""
        return project_to_target_fidelity(
            X=X, target_fidelities=self.target_fidelities, d=self.dim
        )

    def get_mfkg(self):
        """Construct the cost-aware qMultiFidelityKnowledgeGradient acquisition."""
        curr_val_acqf = FixedFeatureAcquisitionFunction(
            acq_function=PosteriorMean(self.model),
            d=self.dim,
            columns=[self.fidelity_col],
            values=[1.0],
        )
        _, current_value = optimize_acqf(
            acq_function=curr_val_acqf,
            bounds=self.bounds[:, :-1],   # design-only bounds
            q=1,
            num_restarts=self.num_restarts_cv,
            raw_samples=self.raw_samples_cv,
            options={"batch_limit": 10, "maxiter": 200},
        )
        return qMultiFidelityKnowledgeGradient(
            model=self.model,
            num_fantasies=self.num_fantasies,
            current_value=current_value,
            cost_aware_utility=self.cost_aware_utility,
            project=self.project,
        )

    # ------------------------------------------------------------------- ask
    def suggest(self):
        """Propose the next batch of points to evaluate (NO scoring done here).

        Returns
        -------
        new_x : (q, dim) tensor
            Candidates. The last column is the fidelity (0.0 or 1.0); use it to
            pick which scoring function to call for each row.
        cost : scalar tensor
            Total cost of evaluating this batch, per the affine cost model.
        """
        mfkg = self.get_mfkg()
        candidates, _ = optimize_acqf_mixed(
            acq_function=mfkg,
            bounds=self.bounds,
            fixed_features_list=[
                {self.fidelity_col: self.LOW_FIDELITY},
                {self.fidelity_col: self.HIGH_FIDELITY},
            ],
            q=self.batch_size,
            num_restarts=self.num_restarts,
            raw_samples=self.raw_samples,
            options={"batch_limit": 5, "maxiter": 200},
        )
        new_x = candidates.detach()
        cost = self.cost_model(new_x).sum()
        return new_x, cost

    # ------------------------------------------------------------------ tell
    def register_observations(self, new_x, new_obj):
        """Add externally-computed scores to the dataset and refit the model.

        Parameters
        ----------
        new_x : (q, dim) tensor
            The candidates returned by suggest() (fidelity column included).
        new_obj : (q, 1) tensor
            Observed objective values, in the same row order as new_x.
        """
        new_x = new_x.to(**self.tkwargs)
        new_obj = new_obj.to(**self.tkwargs)
        if new_obj.ndim == 1:
            new_obj = new_obj.unsqueeze(-1)
        self.train_x    = torch.cat([self.train_x, new_x])
        self.train_obj  = torch.cat([self.train_obj, new_obj])
        self._fit_model()

    # ---------------------------------------------------------- recommendation
    def get_recommendation(self):
        """Best design at the target fidelity (maximizer of the posterior mean)."""
        rec_acqf = FixedFeatureAcquisitionFunction(
            acq_function=PosteriorMean(self.model),
            d=self.dim,
            columns=[self.fidelity_col],
            values=[1.0],
        )
        final_rec, _ = optimize_acqf(
            acq_function=rec_acqf,
            bounds=self.bounds[:, :-1],
            q=1,
            num_restarts=10,
            raw_samples=512,
            options={"batch_limit": 5, "maxiter": 200},
        )
        return rec_acqf._construct_X_full(final_rec)   # re-append fidelity = 1.0


# ---------------------------------------------------------------------------
# The scoring + fidelity dispatch lives in YOUR loop, not in the class:
#
# bo = MultiFidelityBO_Wu2019KG(train_x, train_obj, params={"PROBLEM_DIM": 5})
# cumulative_cost = 0.0
# for _ in range(N_ITER):
#     new_x, cost = bo.suggest()
#     new_obj = torch.empty(new_x.shape[0], 1, dtype=torch.double)
#     for i, row in enumerate(new_x):
#         design = row[:bo.problem_dim]              # 1-D tensor of design vars
#         if row[bo.fidelity_col].item() >= 0.5:
#             new_obj[i] = high_fidelity_score(design)
#         else:
#             new_obj[i] = low_fidelity_score(design)
#     bo.register_observations(new_x, new_obj)
#     cumulative_cost += cost
# best = bo.get_recommendation()
# ---------------------------------------------------------------------------