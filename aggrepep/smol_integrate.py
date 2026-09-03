"""
Functions to integrate and fit the 
finite modified Smoluchowski aggregation (Eq 3 / Eq 28 of Szala-Mendyk et al. 2023)
with fragmentation set to zero, for the CONSTANT and ADDITIVE kernels.

State is worked in NUMBER space n_i (average number of i-mers), i = 1..N, which
maps directly onto MD cluster counts. 
(kf in nm^3 / ps, V in nm^3, time in ps). 
Setting V=1 to work in concentration.
"""
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares, minimize_scalar

# used for fitting the model to data:
# which observable each model curve returns, given the integrated n(t)
_OBSERVABLES = {
    "number_average_size": number_average_size,   # L(t) = M1/M0
    "total_clusters":      total_clusters,         # n_cluster(t) = sum_i n_i
    "monomers":            lambda n: n[0],
    "dimers":              lambda n: n[1],
    "trimers":             lambda n: n[2],
}

# ---------- kernels (kf factored out so it is easily fittable) ----------
def kernel_matrix(kind, N, kf=1.0):
    """
    Return the NxN aggregation-kernel matrix K[i-1, j-1] = K_{i,j}.
    
    Parameters:
    -----------
    kind: str
        'constant' or 'additive' aggregation kernel
    N: int
        Number of chains in simulation/model. Corresponds to max cluster size.
    kf: float
        Rate constant for aggregation kernel. Default is 1.0.

    Returns:
    --------
    Kmat: np.ndarray
        NxN aggregation-kernel matrix K[i-1, j-1] = K_{i,j}.
    """
    i = np.arange(1, N + 1)
    I, J = np.meshgrid(i, i, indexing="ij")
    if kind == "constant":
        base = np.ones((N, N), dtype=float)
    elif kind == "additive":
        base = (I + J).astype(float)
    else:
        raise ValueError("kind must be 'constant' or 'additive'")
    return kf * base


# ---------- rhs of ODE ----------
def _make_rhs(Kmat, N, V):
    """
    Make the right-hand side of the finite Smoluchowski system, 
    given the kernel matrix Kmat.
    
    See Szala-Mendyk et al. 2023, Eq 3 / Eq 28.

    Parameters:
    -----------
    Kmat: np.ndarray
        NxN aggregation-kernel matrix K[i-1, j-1] = K_{i,j}.
    N: int
        Number of chains in simulation/model. Corresponds to max cluster size.
    V: float
        Volume of the system.

    Returns:
    --------
    rhs: callable
        The right-hand side function rhs(t,n) for the ODE.
    """
    # mask for the loss sum: only j with i+j <= N contribute
    i = np.arange(1, N + 1)
    I, J = np.meshgrid(i, i, indexing="ij")
    KM = np.where(I + J <= N, Kmat, 0.0)   # kernel with the "no cluster > N" cap

    def rhs(t, n):
        n = np.clip(n, 0.0, None)
        loss = n * (KM @ n)                 # n_i * sum_{j<=N-i} K_ij n_j
        gain = np.zeros(N)
        for iidx in range(2, N + 1):        # i-mer created from j + (i-j)
            j = np.arange(1, iidx)
            k = iidx - j
            gain[iidx - 1] = 0.5 * np.sum(Kmat[j - 1, k - 1] * n[j - 1] * n[k - 1])
        return (gain - loss) / V
    return rhs


# ---------- integrator ----------
def integrate(kind, kf, N, t_eval, V=1.0, n0=None,
              # method="LSODA", rtol=1e-8, atol=1e-10):
              method="RK45", rtol=1e-8, atol=1e-10):
    """
    Integrate the finite Smoluchowski system.
    Returns n of shape (N, len(t_eval)); n[i-1] is the average number of i-mers.
    Default initial condition: all N monomers free (n_1(0)=N, rest 0).

    Parameters:
    -----------
    kind: str
        'constant' or 'additive' aggregation kernels?
    kf: float
        rate constant 
    N: int
        Number of chains in simulation/model. 
        Corresponds to max cluster size
    t_eval: array-like
        The time points to evaluate the ODE at
    n0:
    method: str
        Integration method to use. Passed to scipy.integrate.solve_ivp
        Default: RK45 (Runge-Kutta)
    rtol, atol: float, float
        Relative and absolute tolerance levels. 

    Returns:
    --------
    sol.y: array-like
        The integration solution for the ODE.
    """
    Kmat = kernel_matrix(kind, N, kf)
    rhs = _make_rhs(Kmat, N, V)
    if n0 is None:
        n0 = np.zeros(N); n0[0] = N
    t_eval = np.asarray(t_eval, float)
    sol = solve_ivp(rhs, (float(t_eval[0]), float(t_eval[-1])), n0,
                    t_eval=t_eval, method=method, rtol=rtol, atol=atol)
    if not sol.success:
        raise RuntimeError(sol.message)
    return np.clip(sol.y, 0.0, None)


# ---------- observables ----------
def total_clusters(n):
    """ 
    M0(t) = sum_i n_i 

    Parameters:
    -----------
    n: numpy.array
        Number of clusters of each size, 
        Index corresponds to cluster size 
        (index 0 --> 1-cluster, index 1-->2-cluster, etc)

    Returns:
    --------
    M_0: float
        The zeroth moment 
    """
    return n.sum(axis=0)

def total_monomers(n):
    """ 
    M1(t) = sum_i i*n_i  (should be conserved = N)
    
    Parameters:
    -----------
    n: numpy.array
        Number of clusters of each size, 
        Index corresponds to cluster size 
        (index 0 --> 1-cluster, index 1-->2-cluster, etc)

    Returns:
    --------
    M_1: float
        The first moment, sum_i i n_i 
    """
    i = np.arange(1, n.shape[0] + 1)[:, None]
    return (i * n).sum(axis=0)

def number_average_size(n):   # L(t) = M1/M0
    """
    Computes the number-average cluster size, \mu_1=M1/M0
    Parameters:
    -----------
    n: numpy.array
        Number of clusters of each size, 
        Index corresponds to cluster size 
        (index 0 --> 1-cluster, index 1-->2-cluster, etc)

    Returns:
    --------
    \mu_1: float
        The number average cluster distribution value 
    """
    return total_monomers(n) / total_clusters(n)


# ---------- fitting ----------


def predict(kf, kind, N, t_data, V=1.0, observable="number_average_size",
            n0=None, **ivp):
    """
    Model prediction of `observable` at times t_data for a given kf.
    Pass observable='raw' to get the full n(t) array instead.

    Parameters:
    -----------
    kf: float
        Aggregation rate constant for a given aggregation kernel choice
    kind: str
        'constant' or 'additive' aggregation kernel
    N: int
        Total number of chains in the system. 
        (Total number of particles that can aggregate)
    t_data: array-like
        The time points to evaluate the ODE at
    V: float
        Volume of the system
    observable: str
        What observable to return. 
    n0: array-like | None
        initial cluster distribution
    ivp:
        scipy solve_ivp args

    Returns:
    --------
    observable
    """
    n = integrate(kind, kf, N, t_data, V=V, n0=n0, **ivp)
    if observable == "raw":
        return n
    return _OBSERVABLES[observable](n)

def fit_kf(t_data, y_data, kind, N, V=1.0, observable="number_average_size",
           kf0=1.0, weights=None, bounds=(0.0, np.inf), n0=None, **ivp):
    """
    Least-squares fit of the single parameter kf so that the finite
    Smoluchowski model reproduces y_data (measured at t_data).

    weights: optional per-point weights (e.g. 1/sigma) for the residuals.
    Returns dict with kf, chi2, reduced chi2, success, and the fitted curve.
    
    Parameters:
    -----------
    t_data: array-like
        The time points to evaluate the ODE at
    y_data: array-like
        The observed data to fit against
    kind: str
        'constant' or 'additive' aggregation kernel
    N: int
        Total number of chains in the system.
    V: float
        Volume of the system
    observable: str
        What observable to return.
    kf0: float
        Initial guess for kf.
    weights: array-like | None
        Optional per-point weights (e.g. 1/sigma) for the residuals.
    bounds: tuple
        Bounds for the kf parameter during fitting.
    n0: array-like | None
        Initial cluster distribution. If None, defaults to all monomers free.
    ivp:
        Additional arguments to pass to the ODE solver (solve_ivp).
    
    Returns:
    --------
    results: dict
        Dictionary containing:
        - 'kf': fitted kf value
        - 'chi2': chi-squared value of the fit
        - 'reduced_chi2': reduced chi-squared value of the fit
        - 'success': boolean indicating if the fit was successful
        - 'fitted_curve': the fitted curve evaluated at t_data
    """
    t_data = np.asarray(t_data, float)
    y_data = np.asarray(y_data, float)
    w = np.ones_like(y_data) if weights is None else np.asarray(weights, float)

    def resid(theta):
        kf = theta[0]
        y = predict(kf, kind, N, t_data, V=V, observable=observable, n0=n0, **ivp)
        return w * (y - y_data)

    res = least_squares(resid, x0=[kf0], bounds=([bounds[0]], [bounds[1]]))
    kf_hat = res.x[0]
    r = res.fun
    chi2 = float(np.sum(r**2))
    dof = max(len(y_data) - 1, 1)
    return {
        "kf": kf_hat,
        "chi2": chi2,
        "reduced_chi2": chi2 / dof,
        "success": res.success,
        "fitted_curve": predict(kf_hat, kind, N, t_data, V=V,
                                observable=observable, n0=n0, **ivp),
    }

# Notes:
# fast fitting (F = 0, single kf-proportional kernel) 
# For F = 0 with a kernel proportional to kf, kf is a pure time-rescaling:
#   n(t; kf) = n(kf*t; 1)  =>  L(t; kf) = L(kf*t; 1).
# So we integrate once with kf=1 (dense output) and fit kf as a 1-D scaling of
# time. Only valid when F = 0 and the kernel is exactly proportional to kf (constant
# or additive here). 

def build_observable_of_scaledtime(kind, N, V=1.0,
                                   observable="number_average_size",
                                   tmax_scaled=1.0, n0=None,
                                   method="RK45", rtol=1e-9, atol=1e-11):
    """
    Integrate once with kf=1 and dense output; return a callable f(tau) that
    evaluates `observable` at scaled time tau = kf*t, for tau in [0, tmax_scaled].
    tau is clipped to [0, tmax_scaled], so choose tmax_scaled >= kf_max * t_max.
    
    Parameters:
    -----------
    kind: str
        'constant' or 'additive' aggregation kernel
    N: int
        Total number of chains in the system.
    V: float
        Volume of the system
    observable: str
        What observable to return.
    tmax_scaled: float
        Maximum scaled time for the integration.
    n0: array-like | None
        Initial cluster distribution. If None, defaults to all monomers free.
    method: str
        Integration method to use. Passed to scipy.integrate.solve_ivp
    rtol: float
        Relative tolerance for the ODE solver.
    atol: float 
        Absolute tolerance for the ODE solver.

    Returns:
    --------
    f: callable
        Function f(tau) that evaluates the observable at scaled time tau = kf*t.
    """
    Kmat = kernel_matrix(kind, N, 1.0)
    rhs = _make_rhs(Kmat, N, V)
    if n0 is None:
        n0 = np.zeros(N); n0[0] = N
    sol = solve_ivp(rhs, (0.0, float(tmax_scaled)), n0, method=method,
                    dense_output=True, rtol=rtol, atol=atol)
    if not sol.success:
        raise RuntimeError(sol.message)
    obs_fn = _OBSERVABLES[observable]

    def f(tau):
        tau = np.clip(np.atleast_1d(np.asarray(tau, float)), 0.0, float(tmax_scaled))
        n = np.clip(sol.sol(tau), 0.0, None)
        return obs_fn(n)
    return f


def fit_kf_fast(t_data, y_data, kind, N, V=1.0,
                observable="number_average_size", weights=None,
                kf_bounds=(1e-3, 1e3), tmax_scaled=None, n0=None,
                rtol=1e-9, atol=1e-11):
    """
    Fast single-parameter fit of kf using the time-rescaling identity
    (F = 0, kernel proportional to kf only). Equivalent result to fit_kf but
    integrates the ODE only once.

    tmax_scaled defaults to kf_bounds[1] * max(t_data); widen kf_bounds and this
    is widened automatically. Returns kf, chi2, reduced chi2, the fitted curve,
    and the reusable scaled-time observable `f` (call f(kf*t) for any kf).

    Parameters:
    -----------
    t_data: array-like
        The time points to evaluate the ODE at
    y_data: array-like
        The observed data to fit against
    kind: str
        'constant' or 'additive' aggregation kernel
    N: int
        Total number of chains in the system.
    V: float
        Volume of the system
    observable: str
        What observable to return.
    weights: array-like | None
        Optional per-point weights (e.g. 1/sigma) for the residuals.
    kf_bounds: tuple
        Bounds for the kf parameter during fitting.
    tmax_scaled: float | None
        Maximum scaled time for the integration. If None, defaults to kf_bounds[1] * max(t_data).
    n0: array-like | None
        Initial cluster distribution. If None, defaults to all monomers free.
    rtol: float
        Relative tolerance for the ODE solver.
    atol: float
        Absolute tolerance for the ODE solver.

    Returns:
    --------
    results: dict
        Dictionary containing:
        - 'kf': fitted kf value
        - 'chi2': chi-squared value of the fit
        - 'reduced_chi2': reduced chi-squared value of the fit
        - 'success': boolean indicating if the fit was successful
        - 'fitted_curve': the fitted curve evaluated at t_data
        - 'f': callable function for the observable at scaled time (kf*t)
    """
    t_data = np.asarray(t_data, float)
    y_data = np.asarray(y_data, float)
    w = np.ones_like(y_data) if weights is None else np.asarray(weights, float)
    if tmax_scaled is None:
        tmax_scaled = kf_bounds[1] * float(t_data.max())

    f = build_observable_of_scaledtime(kind, N, V=V, observable=observable,
                                       tmax_scaled=tmax_scaled, n0=n0,
                                       rtol=rtol, atol=atol)

    def obj(kf):
        return float(np.sum((w * (f(kf * t_data) - y_data)) ** 2))

    res = minimize_scalar(obj, bounds=kf_bounds, method="bounded")
    kf_hat = float(res.x)
    chi2 = float(res.fun)
    dof = max(len(y_data) - 1, 1)
    return {
        "kf": kf_hat,
        "chi2": chi2,
        "reduced_chi2": chi2 / dof,
        "success": bool(res.success),
        "fitted_curve": f(kf_hat * t_data),
        "f": f,   # reuse for bootstrap / repeated fits without re-integrating
    }
