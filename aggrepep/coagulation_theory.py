# this file contains functions related to
# coagulation theory and the Smoluchowski equation, 
# which describes the kinetics of cluster formation
from sklearn.linear_model import LinearRegression

from aggrepep.clustering_analysis import compute_mu_i_t, compute_moment_i_of_cluster_size_distribution

import numpy as np
from pathlib import Path
import MDAnalysis as mda

def load_replicates(base_dir, system="cg", topology="solvated.gro", trajectory="prod.xtc"):
    """
    Load MDAnalysis Universes for all replicates under base_dir/system/.

    Parameters
    ----------
    base_dir : str or Path
        Path to the peptide directory (e.g. 'pepid').
    system : str
        Subdirectory name (e.g. 'cg', 'aa').
    topology : str
        Topology filename inside each replica directory.
    trajectory : str
        Trajectory filename inside each replica directory.

    Yields
    ------
    (replica_name, Universe)
    """
    base = Path(base_dir) / system
    if not base.is_dir():
        raise FileNotFoundError(f"{base} not found")

    for replica_dir in sorted(base.iterdir()):
        if not replica_dir.is_dir():
            continue
        top = replica_dir / topology
        traj = replica_dir / trajectory
        if not (top.exists() and traj.exists()):
            print(f"Skipping {replica_dir.name}: missing files")
            continue
        u = mda.Universe(str(top), str(traj))
        yield replica_dir.name, u

def ci_analytic_solution(c, p, i=1):
    """
    Eqn. 31 in https://doi.org/10.1021/acs.jpcb.3c02884?urlappend=%3Fref%3DPDF&jav=VoR&rel=cite-as
    
    Parameters:
    -----------
    c : float
        initial concentration of MONODISPERSE system (everything starts in monomers)
    p : float
        equation 32 results
    i : int
        cluster size
    """
    
    return c*( (1-p)**2 ) * p**(i-1)

def p_equation(t, c, kf):
    """
    eq. 32 in https://doi.org/10.1021/acs.jpcb.3c02884?urlappend=%3Fref%3DPDF&jav=VoR&rel=cite-as
    
    Parameters:
    -----------
    t : float
        time (ns)
    c : float
        initial concentration of MONODISPERSE system (i.e. everything starts in monomers)
    kf: aggregation probability parameter to fit for
    """
    _numer = c*kf*t
    _denom = c*kf*t + 2
    return _numer/_denom

def number_average_cluster_size_analytic(t,c,kf,n_chains=75, V=15*15*15):
    """
    Compute the number-average cluster size
    \frac{ \sum_r r n_r(t) }{ \sum_r n_r(t) }
    i.e. M_1(t)/M_0(t)
    
    Parameters:
    -----------
    t: float
        time
    c: float
        initial concentration of monodisperse system
    kf: float
        aggregation parameter, to be fitted for
    n_chains: int
        total number of chains used in system

    Returns:
    --------
    n_avg_cluster_size
    """
    Navogadro = 1#6.023e23
    n_avg_cluster_size = 0
    pval = p_equation(t, c, kf)
    M_0t = 0
    M_1t = 0
    for r in range(1,n_chains+1,1):
        _pval = p_equation(t, c, kf)

        # c divides out in the equation, so we can set it to 1.0 for simplicity
        _ci_without_c = ((1-_pval)**2) * (_pval**(r-1))

        M_0t += _ci_without_c
        M_1t += r*_ci_without_c
    return M_1t/M_0t

def param_dependent_n_avg_cluster_analytic(kf, 
                                           c, 
                                           box_volume=15*15*15, 
                                           N_frames=1000, 
                                           frames_per_ns=int(1000/100)
                                           ):
    """
    Compute analytic estimates of the number-average cluster size for a range of times.
    
    Parameters:
    -----------
    kf: float
        aggregation parameter, to be fitted for
    c: float
        initial concentration of monodisperse system
    box_volume: float
        volume of the simulation box (nm^3)
    N_frames: int
        number of time frames to compute over
    frames_per_ns: int
        number of frames per nanosecond (e.g. 1000 frames/ns for 1 frame/ps)

    Returns:
    --------
    analytic_estimates: np.array
        array of length N_frames containing the analytic estimates
        of the number-average cluster size at each time point
    """
    analytic_estimates = np.zeros((N_frames,))
    for _frame in range(0,N_frames,1):
        _t = _frame*(1/frames_per_ns)
        analytic_estimates[_frame] = number_average_cluster_size_analytic(_t, c, kf,V=box_volume)
    return analytic_estimates

def fit_treat1990_to_coagulation_results(tvals,
                                         coagulation_results, 
                                         N_chains=75, 
                                         box_volume=15*15*15,
                                         assume_monodisperse_initial=True
                                        ):
    """
    Fits the Treat 1990 model to the coagulation results 
    by performing a linear regression.
    
    Parameters:
    -----------
    tvals: array-like
        Array of time values corresponding to the coagulation results.
    coagulation_results: array-like
        Array of coagulation results (e.g. number-average cluster size) at the corresponding time values.
    N_chains: int
        Total number of chains in the system (used for scaling the coefficient).
    box_volume: float
        Volume of the simulation box (nm^3), used for scaling the coefficient.
    assume_monodisperse_initial: bool
        If True, assumes the initial condition is monodisperse (all monomers), 
        which implies that the intercept of the linear regression should be 1. 
        If False, the intercept is allowed to vary freely.

    Returns:
    --------
    K_Treat1990: float
        The fitted aggregation rate constant from the Treat 1990 model
    intercept: float
        The intercept of the linear regression (should be 1 if assume_monodisperse_initial is True)
    r_squared: float
        The coefficient of determination (R^2) for the linear regression fit, indicating
        how well the model explains the variance in the coagulation results.

    Notes:
    Treat (1990): J. Phys. A: Math. Gen. 23 (1990) 3003-3016
    Finds exact solution to Smoluchowski equation for constant kernel,
    and zero fragmentation, giving:
    
    M_1(t)/M_0(t) = 1 + 0.5 * N_chains * K_Treat1990 * t
    aka
    mu_1(t) = mu_1(0) + 0.5 * K_Treat1990 * M_1 * t

    where M_1(t)/M_0(t) is the number-average cluster size at time t,
    N_chains is the total number of chains in the system, and
    K_Treat1990 is the aggregation rate constant
    """
    if isinstance(tvals, list):
        tvals = np.array(tvals).reshape(-1,1)
    if isinstance(coagulation_results, list):
        coagulation_results = np.array(coagulation_results).reshape(-1)
    
    if assume_monodisperse_initial:
        _reg = LinearRegression(fit_intercept=False)
    else:
        _reg = LinearRegression(fit_intercept=True) 
    
    if assume_monodisperse_initial:
        _yvals = (coagulation_results - 1).reshape(-1,1)
    else:
        _yvals = coagulation_results.reshape(-1,1)
    
    _reg.fit(
        tvals, 
        _yvals
    )

    # 
    K_Treat1990 = _reg.coef_*2/(N_chains/box_volume)
    if assume_monodisperse_initial:
        intercept = 1
    else:
        intercept = _reg.intercept_

    # compute R^2=1 - (SS_res / SS_tot)
    y_pred = _reg.predict(tvals)
    ss_res = np.sum((_yvals - y_pred)**2)
    ss_tot = np.sum((_yvals - np.mean(_yvals))**2)
    r_squared = 1 - (ss_res / ss_tot)

    return K_Treat1990, intercept, r_squared

def fit_szalamendyk2023_to_coagulation_results(tvals,
                                         coagulation_results, 
                                         initial_concentration=75/(15*15*15), 
                                         assume_monodisperse_initial=True
                                        ):
    """
    Fits Szała-Mendyk et al. (2023) exact solution to the coagulation results.
    This is a nonlinear model, so we use scipy.optimize minimize
    with method="Nelder-Mead" to fit the parameter.

    Parameters:
    -----------
    tvals: array-like
        Array of time values corresponding to the coagulation results.
    coagulation_results: array-like
        Array of coagulation results (e.g. number-average cluster size, mu_1(t) ) 
        at the corresponding time values.
    initial_concentration: float
        The initial concentration of monomers in the system.
        75 chains / (15 nm)^3 box volume = 0.0222 chains/nm^3
    assume_monodisperse_initial: bool
        If True, assumes the initial condition is monodisperse (all monomers),
    
    Returns:
    --------
    kf: float
        The fitted aggregation rate constant from the Szała-Mendyk et al. (2023) model
    r_squared: float
        The coefficient of determination (R^2) for the fit, 
        indicating how well the model explains the variance 
        in the coagulation results.

    Notes:
    --------
    Szała-Mendyk et al. (2023): https://doi.org/10.1021/acs.jpcb.3c02884
    Investigates a modified Smoluchowski equation with a fixed number of chains.
    This is different from the classical Smoluchowski equation, which
    assumes an infinite reservoir of monomers.

    We leverage equations 31 and 32 from the paper, which
    give an exact solution assuming a monodisperse initial condition,
    a constant aggregation kernel, and zero fragmentation. 
    """
    from scipy.optimize import minimize

    def objective(kf):
        # compute the analytic estimates for the given kf
        analytic_estimates = param_dependent_n_avg_cluster_analytic(kf, c=initial_concentration, box_volume=15*15*15, N_frames=len(tvals), frames_per_ns=int(1000/100))
        # compute the mean squared error between the analytic estimates and the coagulation results
        mse = np.mean((analytic_estimates - coagulation_results)**2)
        return mse

    # initial guess for kf
    kf_initial = 0.01
    result = minimize(objective, kf_initial, method="Nelder-Mead")
    kf_fitted = result.x[0]

    # compute R^2 for the fit
    analytic_estimates_fitted = param_dependent_n_avg_cluster_analytic(
        kf_fitted, 
        c=initial_concentration, 
        box_volume=15*15*15, 
        N_frames=len(tvals), 
        frames_per_ns=int(1000/100)
    )
    ss_res = np.sum((analytic_estimates_fitted - coagulation_results)**2)
    ss_tot = np.sum((coagulation_results - np.mean(coagulation_results))**2)
    r_squared = 1 - (ss_res / ss_tot)

    return kf_fitted, r_squared

def analyze_aggregation_trajectory(universe, sequence, frames_per_ns=10, params=None, verbose=False):
    """
    Analyze an MD trajectory to compute the number-average cluster size over time,
    and fit the Treat 1990 and Szała-Mendyk 2023 models to the results.

    Parameters:
    -----------
    universe: MDAnalysis Universe
        The MD trajectory to analyze.
    sequence: str
        The peptide sequence, used for identifying chains in the trajectory.
    frames_per_ns: int
        The number of frames per nanosecond in the trajectory (e.g. 10 frames/ns for 1 frame/100 ps)
    params: dict
        A dictionary of parameters for the analysis.
    verbose: bool
        If True, print detailed results for each replicate.
        
    Returns:
    --------
    results: dict
        A dictionary containing the time values, number-average cluster size values,
        and fitted parameters for the Treat 1990 and Szała-Mendyk 2023 models.

        {
            "tvals": np.array of time values (ns),
            "initial_concentration": initial concentration of monomers (chains/nm^3),
            "n_avg_cluster_size": np.array of number-average cluster size values,
            "Treat1990": {
                "K_Treat1990": fitted aggregation rate constant,
                "intercept": intercept of linear regression,
                "r_squared": R^2 value for the fit
            },
            "SzalaMendyk2023": {
                "kf": fitted aggregation rate constant,
                "r_squared": R^2 value for the fit
            }
        }
    """


    # Top-level storage
    results = {
        "tvals": None,
        "initial_concentration": None,
        "n_avg_cluster_size": None,
        "Treat1990": None,
        "SzalaMendyk2023": None,
        "notes":"Concentration in chains/nm^3, "
    }
    coagulation_results = {}
    ktreat_summary = {}   # mjid -> (mean, std)
    kSM_summary = {}      # mjid -> (mean, std)

    seq = sequence
    _uni=universe
    n_chains = len(_uni.select_atoms("not name W WP WM NA+ CL-").residues) // len(seq)

    if params:
        volume_nm3 = params.get("volume_nm3", 15*15*15)
    else:
        volume_nm3 = 15*15*15
    initial_concentration = n_chains/volume_nm3 # 75 chains in 15nm^3 box

    xframes = np.arange(0, len(_uni.trajectory), 1)
    tvals = xframes / frames_per_ns

    # coagulation_results = {}
    ktreat_results = {}
    kSM_results = {}

    protein_atoms = _uni.select_atoms("not name W WP WM NA+ CL-")
    chain_groups = [
        protein_atoms.residues[len(seq)*i:len(seq)*(i+1)]
        for i in range(n_chains)
    ]

    yvals = [compute_mu_i_t(_uni, chain_groups, _frame) for _frame in xframes]
    coagulation_results = yvals

    ############################################
    ### Fit aggregation rate parameter
    ############################################
    
    # fit using Treat (1990)
    ktreat, treat_intercept, r2_treat = fit_treat1990_to_coagulation_results(
        tvals, yvals, 
        N_chains=n_chains
    )
    ktreat = ktreat.flatten()[0]
    ktreat_results = (ktreat, treat_intercept, r2_treat)

    # fit using Szala-mendyk et al. (2023)
    k_SM, r2_SM = fit_szalamendyk2023_to_coagulation_results(
        tvals, yvals, 
        initial_concentration=initial_concentration
    )
    kSM_results = (k_SM, r2_SM)

    # fill in results dictionary
    results["tvals"] = tvals
    results["initial_concentration"] = initial_concentration
    results["n_avg_cluster_size"] = yvals
    results["Treat1990"] = {
        "K_Treat1990": ktreat,
        "intercept": treat_intercept,
        "r_squared": r2_treat
    }
    results["SzalaMendyk2023"] = {
        "kf": k_SM,
        "r_squared": r2_SM
    }

    return results

    # Aggregate across replicates for this mjid
    # ktreat_vals = np.array([v[0] for v in ktreat_results.values()])
    # kSM_vals    = np.array([v[0] for v in kSM_results.values()])

    # ktreat_mean , ktreat_std = ktreat_vals.mean(), ktreat_vals.std(ddof=1)
    # kSM_mean    , kSM_std    =    kSM_vals.mean(),    kSM_vals.std(ddof=1)
    
    # ktreat_summary[mjid] = (ktreat_mean, ktreat_std)
    # kSM_summary[mjid] = (kSM_mean, kSM_std)
    
    # ktreat_mean, ktreat_std = ktreat_summary[mjid]
    # kSM_mean, kSM_std = kSM_summary[mjid]
    
    # if verbose:
    #     print(f"=== {mjid} (n={len(ktreat_vals)}) ===")
    #     print(f"  ktreat: mean = {ktreat_mean:.6g}, std = {ktreat_std:.6g}")
    #     print(f"  k_SM:   mean = {kSM_mean:.6g}, std = {kSM_std:.6g}")

    # # --- Final summary table ---
    # if verbose:
    #     print("\n=== Summary across all mjids ===")
    #     print(f"{'mjid':<6} {'ktreat_mean':>12} {'ktreat_std':>12} {'kSM_mean':>12} {'kSM_std':>12}")
    #     for mjid in mjids:
    #         kt_m, kt_s = ktreat_summary[mjid]
    #         ks_m, ks_s = kSM_summary[mjid]
    #         print(f"{mjid:<6} {kt_m:>12.6g} {kt_s:>12.6g} {ks_m:>12.6g} {ks_s:>12.6g}")