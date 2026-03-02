"""
rocky_core.py  (FINALIZED CORE)

What this file is
-----------------
Import-safe computational core for PROMCS/Rocky.
- Contains only classes + pure functions.
- NOTHING runs on import.
- All simulation/optimization logic is callable from promcs_engine.py.

Main pipeline supported
-----------------------
1) CRN bank build
2) Decision grids (Block 2)
3) PASS 1 component totals + survival only (Block 4 Pass 1)
4) Component Pareto menus with survival filtering for critical components (Block 4B)
5) PASS 2 cumulative histories per tau (Block 4 Pass 2 on-demand)
6) System splicing evaluator (Block 5)
7) NSGA-II per tau (Standard + Seeded) (Block 6)

Notes
-----
- Uses SciPy ONLY for Gamma PPF in CBM increments.
  SciPy import is inside the function so import-time stays light.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# ============================================================
# 0) CONFIG + COMPONENT DATA MODELS
# ============================================================

@dataclass
class SystemConfig:
    """
    System-level configuration (global parameters).
    """
    T: float
    n_traces: int
    steps_per_year: int = 12
    rng_seed: int = 42

    # Scheduled down (setup) penalties per visit
    c_SD: float = 0.0
    d_SD: float = 0.0
    e_SD: float = 0.0

    # System death penalties (applied once if system dies early)
    system_penalty_cost: float = 0.0
    system_penalty_downtime: float = 0.0
    system_penalty_emission: float = 0.0

    # Survival constraint
    enforce_survival_constraint: bool = True
    survival_target: float = 0.90

    # Major threshold fraction (repair vs replace rule)
    repair_or_replace_rule_threshold: float = 0.66

    @property
    def dt_years(self) -> float:
        return 1.0 / float(self.steps_per_year)


@dataclass
class CBMComponent:
    """
    CBM component (Gamma degradation + threshold-based PM, failure threshold H).
    """
    name: str
    quantity: int
    component_class: str
    critical: bool

    gamma_shape: float
    gamma_scale: float
    X0: float
    H: float

    rho: float

    # PM/CM penalties (cost, downtime, emissions)
    c_PMrep: float
    c_PMrepl: float
    c_CMrepl: float

    d_PMrep: float
    d_PMrepl: float
    d_CMrepl: float

    e_PMrep: float
    e_PMrepl: float
    e_CMrepl: float


@dataclass
class ABMComponent:
    """
    ABM component (Weibull lifetime + age-based PM/CM).
    """
    name: str
    quantity: int
    component_class: str
    critical: bool

    weibull_beta: float
    weibull_alpha: float
    rho: float

    c_PMrep: float
    c_PMrepl: float
    c_CMrepl: float

    d_PMrep: float
    d_PMrepl: float
    d_CMrepl: float

    e_PMrep: float
    e_PMrepl: float
    e_CMrepl: float


@dataclass
class FBMComponent:
    """
    FBM component (Weibull inter-arrival failures, CM only).
    rho retained for consistency but not used in FBM logic.
    """
    name: str
    quantity: int
    component_class: str
    critical: bool

    weibull_beta: float
    weibull_alpha: float
    rho: float

    c_CMrepl: float
    d_CMrepl: float
    e_CMrepl: float


# ============================================================
# 1) CRN BANK
# ============================================================

@dataclass
class CRNBank:
    cbm_u: Dict[str, np.ndarray]  # (n_traces, n_steps)
    abm_u: Dict[str, np.ndarray]  # (n_traces, max_events)
    fbm_u: Dict[str, np.ndarray]  # (n_traces, max_events)


def compute_n_steps(config: SystemConfig) -> int:
    return int(config.T * config.steps_per_year)


def build_crn_bank(
    config: SystemConfig,
    cbm_names: List[str],
    abm_names: List[str],
    fbm_names: List[str],
    max_events: int,
) -> CRNBank:
    n_steps = compute_n_steps(config)
    rng = np.random.default_rng(config.rng_seed)

    low, high = 0.0001, 0.9999  # avoid 0/1
    cbm_u = {name: rng.uniform(low, high, size=(config.n_traces, n_steps)) for name in cbm_names}
    abm_u = {name: rng.uniform(low, high, size=(config.n_traces, max_events)) for name in abm_names}
    fbm_u = {name: rng.uniform(low, high, size=(config.n_traces, max_events)) for name in fbm_names}

    return CRNBank(cbm_u=cbm_u, abm_u=abm_u, fbm_u=fbm_u)


# ============================================================
# 2) DECISION GRIDS (Rocky Block 2)
# ============================================================

VALID_CLASSES = {"replaceable", "repairable_replaceable", "repairable_only"}


def _normalize_component_class(value: Any) -> str:
    if value is None:
        return ""
    v = str(value).strip()
    if v == "replaceable_only":
        return "replaceable"
    return v


@dataclass
class DecisionGrids:
    cbm_control_grids: Dict[str, List[float]]
    abm_age_grids: Dict[str, List[float]]


def build_decision_grids(
    config: SystemConfig,
    cbm_components: List[CBMComponent],
    abm_components: List[ABMComponent],
) -> DecisionGrids:
    cbm_control_grids: Dict[str, List[float]] = {}
    for comp in cbm_components:
        cclass = _normalize_component_class(comp.component_class)
        if cclass in VALID_CLASSES:
            cbm_control_grids[comp.name] = [float(x) for x in np.arange(1.0, comp.H + 1.0, 1.0)]

    abm_age_grids: Dict[str, List[float]] = {}
    for comp in abm_components:
        cclass = _normalize_component_class(comp.component_class)
        if cclass in VALID_CLASSES:
            abm_age_grids[comp.name] = [round(float(x), 2) for x in np.arange(0.5, config.T + 0.5, 0.5)]

    return DecisionGrids(cbm_control_grids=cbm_control_grids, abm_age_grids=abm_age_grids)


# ============================================================
# 3) PASS 1 — totals + survival only (Block 4 Pass 1)
# ============================================================

def _consume_event_u(U_matrix: np.ndarray, life_idx: np.ndarray, mask: np.ndarray, who: str):
    if not np.any(mask):
        return None, life_idx

    life_idx[mask] += 1

    if np.any(life_idx[mask] >= U_matrix.shape[1]):
        raise RuntimeError(
            f"CRN event bank exhausted for {who}. Increase max_events (currently {U_matrix.shape[1]})."
        )

    U_new = U_matrix[mask, life_idx[mask]]
    return U_new, life_idx


def _weibull_lifetime_from_u(U: np.ndarray, eta: float, beta_w: float) -> np.ndarray:
    return eta * (-np.log(1.0 - U)) ** (1.0 / beta_w)


def _apply_pm_cbm(config: SystemConfig, comp: CBMComponent, X: np.ndarray, threshold: float, alive_mask: np.ndarray):
    dZ = np.zeros(config.n_traces)
    dV = np.zeros(config.n_traces)
    dE = np.zeros(config.n_traces)

    cclass = _normalize_component_class(comp.component_class)
    switch_limit = config.repair_or_replace_rule_threshold * comp.H

    if cclass == "repairable_replaceable":
        rep_mask = alive_mask & (X >= switch_limit)
        if np.any(rep_mask):
            dZ[rep_mask] += comp.c_PMrepl
            dV[rep_mask] += comp.d_PMrepl
            dE[rep_mask] += comp.e_PMrepl
            X[rep_mask] = 0.0

        repair_mask = alive_mask & (X >= threshold) & (~rep_mask)
        if np.any(repair_mask):
            dZ[repair_mask] += comp.c_PMrep
            dV[repair_mask] += comp.d_PMrep
            dE[repair_mask] += comp.e_PMrep
            X[repair_mask] *= comp.rho

    elif cclass == "repairable_only":
        repair_mask = alive_mask & (X >= threshold)
        if np.any(repair_mask):
            dZ[repair_mask] += comp.c_PMrep
            dV[repair_mask] += comp.d_PMrep
            dE[repair_mask] += comp.e_PMrep
            X[repair_mask] *= comp.rho

    elif cclass == "replaceable":
        rep_mask = alive_mask & (X >= threshold)
        if np.any(rep_mask):
            dZ[rep_mask] += comp.c_PMrepl
            dV[rep_mask] += comp.d_PMrepl
            dE[rep_mask] += comp.e_PMrepl
            X[rep_mask] = 0.0

    return X, dZ, dV, dE


def _apply_pm_abm(
    config: SystemConfig,
    comp: ABMComponent,
    age: np.ndarray,
    threshold: float,
    alive_mask: np.ndarray,
    U_matrix: np.ndarray,
    life_idx: np.ndarray,
    current_lifetimes: np.ndarray,
):
    dZ = np.zeros(config.n_traces)
    dV = np.zeros(config.n_traces)
    dE = np.zeros(config.n_traces)

    cclass = _normalize_component_class(comp.component_class)
    major_limit = config.repair_or_replace_rule_threshold * current_lifetimes

    if cclass == "repairable_replaceable":
        rep_mask = alive_mask & (age >= major_limit)
        if np.any(rep_mask):
            dZ[rep_mask] += comp.c_PMrepl
            dV[rep_mask] += comp.d_PMrepl
            dE[rep_mask] += comp.e_PMrepl
            age[rep_mask] = 0.0

            U_new, life_idx = _consume_event_u(U_matrix, life_idx, rep_mask, who=comp.name)
            current_lifetimes[rep_mask] = _weibull_lifetime_from_u(U_new, comp.weibull_alpha, comp.weibull_beta)

        repair_mask = alive_mask & (age >= threshold) & (~rep_mask)
        if np.any(repair_mask):
            dZ[repair_mask] += comp.c_PMrep
            dV[repair_mask] += comp.d_PMrep
            dE[repair_mask] += comp.e_PMrep
            age[repair_mask] *= comp.rho

            U_new, life_idx = _consume_event_u(U_matrix, life_idx, repair_mask, who=comp.name)
            val = (age[repair_mask] / comp.weibull_alpha) ** comp.weibull_beta - np.log(1.0 - U_new)
            current_lifetimes[repair_mask] = comp.weibull_alpha * (val ** (1.0 / comp.weibull_beta))

    elif cclass == "repairable_only":
        repair_mask = alive_mask & (age >= threshold)
        if np.any(repair_mask):
            dZ[repair_mask] += comp.c_PMrep
            dV[repair_mask] += comp.d_PMrep
            dE[repair_mask] += comp.e_PMrep
            age[repair_mask] *= comp.rho

            U_new, life_idx = _consume_event_u(U_matrix, life_idx, repair_mask, who=comp.name)
            val = (age[repair_mask] / comp.weibull_alpha) ** comp.weibull_beta - np.log(1.0 - U_new)
            current_lifetimes[repair_mask] = comp.weibull_alpha * (val ** (1.0 / comp.weibull_beta))

    elif cclass == "replaceable":
        rep_mask = alive_mask & (age >= threshold)
        if np.any(rep_mask):
            dZ[rep_mask] += comp.c_PMrepl
            dV[rep_mask] += comp.d_PMrepl
            dE[rep_mask] += comp.e_PMrepl
            age[rep_mask] = 0.0

            U_new, life_idx = _consume_event_u(U_matrix, life_idx, rep_mask, who=comp.name)
            current_lifetimes[rep_mask] = _weibull_lifetime_from_u(U_new, comp.weibull_alpha, comp.weibull_beta)

    return age, dZ, dV, dE, life_idx, current_lifetimes


def _precompute_cbm_increments_pass1(config: SystemConfig, cbm_components: List[CBMComponent], crn_bank: CRNBank):
    from scipy.stats import gamma as gamma_dist  # local import

    n_steps = compute_n_steps(config)
    incs: Dict[str, np.ndarray] = {}

    for comp in cbm_components:
        U_steps = crn_bank.cbm_u[comp.name]
        shape_param = comp.gamma_shape * config.dt_years
        incs[comp.name] = gamma_dist.ppf(U_steps, a=shape_param, scale=comp.gamma_scale)

    return incs


def simulate_CBM_totals_pass1(config: SystemConfig, comp: CBMComponent, tau: float, threshold: float, inc_matrix: np.ndarray):
    total_steps = compute_n_steps(config)

    X = np.full(config.n_traces, comp.X0, dtype=float)
    cum_Z = np.zeros(config.n_traces)
    cum_V = np.zeros(config.n_traces)
    cum_E = np.zeros(config.n_traces)

    death_t = np.full(config.n_traces, config.T, dtype=float)
    alive = np.ones(config.n_traces, dtype=bool)

    pm_interval_steps = int(float(tau) * config.steps_per_year)

    for step in range(1, total_steps + 1):
        X[alive] += inc_matrix[alive, step - 1]

        failed = alive & (X >= comp.H)
        if np.any(failed):
            if comp.critical:
                just_died = failed & (death_t == config.T)
                death_t[just_died] = step * config.dt_years
                alive[just_died] = False
            else:
                cum_Z[failed] += comp.c_CMrepl
                cum_V[failed] += comp.d_CMrepl
                cum_E[failed] += comp.e_CMrepl
                X[failed] = 0.0

        if pm_interval_steps > 0 and (step % pm_interval_steps == 0):
            X, dZ, dV, dE = _apply_pm_cbm(config, comp, X, float(threshold), alive)
            cum_Z += dZ
            cum_V += dV
            cum_E += dE

    p_surv = 1.0 if (not comp.critical) else float(np.mean(np.isclose(death_t, config.T)))
    op_time = np.where(death_t < config.T, death_t, config.T)

    return float(np.mean(cum_Z / op_time)), float(np.mean(cum_V / op_time)), float(np.mean(cum_E / op_time)), p_surv


def simulate_ABM_totals_pass1(config: SystemConfig, comp: ABMComponent, tau: float, threshold: float, crn_events: np.ndarray):
    total_steps = compute_n_steps(config)

    age = np.zeros(config.n_traces, dtype=float)
    cum_Z = np.zeros(config.n_traces)
    cum_V = np.zeros(config.n_traces)
    cum_E = np.zeros(config.n_traces)

    death_t = np.full(config.n_traces, config.T, dtype=float)
    alive = np.ones(config.n_traces, dtype=bool)

    life_idx = -np.ones(config.n_traces, dtype=int)

    init_mask = np.ones(config.n_traces, dtype=bool)
    U0, life_idx = _consume_event_u(crn_events, life_idx, init_mask, who=comp.name)
    current_lifetimes = _weibull_lifetime_from_u(U0, comp.weibull_alpha, comp.weibull_beta)

    pm_interval_steps = int(float(tau) * config.steps_per_year)

    for step in range(1, total_steps + 1):
        age[alive] += config.dt_years

        failed = alive & (age >= current_lifetimes)
        if np.any(failed):
            if comp.critical:
                just_died = failed & (death_t == config.T)
                death_t[just_died] = step * config.dt_years
                alive[just_died] = False
            else:
                cum_Z[failed] += comp.c_CMrepl
                cum_V[failed] += comp.d_CMrepl
                cum_E[failed] += comp.e_CMrepl
                age[failed] = 0.0

                U_new, life_idx = _consume_event_u(crn_events, life_idx, failed, who=comp.name)
                current_lifetimes[failed] = _weibull_lifetime_from_u(U_new, comp.weibull_alpha, comp.weibull_beta)

        if pm_interval_steps > 0 and (step % pm_interval_steps == 0):
            age, dZ, dV, dE, life_idx, current_lifetimes = _apply_pm_abm(
                config, comp, age, float(threshold), alive, crn_events, life_idx, current_lifetimes
            )
            cum_Z += dZ
            cum_V += dV
            cum_E += dE

    p_surv = 1.0 if (not comp.critical) else float(np.mean(np.isclose(death_t, config.T)))
    op_time = np.where(death_t < config.T, death_t, config.T)

    return float(np.mean(cum_Z / op_time)), float(np.mean(cum_V / op_time)), float(np.mean(cum_E / op_time)), p_surv


def simulate_FBM_totals_pass1(config: SystemConfig, comp: FBMComponent, crn_events: np.ndarray):
    total_steps = compute_n_steps(config)

    cum_Z = np.zeros(config.n_traces)
    cum_V = np.zeros(config.n_traces)
    cum_E = np.zeros(config.n_traces)

    death_t = np.full(config.n_traces, config.T, dtype=float)
    alive = np.ones(config.n_traces, dtype=bool)

    life_idx = -np.ones(config.n_traces, dtype=int)

    init_mask = np.ones(config.n_traces, dtype=bool)
    U0, life_idx = _consume_event_u(crn_events, life_idx, init_mask, who=comp.name)
    next_failure = _weibull_lifetime_from_u(U0, comp.weibull_alpha, comp.weibull_beta)

    t_track = np.zeros(config.n_traces, dtype=float)

    for step in range(1, total_steps + 1):
        t_track[alive] += config.dt_years
        failed = alive & (t_track >= next_failure)

        while np.any(failed):
            if comp.critical:
                just_died = failed & (death_t == config.T)
                death_t[just_died] = step * config.dt_years
                alive[just_died] = False
                failed = alive & (t_track >= next_failure)
                continue

            cum_Z[failed] += comp.c_CMrepl
            cum_V[failed] += comp.d_CMrepl
            cum_E[failed] += comp.e_CMrepl

            U_new, life_idx = _consume_event_u(crn_events, life_idx, failed, who=comp.name)
            next_failure[failed] += _weibull_lifetime_from_u(U_new, comp.weibull_alpha, comp.weibull_beta)

            failed = alive & (t_track >= next_failure)

    p_surv = 1.0 if (not comp.critical) else float(np.mean(np.isclose(death_t, config.T)))
    op_time = np.where(death_t < config.T, death_t, config.T)

    return float(np.mean(cum_Z / op_time)), float(np.mean(cum_V / op_time)), float(np.mean(cum_E / op_time)), p_surv


def run_pass1_component_totals(
    config: SystemConfig,
    cbm_components: List[CBMComponent],
    abm_components: List[ABMComponent],
    fbm_components: List[FBMComponent],
    tau_grid_years: List[float],
    decision_grids: DecisionGrids,
    crn_bank: CRNBank,
):
    import gc
    import time

    t0 = time.perf_counter()
    cbm_incs = _precompute_cbm_increments_pass1(config, cbm_components, crn_bank)

    CBM_RESULTS: Dict[tuple, tuple] = {}
    ABM_RESULTS: Dict[tuple, tuple] = {}
    FBM_RESULTS: Dict[tuple, tuple] = {}

    for comp in cbm_components:
        grid = decision_grids.cbm_control_grids.get(comp.name, [])
        for tau in tau_grid_years:
            for thr in grid:
                CBM_RESULTS[(comp.name, float(tau), float(thr))] = simulate_CBM_totals_pass1(
                    config=config, comp=comp, tau=float(tau), threshold=float(thr), inc_matrix=cbm_incs[comp.name]
                )

    for comp in abm_components:
        grid = decision_grids.abm_age_grids.get(comp.name, [])
        U_events = crn_bank.abm_u[comp.name]
        for tau in tau_grid_years:
            for thr in grid:
                ABM_RESULTS[(comp.name, float(tau), float(thr))] = simulate_ABM_totals_pass1(
                    config=config, comp=comp, tau=float(tau), threshold=float(thr), crn_events=U_events
                )

    FBM_ONCE: Dict[str, tuple] = {}
    for comp in fbm_components:
        U_events = crn_bank.fbm_u[comp.name]
        FBM_ONCE[comp.name] = simulate_FBM_totals_pass1(config=config, comp=comp, crn_events=U_events)

    for comp in fbm_components:
        for tau in tau_grid_years:
            FBM_RESULTS[(comp.name, float(tau))] = FBM_ONCE[comp.name]

    gc.collect()
    return CBM_RESULTS, ABM_RESULTS, FBM_RESULTS, float(time.perf_counter() - t0)


# ============================================================
# 4) COMPONENT PARETO MENUS (Block 4B)
# ============================================================

def pareto_filter_3obj(points: List[tuple]) -> List[tuple]:
    keep = []
    for i, p in enumerate(points):
        Zi, Vi, Ei = p[0], p[1], p[2]
        dominated = False
        for j, q in enumerate(points):
            if i == j:
                continue
            Zj, Vj, Ej = q[0], q[1], q[2]
            if (Zj <= Zi and Vj <= Vi and Ej <= Ei) and (Zj < Zi or Vj < Vi or Ej < Ei):
                dominated = True
                break
        if not dominated:
            keep.append(p)
    return keep


def build_component_pareto_menus_pass1(
    config: SystemConfig,
    CBM_RESULTS: Dict[tuple, tuple],
    ABM_RESULTS: Dict[tuple, tuple],
    FBM_RESULTS: Dict[tuple, tuple],
    cbm_components: List[CBMComponent],
    abm_components: List[ABMComponent],
    fbm_components: List[FBMComponent],
) -> Dict[tuple, List[dict]]:
    crit_map = {c.name: c.critical for c in cbm_components}
    crit_map.update({c.name: c.critical for c in abm_components})
    crit_map.update({c.name: c.critical for c in fbm_components})

    menus: Dict[tuple, List[dict]] = {}

    for (name, tau, thr), (Z, V, E, p_surv) in CBM_RESULTS.items():
        menus.setdefault((name, tau), []).append({"thr": thr, "Z": Z, "V": V, "E": E, "p_surv": p_surv})

    for (name, tau, thr), (Z, V, E, p_surv) in ABM_RESULTS.items():
        menus.setdefault((name, tau), []).append({"thr": thr, "Z": Z, "V": V, "E": E, "p_surv": p_surv})

    for (name, tau), (Z, V, E, p_surv) in FBM_RESULTS.items():
        menus.setdefault((name, tau), []).append({"thr": None, "Z": Z, "V": V, "E": E, "p_surv": p_surv})

    for (name, tau), opts in list(menus.items()):
        is_critical = bool(crit_map.get(name, False))

        filtered = opts
        if config.enforce_survival_constraint and is_critical:
            filtered = [o for o in opts if float(o["p_surv"]) >= float(config.survival_target)]

        pts = [(o["Z"], o["V"], o["E"], o["p_surv"], tau, o["thr"]) for o in filtered]
        nd = pareto_filter_3obj(pts)

        menus[(name, tau)] = [{"thr": thr, "Z": Z, "V": V, "E": E, "p_surv": p_surv} for (Z, V, E, p_surv, _t, thr) in nd]

    return menus


# ============================================================
# 5) PASS 2 (Block 4 Pass 2) — histories for one tau
# ============================================================

@dataclass
class HistoryBundle:
    tau: float
    cost_hist: Dict[str, Dict[Optional[float], np.ndarray]]
    down_hist: Dict[str, Dict[Optional[float], np.ndarray]]
    emis_hist: Dict[str, Dict[Optional[float], np.ndarray]]
    death_times: Dict[str, Dict[Optional[float], np.ndarray]]


def simulate_CBM_history_pass2(config: SystemConfig, comp: CBMComponent, tau: float, threshold: float, inc_matrix: np.ndarray):
    total_steps = compute_n_steps(config)

    X = np.full(config.n_traces, comp.X0, dtype=float)
    cum_Z = np.zeros(config.n_traces)
    cum_V = np.zeros(config.n_traces)
    cum_E = np.zeros(config.n_traces)

    hist_Z = np.zeros((config.n_traces, total_steps))
    hist_V = np.zeros((config.n_traces, total_steps))
    hist_E = np.zeros((config.n_traces, total_steps))

    death_t = np.full(config.n_traces, config.T, dtype=float)
    alive = np.ones(config.n_traces, dtype=bool)

    pm_interval_steps = int(float(tau) * config.steps_per_year)

    for step in range(1, total_steps + 1):
        X[alive] += inc_matrix[alive, step - 1]

        failed = alive & (X >= comp.H)
        if np.any(failed):
            if comp.critical:
                just_died = failed & (death_t == config.T)
                death_t[just_died] = step * config.dt_years
                alive[just_died] = False
            else:
                cum_Z[failed] += comp.c_CMrepl
                cum_V[failed] += comp.d_CMrepl
                cum_E[failed] += comp.e_CMrepl
                X[failed] = 0.0

        if pm_interval_steps > 0 and (step % pm_interval_steps == 0):
            X, dZ, dV, dE = _apply_pm_cbm(config, comp, X, float(threshold), alive)
            cum_Z += dZ
            cum_V += dV
            cum_E += dE

        hist_Z[:, step - 1] = cum_Z
        hist_V[:, step - 1] = cum_V
        hist_E[:, step - 1] = cum_E

    return hist_Z, hist_V, hist_E, death_t


def simulate_ABM_history_pass2(config: SystemConfig, comp: ABMComponent, tau: float, threshold: float, U_events: np.ndarray):
    total_steps = compute_n_steps(config)

    age = np.zeros(config.n_traces, dtype=float)
    cum_Z = np.zeros(config.n_traces)
    cum_V = np.zeros(config.n_traces)
    cum_E = np.zeros(config.n_traces)

    hist_Z = np.zeros((config.n_traces, total_steps))
    hist_V = np.zeros((config.n_traces, total_steps))
    hist_E = np.zeros((config.n_traces, total_steps))

    death_t = np.full(config.n_traces, config.T, dtype=float)
    alive = np.ones(config.n_traces, dtype=bool)

    life_idx = -np.ones(config.n_traces, dtype=int)

    init_mask = np.ones(config.n_traces, dtype=bool)
    U0, life_idx = _consume_event_u(U_events, life_idx, init_mask, who=comp.name)
    current_lifetimes = _weibull_lifetime_from_u(U0, comp.weibull_alpha, comp.weibull_beta)

    pm_interval_steps = int(float(tau) * config.steps_per_year)

    for step in range(1, total_steps + 1):
        age[alive] += config.dt_years

        failed = alive & (age >= current_lifetimes)
        if np.any(failed):
            if comp.critical:
                just_died = failed & (death_t == config.T)
                death_t[just_died] = step * config.dt_years
                alive[just_died] = False
            else:
                cum_Z[failed] += comp.c_CMrepl
                cum_V[failed] += comp.d_CMrepl
                cum_E[failed] += comp.e_CMrepl
                age[failed] = 0.0

                U_new, life_idx = _consume_event_u(U_events, life_idx, failed, who=comp.name)
                current_lifetimes[failed] = _weibull_lifetime_from_u(U_new, comp.weibull_alpha, comp.weibull_beta)

        if pm_interval_steps > 0 and (step % pm_interval_steps == 0):
            age, dZ, dV, dE, life_idx, current_lifetimes = _apply_pm_abm(
                config, comp, age, float(threshold), alive, U_events, life_idx, current_lifetimes
            )
            cum_Z += dZ
            cum_V += dV
            cum_E += dE

        hist_Z[:, step - 1] = cum_Z
        hist_V[:, step - 1] = cum_V
        hist_E[:, step - 1] = cum_E

    return hist_Z, hist_V, hist_E, death_t


def simulate_FBM_history_once_pass2(config: SystemConfig, comp: FBMComponent, U_events: np.ndarray):
    total_steps = compute_n_steps(config)

    cum_Z = np.zeros(config.n_traces)
    cum_V = np.zeros(config.n_traces)
    cum_E = np.zeros(config.n_traces)

    hist_Z = np.zeros((config.n_traces, total_steps))
    hist_V = np.zeros((config.n_traces, total_steps))
    hist_E = np.zeros((config.n_traces, total_steps))

    death_t = np.full(config.n_traces, config.T, dtype=float)
    alive = np.ones(config.n_traces, dtype=bool)

    life_idx = -np.ones(config.n_traces, dtype=int)

    init_mask = np.ones(config.n_traces, dtype=bool)
    U0, life_idx = _consume_event_u(U_events, life_idx, init_mask, who=comp.name)
    next_failure = _weibull_lifetime_from_u(U0, comp.weibull_alpha, comp.weibull_beta)

    t_track = np.zeros(config.n_traces, dtype=float)

    for step in range(1, total_steps + 1):
        t_track[alive] += config.dt_years
        failed = alive & (t_track >= next_failure)

        while np.any(failed):
            if comp.critical:
                just_died = failed & (death_t == config.T)
                death_t[just_died] = step * config.dt_years
                alive[just_died] = False
                failed = alive & (t_track >= next_failure)
                continue

            cum_Z[failed] += comp.c_CMrepl
            cum_V[failed] += comp.d_CMrepl
            cum_E[failed] += comp.e_CMrepl

            U_new, life_idx = _consume_event_u(U_events, life_idx, failed, who=comp.name)
            next_failure[failed] += _weibull_lifetime_from_u(U_new, comp.weibull_alpha, comp.weibull_beta)
            failed = alive & (t_track >= next_failure)

        hist_Z[:, step - 1] = cum_Z
        hist_V[:, step - 1] = cum_V
        hist_E[:, step - 1] = cum_E

    return hist_Z, hist_V, hist_E, death_t


def build_fbm_history_cache_pass2(config: SystemConfig, fbm_components: List[FBMComponent], crn_bank: CRNBank):
    cache: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
    for comp in fbm_components:
        U_events = crn_bank.fbm_u[comp.name]
        cache[comp.name] = simulate_FBM_history_once_pass2(config, comp, U_events)
    return cache


def build_histories_for_tau_pass2(
    config: SystemConfig,
    cbm_components: List[CBMComponent],
    abm_components: List[ABMComponent],
    fbm_components: List[FBMComponent],
    tau: float,
    component_pareto_menus: Dict[tuple, List[dict]],
    crn_bank: CRNBank,
    cbm_increments: Optional[Dict[str, np.ndarray]] = None,
    fbm_cache: Optional[Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]] = None,
) -> HistoryBundle:
    tau = float(tau)

    if cbm_increments is None:
        cbm_increments = _precompute_cbm_increments_pass1(config, cbm_components, crn_bank)

    if fbm_cache is None:
        fbm_cache = build_fbm_history_cache_pass2(config, fbm_components, crn_bank)

    all_components = cbm_components + abm_components + fbm_components

    cost_hist: Dict[str, Dict[Optional[float], np.ndarray]] = {c.name: {} for c in all_components}
    down_hist: Dict[str, Dict[Optional[float], np.ndarray]] = {c.name: {} for c in all_components}
    emis_hist: Dict[str, Dict[Optional[float], np.ndarray]] = {c.name: {} for c in all_components}
    death_times: Dict[str, Dict[Optional[float], np.ndarray]] = {c.name: {} for c in all_components}

    for comp in cbm_components:
        opts = component_pareto_menus.get((comp.name, tau), [])
        for o in opts:
            thr = float(o["thr"])
            hz, hv, he, dt = simulate_CBM_history_pass2(config, comp, tau, thr, cbm_increments[comp.name])
            cost_hist[comp.name][thr] = hz
            down_hist[comp.name][thr] = hv
            emis_hist[comp.name][thr] = he
            death_times[comp.name][thr] = dt

    for comp in abm_components:
        opts = component_pareto_menus.get((comp.name, tau), [])
        U_events = crn_bank.abm_u[comp.name]
        for o in opts:
            thr = float(o["thr"])
            hz, hv, he, dt = simulate_ABM_history_pass2(config, comp, tau, thr, U_events)
            cost_hist[comp.name][thr] = hz
            down_hist[comp.name][thr] = hv
            emis_hist[comp.name][thr] = he
            death_times[comp.name][thr] = dt

    for comp in fbm_components:
        hz, hv, he, dt = fbm_cache[comp.name]
        cost_hist[comp.name][None] = hz
        down_hist[comp.name][None] = hv
        emis_hist[comp.name][None] = he
        death_times[comp.name][None] = dt

    return HistoryBundle(tau=tau, cost_hist=cost_hist, down_hist=down_hist, emis_hist=emis_hist, death_times=death_times)


# ============================================================
# 6) SYSTEM SPLICING EVALUATOR (Block 5)
# ============================================================

def evaluate_system_spliced_from_histories(
    config: SystemConfig,
    tau: float,
    thresholds_dict: Dict[str, Optional[float]],
    pass2_bundle: HistoryBundle,
    cbm_components: List[CBMComponent],
    abm_components: List[ABMComponent],
    fbm_components: List[FBMComponent],
) -> Tuple[float, float, float, float]:
    tau = float(tau)
    n_steps = compute_n_steps(config)
    trace_idx = np.arange(config.n_traces)

    qty_map = {c.name: int(c.quantity) for c in (cbm_components + abm_components + fbm_components)}
    crit_map = {c.name: bool(c.critical) for c in (cbm_components + abm_components + fbm_components)}

    death_times = np.full(config.n_traces, config.T, dtype=float)
    has_critical = False

    for comp_name, thr in thresholds_dict.items():
        if crit_map.get(comp_name, False):
            has_critical = True
            dt_arr = pass2_bundle.death_times[comp_name][thr]
            death_times = np.minimum(death_times, np.asarray(dt_arr, dtype=float))

    p_surv_sys = 1.0 if (not has_critical) else float(np.mean(np.isclose(death_times, config.T)))

    death_indices = (death_times / config.dt_years).astype(int) - 1
    death_indices = np.clip(death_indices, 0, n_steps - 1)

    sys_Z = np.zeros(config.n_traces, dtype=float)
    sys_V = np.zeros(config.n_traces, dtype=float)
    sys_E = np.zeros(config.n_traces, dtype=float)

    for comp_name, thr in thresholds_dict.items():
        q = int(qty_map.get(comp_name, 1))
        hz = pass2_bundle.cost_hist[comp_name][thr]
        hv = pass2_bundle.down_hist[comp_name][thr]
        he = pass2_bundle.emis_hist[comp_name][thr]
        sys_Z += q * hz[trace_idx, death_indices]
        sys_V += q * hv[trace_idx, death_indices]
        sys_E += q * he[trace_idx, death_indices]

    died_mask = death_times < config.T
    sys_Z[died_mask] += config.system_penalty_cost
    sys_V[died_mask] += config.system_penalty_downtime
    sys_E[died_mask] += config.system_penalty_emission

    n_visits = np.floor(death_times / max(tau, 1e-12))
    sys_Z += n_visits * config.c_SD
    sys_V += n_visits * config.d_SD
    sys_E += n_visits * config.e_SD

    denom = np.maximum(death_times, 1e-9)
    Z_sys = float(np.mean(sys_Z / denom))
    V_sys = float(np.mean(sys_V / denom))
    E_sys = float(np.mean(sys_E / denom))

    if config.enforce_survival_constraint and (p_surv_sys < float(config.survival_target)):
        return float("inf"), float("inf"), float("inf"), p_surv_sys

    return Z_sys, V_sys, E_sys, p_surv_sys


# ============================================================
# 7) NSGA-II per tau (Block 6) — Standard + Seeded
# ============================================================

def _thr_equal(a: Any, b: Any) -> bool:
    if a is None and b is None:
        return True
    if (a is None) != (b is None):
        return False
    try:
        return bool(np.isclose(float(a), float(b), atol=1e-12, rtol=0.0))
    except Exception:
        return a == b


def fast_non_dominated_sort_vectorized(objectives: np.ndarray) -> List[List[int]]:
    n = objectives.shape[0]
    diffs = objectives[:, np.newaxis, :] - objectives[np.newaxis, :, :]
    dominates = (np.all(diffs <= 0, axis=2)) & (np.any(diffs < 0, axis=2))

    domination_count = np.sum(dominates, axis=0)
    dominated_solutions = [np.where(dominates[p])[0].tolist() for p in range(n)]

    fronts: List[List[int]] = []
    current_front = np.where(domination_count == 0)[0].tolist()

    while current_front:
        fronts.append(current_front)
        next_front: List[int] = []
        for p in current_front:
            for q in dominated_solutions[p]:
                domination_count[q] -= 1
                if domination_count[q] == 0:
                    next_front.append(q)
        current_front = next_front

    return fronts


def calculate_crowding_distance(objectives: np.ndarray, front: List[int]) -> np.ndarray:
    n = len(front)
    if n == 0:
        return np.array([])

    distance = np.zeros(n, dtype=float)
    f_idx = np.array(front, dtype=int)

    for m in range(objectives.shape[1]):
        vals = objectives[f_idx, m]
        order = np.argsort(vals)

        distance[order[0]] = np.inf
        distance[order[-1]] = np.inf

        rng = vals[order[-1]] - vals[order[0]]
        if rng == 0:
            continue

        for i in range(1, n - 1):
            distance[order[i]] += (vals[order[i + 1]] - vals[order[i - 1]]) / rng

    return distance


def run_nsga2_for_tau(
    config: SystemConfig,
    tau: float,
    comp_names: List[str],
    component_pareto_options: Dict[tuple, List[dict]],
    cbm_components: List[CBMComponent],
    abm_components: List[ABMComponent],
    fbm_components: List[FBMComponent],
    crn_bank: CRNBank,
    wsm_seeds: Optional[List[dict]] = None,
    seed: bool = False,
    use_cache: bool = True,
    pop_size: int = 500,
    num_gens: int = 500,
    elite_target: int = 250,
    tournament_k: int = 50,
    offspring_n: int = 200,
    mut_rate: float = 0.10,
) -> Tuple[List[dict], float]:
    import time
    import random
    import ast

    t0 = time.perf_counter()
    tau = float(tau)

    comp_paretos = {name: component_pareto_options.get((name, tau), []) for name in comp_names}
    if any(len(comp_paretos[n]) == 0 for n in comp_names):
        return [], float(time.perf_counter() - t0)

    # PASS2 histories for this tau
    tau_bundle = build_histories_for_tau_pass2(
        config=config,
        cbm_components=cbm_components,
        abm_components=abm_components,
        fbm_components=fbm_components,
        tau=tau,
        component_pareto_menus=component_pareto_options,
        crn_bank=crn_bank,
    )

    eval_cache: Dict[tuple, tuple] = {} if use_cache else {}
    chrom_len = len(comp_names)

    random.seed(config.rng_seed)

    def eval_system_from_chrom(chrom: List[int]):
        key = tuple(chrom)
        if use_cache and (key in eval_cache):
            Z_sys, V_sys, E_sys, p_surv = eval_cache[key]
            return [Z_sys, V_sys, E_sys], p_surv

        thresholds = {name: comp_paretos[name][chrom[i]]["thr"] for i, name in enumerate(comp_names)}
        Z_sys, V_sys, E_sys, p_surv = evaluate_system_spliced_from_histories(
            config=config,
            tau=tau,
            thresholds_dict=thresholds,
            pass2_bundle=tau_bundle,
            cbm_components=cbm_components,
            abm_components=abm_components,
            fbm_components=fbm_components,
        )

        if use_cache:
            eval_cache[key] = (Z_sys, V_sys, E_sys, p_surv)

        return [Z_sys, V_sys, E_sys], p_surv

    # -------------------------
    # Initialization
    # -------------------------
    population: List[List[int]] = []

    if seed and wsm_seeds:
        seeds_tau = [s for s in wsm_seeds if float(s.get("Tau", -1)) == tau]
        for sol in seeds_tau:
            thrs_raw = sol.get("Thresholds", {})
            if isinstance(thrs_raw, str):
                try:
                    seed_thrs = ast.literal_eval(thrs_raw)
                except Exception:
                    seed_thrs = eval(thrs_raw)
            else:
                seed_thrs = thrs_raw

            chrom = []
            for name in comp_names:
                target_thr = seed_thrs.get(name, None)
                opts = comp_paretos[name]
                idx = 0
                for k, o in enumerate(opts):
                    if _thr_equal(o["thr"], target_thr):
                        idx = k
                        break
                chrom.append(idx)

            if chrom not in population:
                population.append(chrom)

    if len(population) > pop_size:
        population = random.sample(population, pop_size)

    while len(population) < pop_size:
        population.append([random.randrange(len(comp_paretos[name])) for name in comp_names])

    # -------------------------
    # Evolution
    # -------------------------
    for _gen in range(num_gens):
        objs = np.zeros((pop_size, 3), dtype=float)

        for i, ind in enumerate(population):
            (Z_sys, V_sys, E_sys), _p = eval_system_from_chrom(ind)
            objs[i, :] = [Z_sys, V_sys, E_sys]

        fronts = fast_non_dominated_sort_vectorized(objs)

        ranks = np.zeros(pop_size, dtype=int)
        for f_idx, front in enumerate(fronts):
            for idx in front:
                ranks[idx] = f_idx

        # elitism
        elite_idx: List[int] = []
        for front in fronts:
            if len(elite_idx) + len(front) <= elite_target:
                elite_idx.extend(front)
            else:
                dists = calculate_crowding_distance(objs, front)
                sorted_front = [x for _, x in sorted(zip(dists, front), reverse=True)]
                elite_idx.extend(sorted_front[: elite_target - len(elite_idx)])
                break
        elites = [population[i] for i in elite_idx]

        # tournament selection (size 2)
        tournament: List[List[int]] = []
        for _ in range(tournament_k):
            i1, i2 = random.sample(range(pop_size), 2)
            if ranks[i1] < ranks[i2]:
                winner = i1
            elif ranks[i2] < ranks[i1]:
                winner = i2
            else:
                winner = random.choice([i1, i2])
            tournament.append(population[winner])

        parents = elites + tournament

        # crossover (two-point)
        offspring: List[List[int]] = []
        for _ in range(offspring_n):
            p1, p2 = random.sample(parents, 2)
            pt1, pt2 = sorted(random.sample(range(chrom_len + 1), 2))
            child = p1[:pt1] + p2[pt1:pt2] + p1[pt2:]
            offspring.append(child)

        # mutation
        for child in offspring:
            if random.random() < mut_rate:
                m = random.randrange(chrom_len)
                child[m] = random.randrange(len(comp_paretos[comp_names[m]]))

        population = elites + tournament + offspring

    # -------------------------
    # Extract best front
    # -------------------------
    final_objs = np.zeros((pop_size, 3), dtype=float)
    final_surv = np.zeros(pop_size, dtype=float)

    for i, ind in enumerate(population):
        (Z_sys, V_sys, E_sys), p_surv = eval_system_from_chrom(ind)
        final_objs[i, :] = [Z_sys, V_sys, E_sys]
        final_surv[i] = p_surv

    best_front = fast_non_dominated_sort_vectorized(final_objs)[0]

    seen = set()
    sols: List[dict] = []

    for idx in best_front:
        Z_sys, V_sys, E_sys = final_objs[idx]
        p_surv = float(final_surv[idx])

        if not (np.isfinite(Z_sys) and np.isfinite(V_sys) and np.isfinite(E_sys)):
            continue

        chrom = population[idx]
        thrs = {name: comp_paretos[name][chrom[j]]["thr"] for j, name in enumerate(comp_names)}

        key = (tau, round(float(Z_sys), 10), round(float(V_sys), 10), round(float(E_sys), 10), str(thrs))
        if key in seen:
            continue
        seen.add(key)

        sols.append({
            "Tau": tau,
            "Z_sys": float(Z_sys),
            "V_sys": float(V_sys),
            "E_sys": float(E_sys),
            "p_surv_sys": p_surv,
            "Thresholds": thrs
        })

    return sols, float(time.perf_counter() - t0)