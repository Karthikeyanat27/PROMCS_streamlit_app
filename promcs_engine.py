"""
promcs_engine.py  (FINALIZED EXPORT VERSION)

BLOCK DESCRIPTION
-----------------
Engine layer called by Streamlit (PROMCS). This version is designed for FINAL tool export.

INPUTS
------
run_promcs(model_inputs: Dict[str, Any], output_dir: str = "outputs")

model_inputs must contain:
- time_horizon_years (int/float)
- n_traces (int)
- crn_bank_size (int)                         # max_events in CRN bank
- survival_probability (float in [0,1])       # system survival target
- tau_years_list (List[float])               # taus in years
- sd_cost, sd_downtime, sd_emissions (float)
- system_penalty_cost, system_penalty_downtime, system_penalty_emissions (float)
- p_major_threshold (float)
- cbm_df, abm_df, fbm_df (pd.DataFrame)

Optional:
- run_wsm (bool)
- run_ga (bool)
- run_sga (bool)

OUTPUTS
-------
EngineResult:
- excel_path : overwritten each run
- plots_dir  : placeholder folder (download ZIP includes it)
- message    : status message
- summary    : key metrics

WHAT THIS ENGINE DOES
---------------------
1) Parse Streamlit inputs -> SystemConfig + component objects
2) Build CRN bank + decision grids
3) PASS 1: component totals (all tau, all thresholds)
4) Build component Pareto menus (with survival filtering for critical comps)
5) System baseline KPI per tau (first menu option per component at that tau)
6) WSM enumeration (3^k) -> system Pareto per tau (optional)
7) GA + Seeded GA using rocky_core.run_nsga2_for_tau (optional)
8) Export a complete Excel report with:
   - Inputs + grids + PASS1 totals
   - Component Pareto menus + Optima sheets (CBM/ABM/FBM)
   - System baseline per tau
   - WSM / GA / Seeded GA results
   - Per_Tau_Extremes (system-level extremes per tau per method)
   - Detailed_Extremes (global extremes per method)
   - Final_Report_2 (your final KPI list)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import os
import time
import itertools
import ast

import pandas as pd
import numpy as np

# Rocky core imports (must exist in rocky_core.py)
from rocky_core import (
    SystemConfig,
    CBMComponent,
    ABMComponent,
    FBMComponent,
    build_crn_bank,
    build_decision_grids,
    compute_n_steps,
    run_pass1_component_totals,
    build_component_pareto_menus_pass1,
    build_histories_for_tau_pass2,
    run_nsga2_for_tau,
)


# ============================
# Result container for Streamlit
# ============================
@dataclass
class EngineResult:
    excel_path: str
    plots_dir: str
    message: str
    summary: Dict[str, Any]


# ============================
# Helpers: UI normalization / parsing
# ============================
def _criticality_to_bool(val: Any) -> bool:
    """Convert UI criticality labels to boolean."""
    if isinstance(val, bool):
        return val
    s = str(val).strip().lower()
    return s in ("critical", "true", "1", "yes", "y")


def _first_existing_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    """Return the first column name from candidates that exists in df, else None."""
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _require_any(df: pd.DataFrame, candidates: List[str], field_name: str, table_name: str) -> str:
    """
    Require that at least one of candidate columns exists.
    Returns the actual column name that exists.
    """
    found = _first_existing_col(df, candidates)
    if found is None:
        raise ValueError(f"{table_name} table: missing column for '{field_name}'. Expected one of: {candidates}")
    return found


def _clean_df(df: pd.DataFrame) -> pd.DataFrame:
    """Copy df and drop helper columns that should not go to engine (like Delete)."""
    out = df.copy()
    if "Delete" in out.columns:
        out = out.drop(columns=["Delete"])
    return out


def _ensure_output_dirs(output_dir: str) -> tuple[str, str]:
    """Ensure outputs folder exists and return (excel_path, plots_dir)."""
    abs_out = os.path.abspath(output_dir)
    os.makedirs(abs_out, exist_ok=True)

    # Overwrite each run (as you requested)
    excel_path = os.path.join(abs_out, "PROMCS_placeholder.xlsx")
    plots_dir = os.path.join(abs_out, "PROMCS_placeholder_plots")
    os.makedirs(plots_dir, exist_ok=True)

    return excel_path, plots_dir


def _parse_thresholds(value: Any) -> Dict[str, Any]:
    """
    Parse Thresholds that may be stored as:
    - dict
    - string representation of dict
    """
    if isinstance(value, dict):
        return value
    if value is None:
        return {}
    s = str(value).strip()
    if s == "":
        return {}
    try:
        return ast.literal_eval(s)
    except Exception:
        # fallback (should rarely happen)
        return eval(s)


# ============================
# DataFrame -> Component objects
# ============================
def _df_to_cbm_components(df: pd.DataFrame) -> List[CBMComponent]:
    if df is None or df.empty:
        return []
    df = _clean_df(df)

    name_col = _require_any(df, ["Name", "name"], "Name", "CBM")
    qty_col = _require_any(df, ["Qty", "quantity", "Quantity"], "Qty", "CBM")
    class_col = _require_any(df, ["Class", "class"], "Class", "CBM")
    crit_col = _require_any(df, ["Criticality", "criticality"], "Criticality", "CBM")

    g_shape_col = _require_any(df, ["Gamma shape (k)", "Gamma shape", "Gamma shape (α)"], "Gamma shape", "CBM")
    g_scale_col = _require_any(df, ["Gamma scale (eta)", "Gamma scale (η)", "Gamma scale"], "Gamma scale", "CBM")
    x0_col = _require_any(df, ["Initial Level (x_i)", "Initial Level (xi)", "Initial Level", "x_i", "X0"], "Initial level", "CBM")
    h_col = _require_any(df, ["Failure threshold (H_i)", "Failure threshold", "H_i", "H"], "Failure threshold", "CBM")

    rho_col = _require_any(df, ["rho (Kijima)", "rho", "Kijima", "Kijima rho"], "rho (Kijima)", "CBM")

    pm_rep_cost = _require_any(df, ["PM Rep cost (EUR)", "PM Rep cost (€)", "PR cost (€)", "PR cost (EUR)"], "PM rep cost", "CBM")
    pm_repl_cost = _require_any(df, ["PM Repl cost (EUR)", "PM Repl cost (€)", "PX cost (€)", "PX cost (EUR)"], "PM repl cost", "CBM")
    cm_repl_cost = _require_any(df, ["CM Repl cost (EUR)", "CM Repl cost (€)", "CX cost (€)", "CX cost (EUR)"], "CM repl cost", "CBM")

    pm_rep_down = _require_any(df, ["PM Rep down (h)", "PR down (h)"], "PM rep down", "CBM")
    pm_repl_down = _require_any(df, ["PM Repl down (h)", "PX down (h)"], "PM repl down", "CBM")
    cm_repl_down = _require_any(df, ["CM Repl down (h)", "CX down (h)"], "CM repl down", "CBM")

    pm_rep_emis = _require_any(df, ["PM Rep emis (kg CO2)", "PM Rep emis (kg CO_2)", "PR emis (kg CO2)", "PR emis (kg CO_2)"], "PM rep emis", "CBM")
    pm_repl_emis = _require_any(df, ["PM Repl emis (kg CO2)", "PM Repl emis (kg CO_2)", "PX emis (kg CO2)", "PX emis (kg CO_2)"], "PM repl emis", "CBM")
    cm_repl_emis = _require_any(df, ["CM Repl emis (kg CO2)", "CM Repl emis (kg CO_2)", "CX emis (kg CO2)", "CX emis (kg CO_2)"], "CM repl emis", "CBM")

    comps: List[CBMComponent] = []
    for _, r in df.iterrows():
        comps.append(
            CBMComponent(
                name=str(r[name_col]).strip(),
                quantity=int(round(float(r[qty_col]))),
                component_class=str(r[class_col]).strip(),
                critical=_criticality_to_bool(r[crit_col]),
                gamma_shape=float(r[g_shape_col]),
                gamma_scale=float(r[g_scale_col]),
                X0=float(r[x0_col]),
                H=float(r[h_col]),
                rho=float(r[rho_col]),
                c_PMrep=float(r[pm_rep_cost]),
                c_PMrepl=float(r[pm_repl_cost]),
                c_CMrepl=float(r[cm_repl_cost]),
                d_PMrep=float(r[pm_rep_down]),
                d_PMrepl=float(r[pm_repl_down]),
                d_CMrepl=float(r[cm_repl_down]),
                e_PMrep=float(r[pm_rep_emis]),
                e_PMrepl=float(r[pm_repl_emis]),
                e_CMrepl=float(r[cm_repl_emis]),
            )
        )
    return comps


def _df_to_abm_components(df: pd.DataFrame) -> List[ABMComponent]:
    if df is None or df.empty:
        return []
    df = _clean_df(df)

    name_col = _require_any(df, ["Name", "name"], "Name", "ABM")
    qty_col = _require_any(df, ["Qty", "quantity", "Quantity"], "Qty", "ABM")
    class_col = _require_any(df, ["Class", "class"], "Class", "ABM")
    crit_col = _require_any(df, ["Criticality", "criticality"], "Criticality", "ABM")

    w_beta_col = _require_any(df, ["Weibull shape (beta)", "Weibull shape (β)", "Weibull shape"], "Weibull shape", "ABM")
    w_alpha_col = _require_any(df, ["Weibull scale (alpha)", "Weibull scale (α)", "Weibull scale"], "Weibull scale", "ABM")

    rho_col = _require_any(df, ["rho (Kijima)", "rho", "Kijima", "Kijima rho"], "rho (Kijima)", "ABM")

    pm_rep_cost = _require_any(df, ["PM Rep cost (EUR)", "PM Rep cost (€)", "PR cost (€)", "PR cost (EUR)"], "PM rep cost", "ABM")
    pm_repl_cost = _require_any(df, ["PM Repl cost (EUR)", "PM Repl cost (€)", "PX cost (€)", "PX cost (EUR)"], "PM repl cost", "ABM")
    cm_repl_cost = _require_any(df, ["CM Repl cost (EUR)", "CM Repl cost (€)", "CX cost (€)", "CX cost (EUR)"], "CM repl cost", "ABM")

    pm_rep_down = _require_any(df, ["PM Rep down (h)", "PR down (h)"], "PM rep down", "ABM")
    pm_repl_down = _require_any(df, ["PM Repl down (h)", "PX down (h)"], "PM repl down", "ABM")
    cm_repl_down = _require_any(df, ["CM Repl down (h)", "CX down (h)"], "CM repl down", "ABM")

    pm_rep_emis = _require_any(df, ["PM Rep emis (kg CO2)", "PM Rep emis (kg CO_2)", "PR emis (kg CO2)", "PR emis (kg CO_2)"], "PM rep emis", "ABM")
    pm_repl_emis = _require_any(df, ["PM Repl emis (kg CO2)", "PM Repl emis (kg CO_2)", "PX emis (kg CO2)", "PX emis (kg CO_2)"], "PM repl emis", "ABM")
    cm_repl_emis = _require_any(df, ["CM Repl emis (kg CO2)", "CM Repl emis (kg CO_2)", "CX emis (kg CO2)", "CX emis (kg CO_2)"], "CM repl emis", "ABM")

    comps: List[ABMComponent] = []
    for _, r in df.iterrows():
        comps.append(
            ABMComponent(
                name=str(r[name_col]).strip(),
                quantity=int(round(float(r[qty_col]))),
                component_class=str(r[class_col]).strip(),
                critical=_criticality_to_bool(r[crit_col]),
                weibull_beta=float(r[w_beta_col]),
                weibull_alpha=float(r[w_alpha_col]),
                rho=float(r[rho_col]),
                c_PMrep=float(r[pm_rep_cost]),
                c_PMrepl=float(r[pm_repl_cost]),
                c_CMrepl=float(r[cm_repl_cost]),
                d_PMrep=float(r[pm_rep_down]),
                d_PMrepl=float(r[pm_repl_down]),
                d_CMrepl=float(r[cm_repl_down]),
                e_PMrep=float(r[pm_rep_emis]),
                e_PMrepl=float(r[pm_repl_emis]),
                e_CMrepl=float(r[cm_repl_emis]),
            )
        )
    return comps


def _df_to_fbm_components(df: pd.DataFrame) -> List[FBMComponent]:
    if df is None or df.empty:
        return []
    df = _clean_df(df)

    name_col = _require_any(df, ["Name", "name"], "Name", "FBM")
    qty_col = _require_any(df, ["Qty", "quantity", "Quantity"], "Qty", "FBM")
    class_col = _require_any(df, ["Class", "class"], "Class", "FBM")
    crit_col = _require_any(df, ["Criticality", "criticality"], "Criticality", "FBM")

    w_beta_col = _require_any(df, ["Weibull shape (beta)", "Weibull shape (β)", "Weibull shape"], "Weibull shape", "FBM")
    w_alpha_col = _require_any(df, ["Weibull scale (alpha)", "Weibull scale (α)", "Weibull scale"], "Weibull scale", "FBM")

    rho_col = _require_any(df, ["rho (Kijima)", "rho", "Kijima", "Kijima rho"], "rho (Kijima)", "FBM")

    cm_repl_cost = _require_any(df, ["CM Repl cost (EUR)", "CM Repl cost (€)", "CX cost (€)", "CX cost (EUR)"], "CM repl cost", "FBM")
    cm_repl_down = _require_any(df, ["CM Repl down (h)", "CX down (h)"], "CM repl down", "FBM")
    cm_repl_emis = _require_any(df, ["CM Repl emis (kg CO2)", "CM Repl emis (kg CO_2)", "CX emis (kg CO2)", "CX emis (kg CO_2)"], "CM repl emis", "FBM")

    comps: List[FBMComponent] = []
    for _, r in df.iterrows():
        comps.append(
            FBMComponent(
                name=str(r[name_col]).strip(),
                quantity=int(round(float(r[qty_col]))),
                component_class=str(r[class_col]).strip(),
                critical=_criticality_to_bool(r[crit_col]),
                weibull_beta=float(r[w_beta_col]),
                weibull_alpha=float(r[w_alpha_col]),
                rho=float(r[rho_col]),
                c_CMrepl=float(r[cm_repl_cost]),
                d_CMrepl=float(r[cm_repl_down]),
                e_CMrepl=float(r[cm_repl_emis]),
            )
        )
    return comps


# ============================
# System evaluation with extra survival stats (engine-side)
# ============================
def _evaluate_system_with_stats(
    config: SystemConfig,
    tau: float,
    thresholds_dict: Dict[str, Any],
    bundle,  # HistoryBundle from rocky_core
    cbm_components: List[CBMComponent],
    abm_components: List[ABMComponent],
    fbm_components: List[FBMComponent],
) -> Tuple[float, float, float, float, float, int]:
    """
    Returns:
      (Z_sys, V_sys, E_sys, p_sur_sys, survival_avg_years, survived_traces)

    This mirrors rocky_core.evaluate_system_spliced_from_histories(...) but also returns:
      - average system lifetime across traces
      - number of survived traces
    """
    n_steps = compute_n_steps(config)
    idx = np.arange(config.n_traces)

    # Maps
    qty_map = {c.name: int(c.quantity) for c in (cbm_components + abm_components + fbm_components)}
    crit_map = {c.name: bool(c.critical) for c in (cbm_components + abm_components + fbm_components)}

    # 1) system death time = earliest critical death time
    death_times = np.full(config.n_traces, config.T, dtype=float)
    has_critical = False

    for comp_name, thr in thresholds_dict.items():
        if crit_map.get(comp_name, False):
            has_critical = True
            dt_arr = np.asarray(bundle.death_times[comp_name][thr], dtype=float)
            death_times = np.minimum(death_times, dt_arr)

    p_sur_sys = 1.0 if (not has_critical) else float(np.mean(np.isclose(death_times, config.T)))
    survival_avg = float(np.mean(death_times))
    survived_traces = int(np.sum(np.isclose(death_times, config.T)))

    # 2) convert death_times -> slice indices
    death_idx = (death_times / config.dt_years).astype(int) - 1
    death_idx = np.clip(death_idx, 0, n_steps - 1)

    # 3) slice component histories at system stop time
    sys_Z = np.zeros(config.n_traces, dtype=float)
    sys_V = np.zeros(config.n_traces, dtype=float)
    sys_E = np.zeros(config.n_traces, dtype=float)

    for comp_name, thr in thresholds_dict.items():
        q = int(qty_map.get(comp_name, 1))
        hz = bundle.cost_hist[comp_name][thr]
        hv = bundle.down_hist[comp_name][thr]
        he = bundle.emis_hist[comp_name][thr]

        sys_Z += q * hz[idx, death_idx]
        sys_V += q * hv[idx, death_idx]
        sys_E += q * he[idx, death_idx]

    # 4) add system death penalty if died early
    died = death_times < config.T
    sys_Z[died] += config.system_penalty_cost
    sys_V[died] += config.system_penalty_downtime
    sys_E[died] += config.system_penalty_emission

    # 5) add scheduled down penalties for visits completed before death
    n_visits = np.floor(death_times / float(tau))
    sys_Z += n_visits * config.c_SD
    sys_V += n_visits * config.d_SD
    sys_E += n_visits * config.e_SD

    # 6) annualize
    denom = np.maximum(death_times, 1e-9)
    Z_sys = float(np.mean(sys_Z / denom))
    V_sys = float(np.mean(sys_V / denom))
    E_sys = float(np.mean(sys_E / denom))

    # survival constraint => infeasible => inf objectives (Rocky behavior)
    if config.enforce_survival_constraint and (p_sur_sys < float(config.survival_target)):
        return float("inf"), float("inf"), float("inf"), p_sur_sys, survival_avg, survived_traces

    return Z_sys, V_sys, E_sys, p_sur_sys, survival_avg, survived_traces


# ============================
# Reporting helpers
# ============================
def _safe_numeric(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    """Convert cols to numeric (coerce), and drop +/- inf -> NaN."""
    out = df.copy()
    out = out.replace([np.inf, -np.inf], np.nan)
    for c in cols:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def _knee_point(df: pd.DataFrame) -> Optional[pd.Series]:
    """
    Knee point = closest to ideal after min-max normalization on (Z_sys,V_sys,E_sys).
    """
    if df.empty:
        return None
    work = df.copy()

    for col in ["Z_sys", "V_sys", "E_sys"]:
        v = work[col].to_numpy(dtype=float)
        vmin, vmax = float(np.nanmin(v)), float(np.nanmax(v))
        if np.isclose(vmax - vmin, 0.0):
            work[col + "_n"] = 0.0
        else:
            work[col + "_n"] = (v - vmin) / (vmax - vmin)

    dist = np.sqrt(work["Z_sys_n"] ** 2 + work["V_sys_n"] ** 2 + work["E_sys_n"] ** 2)
    idx = int(dist.idxmin())
    return df.loc[idx]


def _method_per_tau_extremes(method: str, df: pd.DataFrame, tau_grid: List[float]) -> pd.DataFrame:
    """
    Build Per_Tau_Extremes rows for a system method.

    Expected df columns: Tau, Z_sys, V_sys, E_sys, p_surv_sys (optional)

    Output columns (clear + matches your intended structure):
      Method, Tau,
      Z* (EUR/yr), V_at_Z* (h/yr), E_at_Z* (kg CO2/yr),
      V* (h/yr), Z_at_V* (EUR/yr), E_at_V* (kg CO2/yr),
      E* (kg CO2/yr), Z_at_E* (EUR/yr), V_at_E* (h/yr)
    """
    rows = []
    if df.empty:
        return pd.DataFrame(rows)

    df2 = _safe_numeric(df, ["Tau", "Z_sys", "V_sys", "E_sys"])
    df2 = df2.dropna(subset=["Tau", "Z_sys", "V_sys", "E_sys"])

    for tau in tau_grid:
        dft = df2[df2["Tau"].astype(float) == float(tau)].copy()
        if dft.empty:
            rows.append({"Method": method, "Tau": tau})
            continue

        # min cost
        rZ = dft.loc[dft["Z_sys"].astype(float).idxmin()]
        # min downtime
        rV = dft.loc[dft["V_sys"].astype(float).idxmin()]
        # min emissions
        rE = dft.loc[dft["E_sys"].astype(float).idxmin()]

        rows.append({
            "Method": method,
            "Tau": float(tau),

            "Z* (EUR/yr)": float(rZ["Z_sys"]),
            "V_at_Z* (h/yr)": float(rZ["V_sys"]),
            "E_at_Z* (kg CO2/yr)": float(rZ["E_sys"]),

            "V* (h/yr)": float(rV["V_sys"]),
            "Z_at_V* (EUR/yr)": float(rV["Z_sys"]),
            "E_at_V* (kg CO2/yr)": float(rV["E_sys"]),

            "E* (kg CO2/yr)": float(rE["E_sys"]),
            "Z_at_E* (EUR/yr)": float(rE["Z_sys"]),
            "V_at_E* (h/yr)": float(rE["V_sys"]),
        })

    return pd.DataFrame(rows)


def _component_optima_from_menus(
    df_menus: pd.DataFrame,
    components: List[str],
    tau_grid: List[float],
    sheet_label: str,
) -> pd.DataFrame:
    """
    Builds CBM_Optima / ABM_Optima / FBM_Optima from Component_Pareto_Menus.

    Required df_menus cols:
      Component, Tau, Threshold, Z, V, E, p_surv

    Output cols (as you requested + p_sur):
      Tau, Component,
      Min_Cost_Thr, Cost_at_MinCost, Down_at_MinCost, Emis_at_MinCost, p_sur_at_MinCost,
      Min_Down_Thr, Cost_at_MinDown, Down_at_MinDown, Emis_at_MinDown, p_sur_at_MinDown,
      Min_Emis_Thr, Cost_at_MinEmis, Down_at_MinEmis, Emis_at_MinEmis, p_sur_at_MinEmis
    """
    rows = []
    if df_menus.empty:
        return pd.DataFrame(rows)

    df = df_menus.copy()
    df = _safe_numeric(df, ["Tau", "Z", "V", "E", "p_surv"])
    df = df.dropna(subset=["Tau", "Z", "V", "E"])

    comp_set = set(components)

    for tau in tau_grid:
        dft = df[df["Tau"].astype(float) == float(tau)].copy()
        if dft.empty:
            # still write empty rows for each component? (optional)
            continue

        for comp in sorted(comp_set):
            dfc = dft[dfc_component := (dft["Component"] == comp)].copy()  # keep explicit for readability
            if dfc.empty:
                continue

            rZ = dfc.loc[dfc["Z"].astype(float).idxmin()]
            rV = dfc.loc[dfc["V"].astype(float).idxmin()]
            rE = dfc.loc[dfc["E"].astype(float).idxmin()]

            rows.append({
                "Tau": float(tau),
                "Component": comp,

                "Min_Cost_Thr": rZ.get("Threshold", None),
                "Cost_at_MinCost": float(rZ["Z"]),
                "Down_at_MinCost": float(rZ["V"]),
                "Emis_at_MinCost": float(rZ["E"]),
                "p_sur_at_MinCost": float(rZ.get("p_surv", np.nan)),

                "Min_Down_Thr": rV.get("Threshold", None),
                "Cost_at_MinDown": float(rV["Z"]),
                "Down_at_MinDown": float(rV["V"]),
                "Emis_at_MinDown": float(rV["E"]),
                "p_sur_at_MinDown": float(rV.get("p_surv", np.nan)),

                "Min_Emis_Thr": rE.get("Threshold", None),
                "Cost_at_MinEmis": float(rE["Z"]),
                "Down_at_MinEmis": float(rE["V"]),
                "Emis_at_MinEmis": float(rE["E"]),
                "p_sur_at_MinEmis": float(rE.get("p_surv", np.nan)),
            })

    return pd.DataFrame(rows)


# ============================
# Main entry point
# ============================
def run_promcs(model_inputs: Dict[str, Any], output_dir: str = "outputs") -> EngineResult:
    """
    Main engine entry called by Streamlit.
    Exports a complete Excel report for ALL tau values and ALL evaluated solutions.
    """
    t_total_start = time.perf_counter()
    excel_path, plots_dir = _ensure_output_dirs(output_dir)

    # -------------------------
    # Method toggles (defaults True)
    # -------------------------
    run_wsm = bool(model_inputs.get("run_wsm", True))
    run_ga = bool(model_inputs.get("run_ga", True))
    run_sga = bool(model_inputs.get("run_sga", True))

    # -------------------------
    # 1) Config
    # -------------------------
    config = SystemConfig(
        T=float(model_inputs["time_horizon_years"]),
        n_traces=int(model_inputs["n_traces"]),
        steps_per_year=12,                # fixed (Rocky)
        rng_seed=42,                      # fixed (Rocky)
        c_SD=float(model_inputs["sd_cost"]),
        d_SD=float(model_inputs["sd_downtime"]),
        e_SD=float(model_inputs["sd_emissions"]),
        system_penalty_cost=float(model_inputs["system_penalty_cost"]),
        system_penalty_downtime=float(model_inputs["system_penalty_downtime"]),
        system_penalty_emission=float(model_inputs["system_penalty_emissions"]),
        enforce_survival_constraint=True,  # hardcoded (Rocky)
        survival_target=float(model_inputs["survival_probability"]),
        repair_or_replace_rule_threshold=float(model_inputs["p_major_threshold"]),
    )

    tau_grid_years = list(model_inputs.get("tau_years_list", []))
    if len(tau_grid_years) == 0:
        raise ValueError("Tau grid is empty. Provide at least one tau value.")

    # Ensure sorted tau grid for consistent reporting
    tau_grid_years = sorted([float(x) for x in tau_grid_years])

    n_steps = compute_n_steps(config)

    # -------------------------
    # 2) Components
    # -------------------------
    cbm_components = _df_to_cbm_components(model_inputs["cbm_df"])
    abm_components = _df_to_abm_components(model_inputs["abm_df"])
    fbm_components = _df_to_fbm_components(model_inputs["fbm_df"])

    comp_names = [c.name for c in cbm_components] + [c.name for c in abm_components] + [c.name for c in fbm_components]
    comp_names_sorted = sorted(comp_names)

    cbm_names = [c.name for c in cbm_components]
    abm_names = [c.name for c in abm_components]
    fbm_names = [c.name for c in fbm_components]

    # -------------------------
    # 3) CRN bank
    # -------------------------
    max_events = int(model_inputs["crn_bank_size"])
    crn_bank = build_crn_bank(
        config=config,
        cbm_names=cbm_names,
        abm_names=abm_names,
        fbm_names=fbm_names,
        max_events=max_events,
    )

    # -------------------------
    # 4) Decision grids
    # -------------------------
    decision_grids = build_decision_grids(
        config=config,
        cbm_components=cbm_components,
        abm_components=abm_components,
    )

    # -------------------------
    # 5) PASS 1 totals (ALL tau + ALL thresholds)
    # -------------------------
    CBM_RESULTS, ABM_RESULTS, FBM_RESULTS, pass1_cpu_time = run_pass1_component_totals(
        config=config,
        cbm_components=cbm_components,
        abm_components=abm_components,
        fbm_components=fbm_components,
        tau_grid_years=tau_grid_years,
        decision_grids=decision_grids,
        crn_bank=crn_bank,
    )

    # -------------------------
    # 6) Component Pareto menus (Block 4B)
    # -------------------------
    COMPONENT_PARETO_OPTIONS = build_component_pareto_menus_pass1(
        config=config,
        CBM_RESULTS=CBM_RESULTS,
        ABM_RESULTS=ABM_RESULTS,
        FBM_RESULTS=FBM_RESULTS,
        cbm_components=cbm_components,
        abm_components=abm_components,
        fbm_components=fbm_components,
    )

    # Build menu count + menu table
    menu_count_rows: List[Dict[str, Any]] = []
    menu_rows: List[Dict[str, Any]] = []

    for (name, tau), opts in COMPONENT_PARETO_OPTIONS.items():
        menu_count_rows.append({"Component": name, "Tau": float(tau), "MenuSize": int(len(opts))})
        for o in opts:
            menu_rows.append({
                "Component": name,
                "Tau": float(tau),
                "Threshold": o["thr"],
                "Z": float(o["Z"]),
                "V": float(o["V"]),
                "E": float(o["E"]),
                "p_surv": float(o.get("p_surv", np.nan)),
            })

    component_menu_counts_df = pd.DataFrame(menu_count_rows)
    component_pareto_menus_df = pd.DataFrame(menu_rows)

    # -------------------------
    # 6B) Component Optima sheets (CBM/ABM/FBM)  ✅ (requested)
    # -------------------------
    cbm_optima_df = _component_optima_from_menus(component_pareto_menus_df, cbm_names, tau_grid_years, "CBM")
    abm_optima_df = _component_optima_from_menus(component_pareto_menus_df, abm_names, tau_grid_years, "ABM")
    fbm_optima_df = _component_optima_from_menus(component_pareto_menus_df, fbm_names, tau_grid_years, "FBM")

    # -------------------------
    # 7) System baseline KPI per tau (first menu option per component at that tau)
    # -------------------------
    system_all_tau_rows: List[Dict[str, Any]] = []

    # We'll also store "current policy" (baseline at first tau) with survival stats
    current_policy = {
        "Tau": float(tau_grid_years[0]),
        "Z_sys": float("inf"),
        "V_sys": float("inf"),
        "E_sys": float("inf"),
        "p_sur_sys": 0.0,
        "survival_avg": float("nan"),
        "survived_traces": 0,
        "Thresholds": "{}",
    }

    fbm_name_set = set(fbm_names)

    for tau_val in tau_grid_years:
        tau_val = float(tau_val)

        # Build histories for this tau (Pareto menu points only)
        tau_bundle = build_histories_for_tau_pass2(
            config=config,
            cbm_components=cbm_components,
            abm_components=abm_components,
            fbm_components=fbm_components,
            tau=tau_val,
            component_pareto_menus=COMPONENT_PARETO_OPTIONS,
            crn_bank=crn_bank,
        )

        # Baseline thresholds: first Pareto menu option per component at this tau
        baseline_thresholds: Dict[str, Any] = {}
        missing: List[str] = []

        for comp_name in tau_bundle.cost_hist.keys():
            opts = COMPONENT_PARETO_OPTIONS.get((comp_name, tau_val), [])

            if len(opts) > 0:
                baseline_thresholds[comp_name] = opts[0]["thr"]
                continue

            # FBM fallback
            if comp_name in fbm_name_set:
                baseline_thresholds[comp_name] = None
                continue

            # CBM/ABM missing feasible menu at this tau
            missing.append(comp_name)

        if missing:
            Z_sys, V_sys, E_sys, p_sur, surv_avg, surv_count = float("inf"), float("inf"), float("inf"), 0.0, float("nan"), 0
        else:
            Z_sys, V_sys, E_sys, p_sur, surv_avg, surv_count = _evaluate_system_with_stats(
                config=config,
                tau=tau_val,
                thresholds_dict=baseline_thresholds,
                bundle=tau_bundle,
                cbm_components=cbm_components,
                abm_components=abm_components,
                fbm_components=fbm_components,
            )

        system_all_tau_rows.append({
            "Tau": tau_val,
            "Z_sys": Z_sys,
            "V_sys": V_sys,
            "E_sys": E_sys,
            "p_sur_sys": p_sur,
            "survival_avg": surv_avg,
            "survived_traces": surv_count,
            "baseline_components": int(len(baseline_thresholds)),
            "missing_components": ",".join(missing),
        })

        # Save current policy at first tau
        if np.isclose(tau_val, float(tau_grid_years[0])):
            current_policy = {
                "Tau": tau_val,
                "Z_sys": Z_sys,
                "V_sys": V_sys,
                "E_sys": E_sys,
                "p_sur_sys": p_sur,
                "survival_avg": surv_avg,
                "survived_traces": surv_count,
                "Thresholds": str(baseline_thresholds),
            }

    system_all_tau_df = pd.DataFrame(system_all_tau_rows)

    # -------------------------
    # 8) WSM system enumeration (Block 5)  (optional)
    # -------------------------
    wsm_cpu_time = 0.0
    wsm_combos_evaluated = 0
    wsm_pareto_size = 0

    df_wsm_all = pd.DataFrame()
    df_wsm_pareto = pd.DataFrame()

    WSM_SYSTEM_PARETO_FIXED: List[Dict[str, Any]] = []

    if run_wsm:
        wsm_start = time.perf_counter()
        WSM_SYSTEM_SOLUTIONS: List[Dict[str, Any]] = []

        cbm_set = set(cbm_names)
        abm_set = set(abm_names)

        def _pareto_filter_per_tau_system(df: pd.DataFrame) -> pd.DataFrame:
            """Simple Pareto filter per tau (minimize Z_sys,V_sys,E_sys)."""
            if df.empty:
                return df

            df2 = _safe_numeric(df, ["Tau", "Z_sys", "V_sys", "E_sys"])
            df2 = df2.dropna(subset=["Tau", "Z_sys", "V_sys", "E_sys"])

            keep_idx = []
            vals = df2[["Z_sys", "V_sys", "E_sys"]].to_numpy(dtype=float)
            n = vals.shape[0]

            for i in range(n):
                Zi, Vi, Ei = vals[i]
                dom = False
                for j in range(n):
                    if i == j:
                        continue
                    Zj, Vj, Ej = vals[j]
                    if (Zj <= Zi and Vj <= Vi and Ej <= Ei) and (Zj < Zi or Vj < Vi or Ej < Ei):
                        dom = True
                        break
                if not dom:
                    keep_idx.append(i)

            return df2.iloc[keep_idx].reset_index(drop=True)

        for tau in tau_grid_years:
            tau = float(tau)

            # Skip tau if any component has an empty menu
            if any(len(COMPONENT_PARETO_OPTIONS.get((name, tau), [])) == 0 for name in comp_names):
                continue

            tau_bundle = build_histories_for_tau_pass2(
                config=config,
                cbm_components=cbm_components,
                abm_components=abm_components,
                fbm_components=fbm_components,
                tau=tau,
                component_pareto_menus=COMPONENT_PARETO_OPTIONS,
                crn_bank=crn_bank,
            )

            # Per-component extremes used by WSM (min cost/min down/min emis)
            comp_best: Dict[str, Dict[str, Dict[str, Any]]] = {}
            per_comp_keys: List[List[str]] = []

            for name in comp_names:
                opts = COMPONENT_PARETO_OPTIONS[(name, tau)]

                if (name in cbm_set) or (name in abm_set):
                    c_opt = min(opts, key=lambda o: float(o["Z"]))
                    d_opt = min(opts, key=lambda o: float(o["V"]))
                    e_opt = min(opts, key=lambda o: float(o["E"]))
                    comp_best[name] = {"cost": c_opt, "downtime": d_opt, "emission": e_opt}
                    per_comp_keys.append(["cost", "downtime", "emission"])
                else:
                    # FBM fixed (thr=None)
                    comp_best[name] = {"fixed": opts[0]}
                    per_comp_keys.append(["fixed"])

            for combo in itertools.product(*per_comp_keys):
                thresholds = {name: comp_best[name][combo[i]]["thr"] for i, name in enumerate(comp_names)}

                # Use rocky_core evaluator (fast) for objective rates
                Z_sys, V_sys, E_sys, p_surv_sys = _evaluate_system_with_stats(
                    config=config,
                    tau=tau,
                    thresholds_dict=thresholds,
                    bundle=tau_bundle,
                    cbm_components=cbm_components,
                    abm_components=abm_components,
                    fbm_components=fbm_components,
                )[0:4]  # ignore survival avg details here

                WSM_SYSTEM_SOLUTIONS.append({
                    "Tau": tau,
                    "Z_sys": Z_sys,
                    "V_sys": V_sys,
                    "E_sys": E_sys,
                    "p_surv_sys": p_surv_sys,
                    "Thresholds": str(thresholds),
                })

        wsm_cpu_time = float(time.perf_counter() - wsm_start)
        wsm_combos_evaluated = int(len(WSM_SYSTEM_SOLUTIONS))

        df_wsm_all = pd.DataFrame(WSM_SYSTEM_SOLUTIONS)

        # Pareto per tau
        pareto_rows: List[Dict[str, Any]] = []
        for tau in tau_grid_years:
            dft = df_wsm_all[df_wsm_all["Tau"].astype(float) == float(tau)].copy()
            dft = _safe_numeric(dft, ["Z_sys", "V_sys", "E_sys"])
            dft = dft.dropna(subset=["Z_sys", "V_sys", "E_sys"])
            if dft.empty:
                continue
            pareto_rows.extend(_pareto_filter_per_tau_system(dft).to_dict("records"))

        df_wsm_pareto = pd.DataFrame(pareto_rows)
        WSM_SYSTEM_PARETO_FIXED = df_wsm_pareto.to_dict("records")
        wsm_pareto_size = int(len(df_wsm_pareto))

    # -------------------------
    # 9) Standard GA + Seeded GA (optional)
    # -------------------------
    GA_POP_SIZE = 500
    GA_NUM_GENS = 500
    ELITE_TARGET = 250
    TOURNAMENT_K = 50
    OFFSPRING_N = 200
    MUT_RATE = 0.10

    ga_cpu_time = 0.0
    seeded_ga_cpu_time = 0.0

    df_ga = pd.DataFrame()
    df_sga = pd.DataFrame()

    df_ga_tau = pd.DataFrame()
    df_sga_tau = pd.DataFrame()

    ga_pareto_size = 0
    seeded_ga_pareto_size = 0

    # ---- Standard GA
    if run_ga:
        ga_start = time.perf_counter()
        GA_RESULTS_ALL: List[Dict[str, Any]] = []
        ga_tau_rows: List[Dict[str, Any]] = []

        for t in tau_grid_years:
            t = float(t)
            sols_tau, rt_tau = run_nsga2_for_tau(
                config=config,
                tau=t,
                comp_names=comp_names_sorted,
                component_pareto_options=COMPONENT_PARETO_OPTIONS,
                cbm_components=cbm_components,
                abm_components=abm_components,
                fbm_components=fbm_components,
                crn_bank=crn_bank,
                wsm_seeds=None,
                seed=False,
                use_cache=True,
                pop_size=GA_POP_SIZE,
                num_gens=GA_NUM_GENS,
                elite_target=ELITE_TARGET,
                tournament_k=TOURNAMENT_K,
                offspring_n=OFFSPRING_N,
                mut_rate=MUT_RATE,
            )
            GA_RESULTS_ALL.extend(sols_tau)
            ga_tau_rows.append({"Tau": t, "ga_tau_time_s": float(rt_tau), "ga_tau_pareto": int(len(sols_tau))})

        ga_cpu_time = float(time.perf_counter() - ga_start)

        df_ga = pd.DataFrame(GA_RESULTS_ALL)
        if not df_ga.empty and "Thresholds" in df_ga.columns:
            df_ga["Thresholds"] = df_ga["Thresholds"].apply(str)

        df_ga_tau = pd.DataFrame(ga_tau_rows)
        ga_pareto_size = int(len(df_ga))

    # ---- Seeded GA
    if run_sga:
        sga_start = time.perf_counter()
        SEEDED_GA_RESULTS_ALL: List[Dict[str, Any]] = []
        sga_tau_rows: List[Dict[str, Any]] = []

        # If WSM not run, seeds will be empty (Seeded GA becomes effectively random init)
        seeds_for_sga = WSM_SYSTEM_PARETO_FIXED if (run_wsm and len(WSM_SYSTEM_PARETO_FIXED) > 0) else []

        for t in tau_grid_years:
            t = float(t)
            sols_tau, rt_tau = run_nsga2_for_tau(
                config=config,
                tau=t,
                comp_names=comp_names_sorted,
                component_pareto_options=COMPONENT_PARETO_OPTIONS,
                cbm_components=cbm_components,
                abm_components=abm_components,
                fbm_components=fbm_components,
                crn_bank=crn_bank,
                wsm_seeds=seeds_for_sga,
                seed=True,
                use_cache=True,
                pop_size=GA_POP_SIZE,
                num_gens=GA_NUM_GENS,
                elite_target=ELITE_TARGET,
                tournament_k=TOURNAMENT_K,
                offspring_n=OFFSPRING_N,
                mut_rate=MUT_RATE,
            )
            SEEDED_GA_RESULTS_ALL.extend(sols_tau)
            sga_tau_rows.append({"Tau": t, "sga_tau_time_s": float(rt_tau), "sga_tau_pareto": int(len(sols_tau))})

        seeded_ga_cpu_time = float(time.perf_counter() - sga_start)

        df_sga = pd.DataFrame(SEEDED_GA_RESULTS_ALL)
        if not df_sga.empty and "Thresholds" in df_sga.columns:
            df_sga["Thresholds"] = df_sga["Thresholds"].apply(str)

        df_sga_tau = pd.DataFrame(sga_tau_rows)
        seeded_ga_pareto_size = int(len(df_sga))

    # -------------------------
    # 10) Per_Tau_Extremes sheet  ✅ (requested)
    # -------------------------
    per_tau_extremes_frames: List[pd.DataFrame] = []

    # WSM per-tau extremes from WSM Pareto (or all WSM if Pareto empty)
    if run_wsm:
        base_wsm_df = df_wsm_pareto if (not df_wsm_pareto.empty) else df_wsm_all
        per_tau_extremes_frames.append(_method_per_tau_extremes("WSM", base_wsm_df, tau_grid_years))

    # GA per-tau extremes from GA Pareto
    if run_ga:
        per_tau_extremes_frames.append(_method_per_tau_extremes("GA", df_ga, tau_grid_years))

    # Seeded GA per-tau extremes
    if run_sga:
        per_tau_extremes_frames.append(_method_per_tau_extremes("Seeded GA", df_sga, tau_grid_years))

    df_per_tau_extremes = pd.concat(per_tau_extremes_frames, ignore_index=True) if per_tau_extremes_frames else pd.DataFrame()

    # -------------------------
    # 11) Detailed_Extremes (global min)  (already used by app)
    # -------------------------
    def _detailed_extremes(method: str, df: pd.DataFrame) -> List[dict]:
        """
        Create 3 rows (min cost / min downtime / min emission) for a method.
        Expected df columns: Tau, Z_sys, V_sys, E_sys, p_surv_sys
        """
        out: List[dict] = []
        if df.empty:
            return out

        finite = _safe_numeric(df, ["Tau", "Z_sys", "V_sys", "E_sys"])
        finite = finite.dropna(subset=["Z_sys", "V_sys", "E_sys"])
        if finite.empty:
            return out

        row_c = finite.loc[finite["Z_sys"].astype(float).idxmin()]
        row_d = finite.loc[finite["V_sys"].astype(float).idxmin()]
        row_e = finite.loc[finite["E_sys"].astype(float).idxmin()]

        out.append({"Method": method, "Optimized_For": "Cost", "Tau (yrs)": row_c["Tau"], "Cost Rate (€/yr)": row_c["Z_sys"], "Downtime Rate (h/yr)": row_c["V_sys"], "Emission Rate (kg CO2/yr)": row_c["E_sys"], "p_sur_sys": row_c.get("p_surv_sys", np.nan)})
        out.append({"Method": method, "Optimized_For": "Downtime", "Tau (yrs)": row_d["Tau"], "Cost Rate (€/yr)": row_d["Z_sys"], "Downtime Rate (h/yr)": row_d["V_sys"], "Emission Rate (kg CO2/yr)": row_d["E_sys"], "p_sur_sys": row_d.get("p_surv_sys", np.nan)})
        out.append({"Method": method, "Optimized_For": "Emission", "Tau (yrs)": row_e["Tau"], "Cost Rate (€/yr)": row_e["Z_sys"], "Downtime Rate (h/yr)": row_e["V_sys"], "Emission Rate (kg CO2/yr)": row_e["E_sys"], "p_sur_sys": row_e.get("p_surv_sys", np.nan)})
        return out

    detailed_rows: List[dict] = []
    if run_wsm:
        detailed_rows += _detailed_extremes("WSM", df_wsm_pareto if not df_wsm_pareto.empty else df_wsm_all)
    if run_ga:
        detailed_rows += _detailed_extremes("GA", df_ga)
    if run_sga:
        detailed_rows += _detailed_extremes("Seeded GA", df_sga)

    df_detailed_extremes = pd.DataFrame(detailed_rows)

    # -------------------------
    # 12) Final report 2 (Metric/Value)  ✅ includes your requested list
    # -------------------------
    total_cpu_time = float(time.perf_counter() - t_total_start)

    # Build knee from Seeded GA -> GA -> WSM Pareto
    knee_source = "Seeded GA"
    knee_df = df_sga.copy() if (run_sga and not df_sga.empty) else pd.DataFrame()

    if knee_df.empty:
        knee_source = "GA"
        knee_df = df_ga.copy() if (run_ga and not df_ga.empty) else pd.DataFrame()

    if knee_df.empty:
        knee_source = "WSM"
        knee_df = df_wsm_pareto.copy() if (run_wsm and not df_wsm_pareto.empty) else pd.DataFrame()

    knee_row = _knee_point(_safe_numeric(knee_df, ["Tau", "Z_sys", "V_sys", "E_sys"])) if not knee_df.empty else None

    # Evaluate knee survival stats (avg + survived traces) if thresholds exist
    knee_surv_avg = float("nan")
    knee_surv_count = 0
    knee_p_sur = float("nan")

    if knee_row is not None and ("Tau" in knee_row.index):
        knee_tau = float(knee_row["Tau"])
        knee_thresholds = _parse_thresholds(knee_row.get("Thresholds", "{}"))

        # Build histories for that tau
        knee_bundle = build_histories_for_tau_pass2(
            config=config,
            cbm_components=cbm_components,
            abm_components=abm_components,
            fbm_components=fbm_components,
            tau=knee_tau,
            component_pareto_menus=COMPONENT_PARETO_OPTIONS,
            crn_bank=crn_bank,
        )

        Zk, Vk, Ek, pk, savg, scount = _evaluate_system_with_stats(
            config=config,
            tau=knee_tau,
            thresholds_dict=knee_thresholds,
            bundle=knee_bundle,
            cbm_components=cbm_components,
            abm_components=abm_components,
            fbm_components=fbm_components,
        )
        knee_p_sur = float(pk)
        knee_surv_avg = float(savg)
        knee_surv_count = int(scount)

    # Helper to get min tau for a df and column
    def _min_tau_for(df: pd.DataFrame, col: str) -> Optional[float]:
        if df.empty:
            return None
        d = _safe_numeric(df, ["Tau", col]).dropna(subset=["Tau", col])
        if d.empty:
            return None
        return float(d.loc[d[col].astype(float).idxmin(), "Tau"])

    wsm_min_cost_tau = _min_tau_for(df_wsm_pareto, "Z_sys") if run_wsm else None
    wsm_min_down_tau = _min_tau_for(df_wsm_pareto, "V_sys") if run_wsm else None
    wsm_min_emis_tau = _min_tau_for(df_wsm_pareto, "E_sys") if run_wsm else None

    sga_min_cost_tau = _min_tau_for(df_sga, "Z_sys") if run_sga else None
    sga_min_down_tau = _min_tau_for(df_sga, "V_sys") if run_sga else None
    sga_min_emis_tau = _min_tau_for(df_sga, "E_sys") if run_sga else None

    # CPU time per trace (ms) — use total engine time for user-facing KPI
    cpu_time_per_trace_ms = (total_cpu_time / max(config.n_traces, 1)) * 1000.0

    final_metrics: List[Dict[str, Any]] = []
    final_metrics.append({"Metric": "Current Policy Cost", "Value": current_policy["Z_sys"]})
    final_metrics.append({"Metric": "Current Policy Downtime", "Value": current_policy["V_sys"]})
    final_metrics.append({"Metric": "Current Policy Emissions", "Value": current_policy["E_sys"]})
    final_metrics.append({"Metric": "Current Survival Avg", "Value": current_policy["survival_avg"]})
    final_metrics.append({"Metric": "Current Survived Traces", "Value": current_policy["survived_traces"]})

    if knee_row is not None:
        final_metrics.append({"Metric": "Optimal Policy Cost (Knee)", "Value": float(knee_row["Z_sys"])})
        final_metrics.append({"Metric": "Optimal Policy Downtime (Knee)", "Value": float(knee_row["V_sys"])})
        final_metrics.append({"Metric": "Optimal Policy Emissions (Knee)", "Value": float(knee_row["E_sys"])})
        final_metrics.append({"Metric": "Optimal Survival Avg", "Value": knee_surv_avg})
        final_metrics.append({"Metric": "Optimal Survived Traces", "Value": knee_surv_count})
    else:
        final_metrics.append({"Metric": "Optimal Policy Cost (Knee)", "Value": "N/A"})
        final_metrics.append({"Metric": "Optimal Policy Downtime (Knee)", "Value": "N/A"})
        final_metrics.append({"Metric": "Optimal Policy Emissions (Knee)", "Value": "N/A"})
        final_metrics.append({"Metric": "Optimal Survival Avg", "Value": "N/A"})
        final_metrics.append({"Metric": "Optimal Survived Traces", "Value": "N/A"})

    final_metrics.append({"Metric": "WSM Min Cost Tau", "Value": wsm_min_cost_tau})
    final_metrics.append({"Metric": "WSM Min Down Tau", "Value": wsm_min_down_tau})
    final_metrics.append({"Metric": "WSM Min Emis Tau", "Value": wsm_min_emis_tau})

    final_metrics.append({"Metric": "SGA Min Cost Tau", "Value": sga_min_cost_tau})
    final_metrics.append({"Metric": "SGA Min Down Tau", "Value": sga_min_down_tau})
    final_metrics.append({"Metric": "SGA Min Emis Tau", "Value": sga_min_emis_tau})

    final_metrics.append({"Metric": "Simulation CPU Time (s)", "Value": pass1_cpu_time})
    final_metrics.append({"Metric": "CPU Time per Trace (ms)", "Value": cpu_time_per_trace_ms})
    final_metrics.append({"Metric": "WSM Combos Evaluated", "Value": wsm_combos_evaluated})
    final_metrics.append({"Metric": "WSM Pareto Size", "Value": wsm_pareto_size})
    final_metrics.append({"Metric": "Standard GA Pareto Size", "Value": ga_pareto_size})
    final_metrics.append({"Metric": "Seeded GA Pareto Size", "Value": seeded_ga_pareto_size})

    df_final_report2 = pd.DataFrame(final_metrics)

    # -------------------------
    # 13) Excel export (overwrite each run)  ✅ includes requested missing sheets
    # -------------------------
    with pd.ExcelWriter(excel_path, engine="openpyxl", mode="w") as writer:
        # Inputs_Global
        globals_df = pd.DataFrame(
            [
                ("time_horizon_years", config.T),
                ("n_traces", config.n_traces),
                ("steps_per_year", config.steps_per_year),
                ("dt_years", config.dt_years),
                ("n_steps", n_steps),
                ("tau_years_list", str(tau_grid_years)),
                ("survival_target", config.survival_target),
                ("repair_or_replace_threshold", config.repair_or_replace_rule_threshold),
                ("sd_cost", config.c_SD),
                ("sd_downtime", config.d_SD),
                ("sd_emissions", config.e_SD),
                ("system_penalty_cost", config.system_penalty_cost),
                ("system_penalty_downtime", config.system_penalty_downtime),
                ("system_penalty_emission", config.system_penalty_emission),
                ("crn_max_events_abm_fbm", max_events),
                ("rng_seed", config.rng_seed),
                ("enforce_survival_constraint", config.enforce_survival_constraint),
                ("run_wsm", run_wsm),
                ("run_ga", run_ga),
                ("run_sga", run_sga),
            ],
            columns=["Parameter", "Value"],
        )
        globals_df.to_excel(writer, sheet_name="Inputs_Global", index=False)

        # Input component tables (as entered)
        model_inputs["cbm_df"].to_excel(writer, sheet_name="Inputs_CBM", index=False)
        model_inputs["abm_df"].to_excel(writer, sheet_name="Inputs_ABM", index=False)
        model_inputs["fbm_df"].to_excel(writer, sheet_name="Inputs_FBM", index=False)

        # Decision grids
        pd.DataFrame(
            {
                "Component": list(decision_grids.cbm_control_grids.keys()),
                "Control grid (1..H step 1)": [str(v) for v in decision_grids.cbm_control_grids.values()],
            }
        ).to_excel(writer, sheet_name="DecisionGrid_CBM", index=False)

        pd.DataFrame(
            {
                "Component": list(decision_grids.abm_age_grids.keys()),
                "Age grid (0.5..T step 0.5)": [str(v) for v in decision_grids.abm_age_grids.values()],
            }
        ).to_excel(writer, sheet_name="DecisionGrid_ABM", index=False)

        # PASS 1 exports (full)
        pd.DataFrame(
            [{"Name": k[0], "Tau": k[1], "Threshold": k[2], "Z": v[0], "V": v[1], "E": v[2], "p_surv": v[3]} for k, v in CBM_RESULTS.items()]
        ).to_excel(writer, sheet_name="Pass1_CBM", index=False)

        pd.DataFrame(
            [{"Name": k[0], "Tau": k[1], "Threshold": k[2], "Z": v[0], "V": v[1], "E": v[2], "p_surv": v[3]} for k, v in ABM_RESULTS.items()]
        ).to_excel(writer, sheet_name="Pass1_ABM", index=False)

        pd.DataFrame(
            [{"Name": k[0], "Tau": k[1], "Z": v[0], "V": v[1], "E": v[2], "p_surv": v[3]} for k, v in FBM_RESULTS.items()]
        ).to_excel(writer, sheet_name="Pass1_FBM", index=False)

        pd.DataFrame([{"pass1_cpu_time_s": pass1_cpu_time}]).to_excel(writer, sheet_name="Pass1_CPU", index=False)

        # Component Pareto menus
        component_menu_counts_df.to_excel(writer, sheet_name="Component_Menu_Counts", index=False)
        component_pareto_menus_df.to_excel(writer, sheet_name="Component_Pareto_Menus", index=False)

        # Component optima sheets ✅
        cbm_optima_df.to_excel(writer, sheet_name="CBM_Optima", index=False)
        abm_optima_df.to_excel(writer, sheet_name="ABM_Optima", index=False)
        fbm_optima_df.to_excel(writer, sheet_name="FBM_Optima", index=False)

        # System baseline per tau
        system_all_tau_df.to_excel(writer, sheet_name="System_Splice_AllTau", index=False)

        # Per_Tau_Extremes ✅
        df_per_tau_extremes.to_excel(writer, sheet_name="Per_Tau_Extremes", index=False)

        # WSM exports
        if run_wsm:
            df_wsm_all.to_excel(writer, sheet_name="WSM_System_Solutions", index=False)
            df_wsm_pareto.to_excel(writer, sheet_name="WSM_System_Pareto_PerTau", index=False)
            pd.DataFrame([{
                "wsm_cpu_time_s": wsm_cpu_time,
                "wsm_combos_evaluated": wsm_combos_evaluated,
                "wsm_pareto_size": wsm_pareto_size
            }]).to_excel(writer, sheet_name="WSM_Runtime", index=False)
        else:
            pd.DataFrame().to_excel(writer, sheet_name="WSM_System_Solutions", index=False)
            pd.DataFrame().to_excel(writer, sheet_name="WSM_System_Pareto_PerTau", index=False)
            pd.DataFrame([{"note": "WSM disabled"}]).to_excel(writer, sheet_name="WSM_Runtime", index=False)

        # GA exports
        df_ga.to_excel(writer, sheet_name="GA_System_Pareto", index=False)
        df_sga.to_excel(writer, sheet_name="SeededGA_System_Pareto", index=False)
        df_ga_tau.to_excel(writer, sheet_name="GA_Tau_Runtime", index=False)
        df_sga_tau.to_excel(writer, sheet_name="SeededGA_Tau_Runtime", index=False)

        pd.DataFrame([{
            "ga_cpu_time_s": ga_cpu_time,
            "seeded_ga_cpu_time_s": seeded_ga_cpu_time,
            "ga_pareto_size": ga_pareto_size,
            "seeded_ga_pareto_size": seeded_ga_pareto_size,
            "pop_size": GA_POP_SIZE,
            "num_gens": GA_NUM_GENS,
            "elite_target": ELITE_TARGET,
            "tournament_k": TOURNAMENT_K,
            "offspring_n": OFFSPRING_N,
            "mutation_rate": MUT_RATE,
            "use_cache": True,
        }]).to_excel(writer, sheet_name="GA_Runtime", index=False)

        # Final reporting sheets
        df_detailed_extremes.to_excel(writer, sheet_name="Detailed_Extremes", index=False)
        df_final_report2.to_excel(writer, sheet_name="Final_Report_2", index=False)

    # -------------------------
    # 14) Return summary to Streamlit
    # -------------------------
    summary = {
        "horizon_years": config.T,
        "traces": config.n_traces,
        "steps_per_year": config.steps_per_year,
        "dt_years": config.dt_years,
        "n_steps": n_steps,
        "tau_years_list": tau_grid_years,
        "cbm_components": len(cbm_components),
        "abm_components": len(abm_components),
        "fbm_components": len(fbm_components),
        "crn_max_events_abm_fbm": max_events,
        "pass1_cpu_time_s": pass1_cpu_time,
        "system_all_tau_rows": int(len(system_all_tau_df)),
        "wsm_enabled": run_wsm,
        "ga_enabled": run_ga,
        "sga_enabled": run_sga,
        "wsm_cpu_time_s": wsm_cpu_time,
        "wsm_combos_evaluated": wsm_combos_evaluated,
        "wsm_pareto_size": wsm_pareto_size,
        "ga_cpu_time_s": ga_cpu_time,
        "seeded_ga_cpu_time_s": seeded_ga_cpu_time,
        "ga_pareto_size": ga_pareto_size,
        "seeded_ga_pareto_size": seeded_ga_pareto_size,
        "total_engine_cpu_time_s": float(time.perf_counter() - t_total_start),
    }

    return EngineResult(
        excel_path=excel_path,
        plots_dir=plots_dir,
        message="FINAL EXPORT: Pass1 + Component Pareto + System Baseline + (optional) WSM/GA/Seeded GA + Per_Tau_Extremes + Optima + Reports exported to Excel.",
        summary=summary,
    )