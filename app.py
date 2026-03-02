# ==========================================
# PROMCS STREAMLIT APP (FINAL UX + DASHBOARD)
# Preventive Refurbishment Optimization for Multi-Component System
#
# Run:
#   streamlit run app.py
#
# Notes:
# - No sidebar
# - No sliders
# - Tabs: Inputs + Results Dashboard
# - Method toggles: WSM / GA / Seeded GA
# - Uses Plotly for interactive 3D/2D plots (rotate/zoom)
# - Excel is overwritten each run (engine behavior)
# ==========================================

from __future__ import annotations

import os
import io
import time
import zipfile
from datetime import datetime
from typing import Dict, List, Optional, Any
import numpy as np 
import streamlit as st
import pandas as pd

# --- Plotly (interactive charts)
try:
    import plotly.express as px
    import plotly.graph_objects as go
except Exception:
    st.error(
        "Plotly is not installed in this environment.\n\n"
        "Install it in your active environment:\n"
        "  python -m pip install plotly\n\n"
        "Then re-run:\n"
        "  streamlit run app.py"
    )
    st.stop()

from promcs_engine import run_promcs


# =========================================================
# Streamlit compatibility wrappers (avoids `use_container_width` breaking)
# =========================================================
def _df_show(df: pd.DataFrame, **kwargs):
    """Dataframe display that works on newer Streamlit (width='stretch') and older ones."""
    try:
        st.dataframe(df, width="stretch", **kwargs)
    except TypeError:
        st.dataframe(df, use_container_width=True, **kwargs)


def _editor(df: pd.DataFrame, **kwargs) -> pd.DataFrame:
    """Data editor display compatible across Streamlit versions."""
    try:
        return st.data_editor(df, width="stretch", **kwargs)
    except TypeError:
        return st.data_editor(df, use_container_width=True, **kwargs)


def _plotly(fig):
    """Plotly chart wrapper compatible across Streamlit versions."""
    try:
        st.plotly_chart(fig, use_container_width=True)
    except TypeError:
        st.plotly_chart(fig)


# =========================================================
# Page config + title
# =========================================================
st.set_page_config(
    page_title="Preventive Refurbishment Optimization for Multi-Component System",
    layout="wide",
)
st.title("Preventive Refurbishment Optimization for Multi-Component System")


# =========================================================
# Helpers
# =========================================================
def parse_tau_grid(tau_text: str, unit: str):
    """
    Parse comma-separated tau input and convert into years if needed.

    Inputs:
      - tau_text: e.g. "6, 12, 18" or "0.5, 1, 2"
      - unit: "Years" or "Months"

    Outputs:
      - tau_years_list: sorted unique list of floats in years
      - error_message: None if OK, else str
    """
    if tau_text is None or str(tau_text).strip() == "":
        return [], "Tau grid is empty. Enter values like: 0.5, 1, 2"

    tokens = [t.strip() for t in str(tau_text).split(",") if t.strip() != ""]
    if len(tokens) == 0:
        return [], "Tau grid is empty. Enter values like: 0.5, 1, 2"

    tau_vals = []
    for token in tokens:
        try:
            tau_vals.append(float(token))
        except Exception:
            return [], f"Tau grid contains a non-numeric value: '{token}'"

    if unit == "Months":
        tau_vals = [v / 12.0 for v in tau_vals]

    tau_vals = sorted(set(tau_vals))

    if any(v <= 0 for v in tau_vals):
        return [], "Tau values must be > 0."

    return tau_vals, None


def validate_component_table(df: pd.DataFrame, table_name: str):
    """
    Validate a component input table.
    Rules:
      - Qty must be an integer >= 1
      - rho (Kijima) must be between 0 and 1 (inclusive)
      - Name must not be empty
    """
    errors = []
    if df is None or df.empty:
        return errors

    temp = df.copy()

    required_cols = ["Name", "Qty", "rho (Kijima)"]
    for col in required_cols:
        if col not in temp.columns:
            errors.append(f"{table_name}: Missing required column '{col}'.")
            return errors

    for idx in range(len(temp)):
        row_num = idx + 1

        name_val = str(temp.at[idx, "Name"]).strip() if temp.at[idx, "Name"] is not None else ""
        if name_val == "":
            errors.append(f"{table_name} row {row_num}: Name is empty.")

        # Qty check
        try:
            qty_val = int(round(float(temp.at[idx, "Qty"])))
            if qty_val < 1:
                errors.append(f"{table_name} row {row_num}: Qty must be >= 1.")
        except Exception:
            errors.append(f"{table_name} row {row_num}: Qty must be a number (integer >= 1).")

        # rho check
        try:
            rho_val = float(temp.at[idx, "rho (Kijima)"])
            if not (0.0 <= rho_val <= 1.0):
                errors.append(f"{table_name} row {row_num}: rho (Kijima) must be between 0 and 1.")
        except Exception:
            errors.append(f"{table_name} row {row_num}: rho (Kijima) must be a number between 0 and 1.")

    return errors


# =========================================================
# Default rows (ASCII labels for Windows safety)
# =========================================================
def default_cbm_row():
    return {
        "Name": "CBM_Component_1",
        "Qty": 1,
        "Class": "repairable_replaceable",
        "Criticality": "Non-Critical",
        "Gamma shape (k)": 1.0,
        "Gamma scale (eta)": 1.0,
        "Initial Level (x_i)": 0.0,
        "Failure threshold (H_i)": 1.0,
        "rho (Kijima)": 0.9,
        "PM Rep cost (EUR)": 0.0,
        "PM Repl cost (EUR)": 0.0,
        "CM Repl cost (EUR)": 0.0,
        "PM Rep down (h)": 0.0,
        "PM Repl down (h)": 0.0,
        "CM Repl down (h)": 0.0,
        "PM Rep emis (kg CO2)": 0.0,
        "PM Repl emis (kg CO2)": 0.0,
        "CM Repl emis (kg CO2)": 0.0,
        "Delete": False,
    }


def default_abm_row():
    return {
        "Name": "ABM_Component_1",
        "Qty": 1,
        "Class": "repairable_replaceable",
        "Criticality": "Non-Critical",
        "Weibull shape (beta)": 1.5,
        "Weibull scale (alpha)": 1.0,
        "rho (Kijima)": 0.9,
        "PM Rep cost (EUR)": 0.0,
        "PM Repl cost (EUR)": 0.0,
        "CM Repl cost (EUR)": 0.0,
        "PM Rep down (h)": 0.0,
        "PM Repl down (h)": 0.0,
        "CM Repl down (h)": 0.0,
        "PM Rep emis (kg CO2)": 0.0,
        "PM Repl emis (kg CO2)": 0.0,
        "CM Repl emis (kg CO2)": 0.0,
        "Delete": False,
    }


def default_fbm_row():
    return {
        "Name": "FBM_Component_1",
        "Qty": 1,
        "Class": "replaceable_only",
        "Criticality": "Non-Critical",
        "Weibull shape (beta)": 1.5,
        "Weibull scale (alpha)": 1.0,
        "rho (Kijima)": 0.9,
        "CM Repl cost (EUR)": 0.0,
        "CM Repl down (h)": 0.0,
        "CM Repl emis (kg CO2)": 0.0,
        "Delete": False,
    }


def reset_all_inputs():
    for k in ["model_inputs", "engine_result", "excel_cache", "excel_data_cache"]:
        if k in st.session_state:
            del st.session_state[k]

    st.session_state["cbm_df"] = pd.DataFrame([default_cbm_row()])
    st.session_state["abm_df"] = pd.DataFrame([default_abm_row()])
    st.session_state["fbm_df"] = pd.DataFrame([default_fbm_row()])

    st.rerun()


# =========================================================
# Session state init
# =========================================================
if "cbm_df" not in st.session_state:
    st.session_state["cbm_df"] = pd.DataFrame([default_cbm_row()])
if "abm_df" not in st.session_state:
    st.session_state["abm_df"] = pd.DataFrame([default_abm_row()])
if "fbm_df" not in st.session_state:
    st.session_state["fbm_df"] = pd.DataFrame([default_fbm_row()])


# =========================================================
# Excel loading (cached)
# =========================================================
@st.cache_data(show_spinner=False)
def _read_excel_cached(excel_path: str, mtime: float) -> Dict[str, pd.DataFrame]:
    """
    Read specific sheets if they exist. Cached by (path + modified time).
    """
    sheets_needed = [
        # core
        "Component_Pareto_Menus",
        "System_Splice_AllTau",
        "WSM_System_Pareto_PerTau",
        "GA_System_Pareto",
        "SeededGA_System_Pareto",
        # reports
        "Detailed_Extremes",
        "Final_Report_2",
        # optional future sheets (if your engine adds them)
        "Per_Tau_Extremes",
        "CBM_Optima",
        "ABM_Optima",
        "FBM_Optima",
    ]

    out: Dict[str, pd.DataFrame] = {}
    try:
        xls = pd.ExcelFile(excel_path)
        for s in sheets_needed:
            if s in xls.sheet_names:
                out[s] = pd.read_excel(excel_path, sheet_name=s)
            else:
                out[s] = pd.DataFrame()
    except Exception:
        for s in sheets_needed:
            out[s] = pd.DataFrame()

    return out


def _coerce_numeric(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def _make_3d_system_plot(df: pd.DataFrame, title: str):
    if df.empty:
        st.info("No data available for 3D system plot.")
        return

    fig = px.scatter_3d(
        df,
        x="Z_sys",
        y="V_sys",
        z="E_sys",
        color="Method",
        symbol="Method",
        hover_data=["Tau", "p_surv_sys"],
        title=title,
    )
    fig.update_layout(height=650, legend_title_text="Method")
    _plotly(fig)


def _make_2d_system_plot(df: pd.DataFrame, x: str, y: str, title: str):
    if df.empty:
        st.info("No data available for 2D system plot.")
        return

    fig = px.scatter(
        df,
        x=x,
        y=y,
        color="Method",
        hover_data=["Tau", "p_surv_sys"],
        title=title,
    )
    fig.update_layout(height=450)
    _plotly(fig)


def _make_3d_component_plot(df: pd.DataFrame, title: str):
    """
    Component Pareto menu: Z vs V vs E, hover Threshold (+p_surv if present).
    Expected columns: Z, V, E, Threshold
    """
    if df.empty:
        st.info("No data available for 3D component plot.")
        return

    hover_cols = ["Threshold"]
    if "p_surv" in df.columns:
        hover_cols.append("p_surv")

    fig = px.scatter_3d(
        df,
        x="Z",
        y="V",
        z="E",
        hover_data=hover_cols,
        title=title,
    )
    fig.update_layout(height=650)
    _plotly(fig)


def _make_2d_component_plot(df: pd.DataFrame, x: str, y: str, title: str):
    if df.empty:
        st.info("No data available for 2D component plot.")
        return

    hover_cols = ["Threshold"]
    if "p_surv" in df.columns:
        hover_cols.append("p_surv")

    fig = px.scatter(
        df,
        x=x,
        y=y,
        hover_data=hover_cols,
        title=title,
    )
    fig.update_layout(height=450)
    _plotly(fig)


# =========================================================
# Tabs + reset
# =========================================================
tab_inputs, tab_dashboard = st.tabs(["Inputs", "Results Dashboard"])

reset_col1, reset_col2 = st.columns([4, 1])
with reset_col2:
    if st.button("Reset Inputs"):
        reset_all_inputs()


# =========================================================
# TAB 1: INPUTS & RUN
# =========================================================
with tab_inputs:
    st.subheader("Global Inputs")

    col_a, col_b, col_c = st.columns([1, 1, 1])

    with col_a:
        time_horizon_years = st.number_input("Time horizon (years)", min_value=1, max_value=100, step=1, value=20)
        n_traces = st.number_input("Number of traces", min_value=1, max_value=2000, step=1, value=500)
        crn_bank_size = st.number_input("CRN bank size (recommended >= traces)", min_value=1, max_value=5000, step=1, value=1000)
        survival_probability = st.number_input("Survival probability target p_sur (0-1)", min_value=0.0, max_value=1.0, value=0.90, step=0.01)

    with col_b:
        tau_unit = st.selectbox("Tau unit", ["Years", "Months"], index=0)
        tau_text = st.text_input("Tau grid (Intervals)", value="0.5, 1, 2, 3")

        sd_cost = st.number_input("Scheduled Down cost (EUR)", value=0.0)
        sd_downtime = st.number_input("Scheduled Down downtime (hours)", value=0.0)
        sd_emissions = st.number_input("Scheduled Down emissions (kg CO2)", value=0.0)

    with col_c:
        threshold_options = [0, 0.25, 0.33, 0.50, 0.66, 0.75, 0.85, 0.90, 1]
        p_major_threshold = st.selectbox("Repair vs replacement threshold", threshold_options, index=threshold_options.index(0.66))

        system_penalty_cost = st.number_input("System penalty cost (EUR)", value=30000.0)
        system_penalty_downtime = st.number_input("System penalty downtime (hours)", value=3024.0)
        system_penalty_emissions = st.number_input("System penalty emissions (kg CO2)", value=26894.57)

        st.caption("steps_per_year = 12 (fixed)")
        st.caption("dt_years = 1/12 (fixed)")
        st.info("Imperfect repair is controlled via rho (Kijima factor) per component in CBM/ABM/FBM tables.")

    tau_years_list, tau_error = parse_tau_grid(tau_text, tau_unit)
    if tau_error:
        st.error(tau_error)
    else:
        st.success(f"Tau grid parsed (years): {tau_years_list}")

    # --- Performance toggles
    st.subheader("Method Execution Options")
    m1, m2, m3 = st.columns([1, 1, 1])
    with m1:
        run_wsm = st.checkbox("Run WSM", value=True)
    with m2:
        run_ga = st.checkbox("Run Standard GA", value=True)
    with m3:
        run_sga = st.checkbox("Run Seeded GA", value=True)

    if not (run_wsm or run_ga or run_sga):
        st.warning("All methods are disabled. Enable at least one of WSM / GA / Seeded GA.")

    # =========================================================
    # Component tables
    # =========================================================
    st.subheader("Component Inputs")

    class_options = ["repairable_replaceable", "repairable_only", "replaceable_only"]
    criticality_options = ["Critical", "Non-Critical"]

    # --- CBM
    st.markdown("### CBM Components")
    cbm_add_col, cbm_rm_col = st.columns([1, 1])

    with cbm_add_col:
        if st.button("Add CBM component"):
            st.session_state["cbm_df"] = pd.concat([st.session_state["cbm_df"], pd.DataFrame([default_cbm_row()])], ignore_index=True)

    with cbm_rm_col:
        if st.button("Remove marked CBM rows"):
            df = st.session_state["cbm_df"].copy()
            df = df[df["Delete"] == False].reset_index(drop=True)
            if "Delete" in df.columns:
                df["Delete"] = False
            st.session_state["cbm_df"] = df
            st.rerun()

    st.session_state["cbm_df"] = _editor(
        st.session_state["cbm_df"],
        num_rows="dynamic",
        column_config={
            "Class": st.column_config.SelectboxColumn("Class", options=class_options),
            "Criticality": st.column_config.SelectboxColumn("Criticality", options=criticality_options),
            "Delete": st.column_config.CheckboxColumn("Delete"),
        },
    )

    # --- ABM
    st.markdown("### ABM Components")
    abm_add_col, abm_rm_col = st.columns([1, 1])

    with abm_add_col:
        if st.button("Add ABM component"):
            st.session_state["abm_df"] = pd.concat([st.session_state["abm_df"], pd.DataFrame([default_abm_row()])], ignore_index=True)

    with abm_rm_col:
        if st.button("Remove marked ABM rows"):
            df = st.session_state["abm_df"].copy()
            df = df[df["Delete"] == False].reset_index(drop=True)
            if "Delete" in df.columns:
                df["Delete"] = False
            st.session_state["abm_df"] = df
            st.rerun()

    st.session_state["abm_df"] = _editor(
        st.session_state["abm_df"],
        num_rows="dynamic",
        column_config={
            "Class": st.column_config.SelectboxColumn("Class", options=class_options),
            "Criticality": st.column_config.SelectboxColumn("Criticality", options=criticality_options),
            "Delete": st.column_config.CheckboxColumn("Delete"),
        },
    )

    # --- FBM
    st.markdown("### FBM Components")
    fbm_add_col, fbm_rm_col = st.columns([1, 1])

    with fbm_add_col:
        if st.button("Add FBM component"):
            st.session_state["fbm_df"] = pd.concat([st.session_state["fbm_df"], pd.DataFrame([default_fbm_row()])], ignore_index=True)

    with fbm_rm_col:
        if st.button("Remove marked FBM rows"):
            df = st.session_state["fbm_df"].copy()
            df = df[df["Delete"] == False].reset_index(drop=True)
            if "Delete" in df.columns:
                df["Delete"] = False
            st.session_state["fbm_df"] = df
            st.rerun()

    st.session_state["fbm_df"] = _editor(
        st.session_state["fbm_df"],
        num_rows="dynamic",
        column_config={
            "Class": st.column_config.SelectboxColumn("Class", options=class_options),
            "Criticality": st.column_config.SelectboxColumn("Criticality", options=criticality_options),
            "Delete": st.column_config.CheckboxColumn("Delete"),
        },
    )

    # =========================================================
    # Input summary
    # =========================================================
    st.subheader("Input Summary")

    left, right = st.columns([1, 1])
    with left:
        st.markdown("**Global**")
        st.write(f"Horizon (years): {int(time_horizon_years)}")
        st.write(f"Traces: {int(n_traces)}")
        st.write(f"CRN bank size: {int(crn_bank_size)}")
        st.write(f"Survival target p_sur: {float(survival_probability)}")
        st.write(f"Repair vs replacement threshold: {float(p_major_threshold)}")

        st.markdown("**System penalties**")
        st.write(f"Cost (EUR): {float(system_penalty_cost)}")
        st.write(f"Downtime (h): {float(system_penalty_downtime)}")
        st.write(f"Emissions (kg CO2): {float(system_penalty_emissions)}")

    with right:
        st.markdown("**Tau grid**")
        if tau_error:
            st.error("Invalid tau grid")
        else:
            st.write(f"Unit: {tau_unit}")
            st.write(f"Values (years): {tau_years_list}")

        st.markdown("**Components**")
        comp_counts = pd.DataFrame(
            {
                "Type": ["CBM", "ABM", "FBM"],
                "Rows": [len(st.session_state["cbm_df"]), len(st.session_state["abm_df"]), len(st.session_state["fbm_df"])],
            }
        )
        _df_show(comp_counts, hide_index=True)

        st.markdown("**Methods**")
        st.write(f"Run WSM: {run_wsm}")
        st.write(f"Run GA: {run_ga}")
        st.write(f"Run Seeded GA: {run_sga}")

    # =========================================================
    # Run
    # =========================================================
    st.divider()
    st.subheader("Run")

    run_clicked = st.button("Run Simulation", type="primary")

    if run_clicked:
        with st.status("Processing...", expanded=True) as status:
            start_time = time.perf_counter()

            st.write("Step 1/3: Validating inputs...")

            if tau_error:
                status.update(label="Validation failed", state="error")
                st.stop()

            if crn_bank_size < n_traces:
                status.update(label="Validation failed", state="error")
                st.error("CRN bank size should be >= number of traces (recommended).")
                st.stop()

            total_rows = len(st.session_state["cbm_df"]) + len(st.session_state["abm_df"]) + len(st.session_state["fbm_df"])
            if total_rows == 0:
                status.update(label="Validation failed", state="error")
                st.error("No components provided. Add at least 1 component.")
                st.stop()

            errors = []
            errors += validate_component_table(st.session_state["cbm_df"], "CBM")
            errors += validate_component_table(st.session_state["abm_df"], "ABM")
            errors += validate_component_table(st.session_state["fbm_df"], "FBM")
            if errors:
                status.update(label="Validation failed", state="error")
                for msg in errors[:15]:
                    st.error(msg)
                if len(errors) > 15:
                    st.error(f"...and {len(errors) - 15} more errors.")
                st.stop()

            if not (run_wsm or run_ga or run_sga):
                status.update(label="Validation failed", state="error")
                st.error("Enable at least one method (WSM / GA / Seeded GA).")
                st.stop()

            st.write("Step 2/3: Preparing model inputs...")
            st.session_state["model_inputs"] = {
                "time_horizon_years": int(time_horizon_years),
                "n_traces": int(n_traces),
                "crn_bank_size": int(crn_bank_size),
                "survival_probability": float(survival_probability),
                "tau_years_list": tau_years_list,
                "sd_cost": float(sd_cost),
                "sd_downtime": float(sd_downtime),
                "sd_emissions": float(sd_emissions),
                "p_major_threshold": float(p_major_threshold),
                "system_penalty_cost": float(system_penalty_cost),
                "system_penalty_downtime": float(system_penalty_downtime),
                "system_penalty_emissions": float(system_penalty_emissions),
                "cbm_df": st.session_state["cbm_df"].copy(),
                "abm_df": st.session_state["abm_df"].copy(),
                "fbm_df": st.session_state["fbm_df"].copy(),
                # method flags (engine may use these)
                "run_wsm": bool(run_wsm),
                "run_ga": bool(run_ga),
                "run_sga": bool(run_sga),
            }

            st.write("Step 3/3: Running engine (WSM/GA can be heavy)...")
            bar = st.progress(0, text="Starting engine...")
            bar.progress(10, text="Launching engine...")

            try:
                engine_result = run_promcs(st.session_state["model_inputs"], output_dir="outputs")
            except Exception as e:
                bar.progress(100, text="Engine failed.")
                status.update(label="Engine error", state="error")
                st.exception(e)
                st.stop()

            bar.progress(100, text="Engine finished. Loading dashboard...")

            st.session_state["engine_result"] = engine_result
            st.session_state["excel_cache"] = {"path": engine_result.excel_path, "loaded_at": datetime.now().isoformat()}

            elapsed = time.perf_counter() - start_time
            status.update(label=f"Completed in {elapsed:.1f}s", state="complete")

        st.success("Run finished. Open the Results Dashboard tab.")


# =========================================================
# TAB 2: DASHBOARD
# =========================================================
with tab_dashboard:
    st.subheader("Results Dashboard")

    if "engine_result" not in st.session_state:
        st.info("Run the model from the Inputs tab first.")
        st.stop()

    result = st.session_state["engine_result"]
    st.success("Engine executed successfully.")
    st.write(result.message)

    st.markdown("**Key Summary**")
    st.json(result.summary)

    if not os.path.exists(result.excel_path):
        st.error("Excel output file not found. Please re-run.")
        st.stop()

    # Load excel sheets (cached by modification time)
    mtime = os.path.getmtime(result.excel_path)
    data = _read_excel_cached(result.excel_path, mtime)

    df_sys_tau = data.get("System_Splice_AllTau", pd.DataFrame()).copy()
    df_wsm = data.get("WSM_System_Pareto_PerTau", pd.DataFrame()).copy()
    df_ga = data.get("GA_System_Pareto", pd.DataFrame()).copy()
    df_sga = data.get("SeededGA_System_Pareto", pd.DataFrame()).copy()
    df_comp = data.get("Component_Pareto_Menus", pd.DataFrame()).copy()
    df_ext = data.get("Detailed_Extremes", pd.DataFrame()).copy()
    df_report = data.get("Final_Report_2", pd.DataFrame()).copy()

    df_per_tau_ext = data.get("Per_Tau_Extremes", pd.DataFrame()).copy()
    df_cbm_opt = data.get("CBM_Optima", pd.DataFrame()).copy()
    df_abm_opt = data.get("ABM_Optima", pd.DataFrame()).copy()
    df_fbm_opt = data.get("FBM_Optima", pd.DataFrame()).copy()

    # ---------------------------------------------------------
    # Downloads
    # ---------------------------------------------------------
    c1, c2 = st.columns([1, 1])
    with c1:
        with open(result.excel_path, "rb") as f:
            st.download_button(
                label="Download Excel (Inputs + Outputs)",
                data=f,
                file_name=os.path.basename(result.excel_path),
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
    with c2:
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.write(result.excel_path, arcname=os.path.basename(result.excel_path))
            if os.path.isdir(result.plots_dir):
                for root, _, files in os.walk(result.plots_dir):
                    for file in files:
                        full_path = os.path.join(root, file)
                        rel_path = os.path.relpath(full_path, start=os.path.dirname(result.plots_dir))
                        zf.write(full_path, arcname=rel_path)
        zip_buffer.seek(0)

        st.download_button(
            label="Download All Outputs (ZIP)",
            data=zip_buffer,
            file_name="PROMCS_outputs.zip",
            mime="application/zip",
        )

    st.divider()

    # ---------------------------------------------------------
    # Final report tables
    # ---------------------------------------------------------
    st.markdown("## Final Report")
    if df_report.empty:
        st.info("Final_Report_2 sheet not found or empty.")
    else:
        _df_show(df_report, hide_index=True)

    st.markdown("## Detailed Extremes")
    if df_ext.empty:
        st.info("Detailed_Extremes sheet not found or empty.")
    else:
        _df_show(df_ext, hide_index=True)

    if not df_per_tau_ext.empty:
        st.markdown("## Per Tau Extremes")
        _df_show(df_per_tau_ext, hide_index=True)

    with st.expander("Component Optima Sheets (if available)", expanded=False):
        if df_cbm_opt.empty and df_abm_opt.empty and df_fbm_opt.empty:
            st.info("CBM_Optima / ABM_Optima / FBM_Optima sheets not found yet.")
        else:
            if not df_cbm_opt.empty:
                st.markdown("### CBM_Optima")
                _df_show(df_cbm_opt, hide_index=True)
            if not df_abm_opt.empty:
                st.markdown("### ABM_Optima")
                _df_show(df_abm_opt, hide_index=True)
            if not df_fbm_opt.empty:
                st.markdown("### FBM_Optima")
                _df_show(df_fbm_opt, hide_index=True)

    st.divider()

    # ---------------------------------------------------------
    # SYSTEM-LEVEL PARETO PLOTS
    # ---------------------------------------------------------
    st.markdown("## System-Level Pareto Plots")

    frames = []
    if not df_wsm.empty:
        df_wsm2 = _coerce_numeric(df_wsm, ["Tau", "Z_sys", "V_sys", "E_sys", "p_surv_sys"]).copy()
        df_wsm2["Method"] = "WSM"
        cols = [c for c in ["Tau", "Z_sys", "V_sys", "E_sys", "p_surv_sys", "Method"] if c in df_wsm2.columns]
        frames.append(df_wsm2[cols])
    if not df_ga.empty:
        df_ga2 = _coerce_numeric(df_ga, ["Tau", "Z_sys", "V_sys", "E_sys", "p_surv_sys"]).copy()
        df_ga2["Method"] = "GA"
        cols = [c for c in ["Tau", "Z_sys", "V_sys", "E_sys", "p_surv_sys", "Method"] if c in df_ga2.columns]
        frames.append(df_ga2[cols])
    if not df_sga.empty:
        df_sga2 = _coerce_numeric(df_sga, ["Tau", "Z_sys", "V_sys", "E_sys", "p_surv_sys"]).copy()
        df_sga2["Method"] = "Seeded GA"
        cols = [c for c in ["Tau", "Z_sys", "V_sys", "E_sys", "p_surv_sys", "Method"] if c in df_sga2.columns]
        frames.append(df_sga2[cols])

    df_sys_pareto = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    if df_sys_pareto.empty:
        st.info("No system Pareto data available (WSM / GA / Seeded GA sheets empty).")
    else:
        tau_vals = sorted([float(x) for x in df_sys_pareto["Tau"].dropna().unique().tolist()])
        method_vals = sorted(df_sys_pareto["Method"].dropna().unique().tolist())

        f1, f2, f3 = st.columns([1, 2, 1])
        with f1:
            selected_tau = st.selectbox("Tau (years) for plotting", options=["All"] + tau_vals, index=0)
        with f2:
            selected_methods = st.multiselect("Methods", options=method_vals, default=method_vals)
        with f3:
            show_only_feasible = st.checkbox("Hide infeasible (inf)", value=True)

        df_plot = df_sys_pareto.copy()
        if selected_tau != "All":
            df_plot = df_plot[df_plot["Tau"] == float(selected_tau)]
        if selected_methods:
            df_plot = df_plot[df_plot["Method"].isin(selected_methods)]

        if show_only_feasible:
            df_plot = df_plot.replace([np.inf, -np.inf], np.nan).dropna(subset=["Z_sys", "V_sys", "E_sys"])

        _make_3d_system_plot(df_plot, "System Pareto (3D): Cost vs Downtime vs Emissions")

        st.markdown("### System 2D Projections")
        x1, x2, x3 = st.columns([1, 1, 1])
        with x1:
            _make_2d_system_plot(df_plot, "Z_sys", "V_sys", "Cost vs Downtime")
        with x2:
            _make_2d_system_plot(df_plot, "Z_sys", "E_sys", "Cost vs Emissions")
        with x3:
            _make_2d_system_plot(df_plot, "V_sys", "E_sys", "Downtime vs Emissions")

    st.divider()

    # ---------------------------------------------------------
    # Baseline per tau (System_Splice_AllTau)
    # ---------------------------------------------------------
    st.markdown("## System Baseline per Tau (System_Splice_AllTau)")

    if df_sys_tau.empty:
        st.info("System_Splice_AllTau sheet missing or empty.")
    else:
        df_sys_tau = _coerce_numeric(df_sys_tau, ["Tau", "Z_sys", "V_sys", "E_sys", "p_sur_sys"])
        df_sys_tau = df_sys_tau.dropna(subset=["Tau"]).sort_values("Tau")

        fig1 = px.line(df_sys_tau, x="Tau", y="Z_sys", title="Baseline Cost Rate vs Tau")
        fig2 = px.line(df_sys_tau, x="Tau", y="V_sys", title="Baseline Downtime Rate vs Tau")
        fig3 = px.line(df_sys_tau, x="Tau", y="E_sys", title="Baseline Emission Rate vs Tau")
        if "p_sur_sys" in df_sys_tau.columns:
            fig4 = px.line(df_sys_tau, x="Tau", y="p_sur_sys", title="Baseline System Survival vs Tau")
        else:
            fig4 = None

        _plotly(fig1)
        _plotly(fig2)
        _plotly(fig3)
        if fig4 is not None:
            _plotly(fig4)

    st.divider()

    # ---------------------------------------------------------
    # COMPONENT-LEVEL ANALYSIS (Pareto menus)
    # ---------------------------------------------------------
    st.markdown("## Component-Level Pareto Menus")

    if df_comp.empty:
        st.info("Component_Pareto_Menus sheet missing or empty.")
    else:
        needed = {"Component", "Tau", "Threshold", "Z", "V", "E"}
        if not needed.issubset(set(df_comp.columns)):
            st.error(f"Component_Pareto_Menus is missing required columns: {sorted(list(needed))}")
        else:
            df_comp = _coerce_numeric(df_comp, ["Tau", "Z", "V", "E", "p_surv"])
            comp_list = sorted(df_comp["Component"].dropna().unique().tolist())
            tau_list = sorted([float(x) for x in df_comp["Tau"].dropna().unique().tolist()])

            s1, s2, s3 = st.columns([2, 1, 1])
            with s1:
                selected_comp = st.selectbox("Component", options=comp_list)
            with s2:
                selected_tau2 = st.selectbox("Tau (years)", options=tau_list)
            with s3:
                view_type = st.selectbox("View", options=["3D Pareto", "2D Cost vs Down", "2D Cost vs Emis", "2D Down vs Emis"])

            df_c = df_comp[(df_comp["Component"] == selected_comp) & (df_comp["Tau"] == float(selected_tau2))].copy()
            if df_c.empty:
                st.info("No Pareto menu points for this selection.")
            else:
                # Make sure Threshold is displayed nicely
                if "Threshold" not in df_c.columns:
                    df_c["Threshold"] = None

                if view_type == "3D Pareto":
                    _make_3d_component_plot(df_c, f"{selected_comp} Pareto Menu (Tau={selected_tau2})")
                elif view_type == "2D Cost vs Down":
                    _make_2d_component_plot(df_c, "Z", "V", f"{selected_comp}: Cost vs Downtime (Tau={selected_tau2})")
                elif view_type == "2D Cost vs Emis":
                    _make_2d_component_plot(df_c, "Z", "E", f"{selected_comp}: Cost vs Emissions (Tau={selected_tau2})")
                else:
                    _make_2d_component_plot(df_c, "V", "E", f"{selected_comp}: Downtime vs Emissions (Tau={selected_tau2})")

            with st.expander("Show underlying Pareto menu table for this selection", expanded=False):
                _df_show(df_c.reset_index(drop=True), hide_index=True)