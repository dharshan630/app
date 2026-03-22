"""
Usage:
    streamlit run employee_engagement_dashboard.py
"""

from __future__ import annotations

import math
from datetime import datetime
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd
import streamlit as st
import altair as alt

# Default dataset file (auto-loaded if present)
DEFAULT_DATA_FILE = "Palo Alto Networks.csv"
DEFAULT_DATA_PATH = Path(__file__).parent / DEFAULT_DATA_FILE


@st.cache_data
def _read_csv_cached(path: str, mtime: float, refresh_counter: int) -> pd.DataFrame:
    """Read CSV with caching that invalidates when the file changes or user refreshes."""
    return pd.read_csv(path)


def load_dataset(path: Path, refresh_counter: int = 0) -> pd.DataFrame:
    """Load dataset from disk with caching and file-change awareness."""
    if not path.exists():
        raise FileNotFoundError(path)

    mtime = path.stat().st_mtime
    df = _read_csv_cached(str(path), mtime, refresh_counter)
    return df


def _format_percentage(value: float, decimals: int = 1) -> str:
    """Format a numeric value as a percentage string."""
    return f"{value:.{decimals}f}%"


def build_executive_summary(
    df: pd.DataFrame,
    dataset_source: str,
    last_modified: Optional[str],
    filters: dict,
) -> str:
    """Build a text executive summary for download/export."""

    total = len(df)
    avg_eng = df["EngagementIndex"].mean() if "EngagementIndex" in df else float("nan")
    avg_wlb = df["WorkLifeBalance"].mean() if "WorkLifeBalance" in df else float("nan")
    burn_risk = (
        df["BurnoutRiskLevel"].value_counts(normalize=True).mul(100).round(1)
        if "BurnoutRiskLevel" in df
        else pd.Series(dtype=float)
    )

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    md_lines = [
        f"# Executive Summary — Employee Engagement & Burnout Risk",
        "",
        f"**Report generated:** {now}",
        f"**Dataset source:** {dataset_source}",
    ]

    if last_modified:
        md_lines.append(f"**Dataset last modified:** {last_modified}")

    md_lines += [
        "",
        "## Filters applied",
        "",
    ]

    for key, val in filters.items():
        if val is None or (isinstance(val, (list, tuple)) and len(val) == 0):
            continue
        md_lines.append(f"- **{key}**: {val}")

    md_lines += [
        "",
        "## Key metrics",
        "",
        f"- Total employees analyzed: **{total:,}**",
        f"- Average engagement index: **{avg_eng:.1f}/100**",
        f"- Average work-life balance: **{avg_wlb:.1f}/4**",
    ]

    if not burn_risk.empty:
        high_pct = burn_risk.get("High", 0.0)
        med_pct = burn_risk.get("Medium", 0.0)
        low_pct = burn_risk.get("Low", 0.0)
        md_lines += [
            f"- Burnout risk distribution: **High {high_pct:.1f}%**, **Medium {med_pct:.1f}%**, **Low {low_pct:.1f}%**",
        ]

    # Add a short section on lowest engagement by role/department
    if "JobRole" in df and "EngagementIndex" in df:
        low_roles = (
            df.groupby("JobRole")["EngagementIndex"].mean().nsmallest(3).round(1)
        )
        md_lines += ["", "### Lowest engagement by job role"]
        for role, val in low_roles.items():
            md_lines.append(f"- {role}: {val:.1f}")

    if "Department" in df and "BurnoutRiskLevel" in df:
        dept_risk = (
            df.groupby("Department")["BurnoutRiskLevel"]
            .apply(lambda s: (s == "High").mean() * 100)
            .sort_values(ascending=False)
            .head(3)
            .round(1)
        )
        if not dept_risk.empty:
            md_lines += ["", "### Highest high-risk departments"]
            for dept, val in dept_risk.items():
                md_lines.append(f"- {dept}: {val:.1f}% high risk")

    md_lines += [
        "",
        "---",
        "*This report is diagnostic and designed to surface engagement gaps and burnout signals. It does not predict attrition but highlights leading indicators for proactive interventions.*",
    ]

    return "\n".join(md_lines)


# -----------------------------------------------------------------------------
# Data helpers
# -----------------------------------------------------------------------------

REQUIRED_COLUMNS: list[str] = [
    "Age",
    "Attrition",
    "BusinessTravel",
    "Department",
    "DistanceFromHome",
    "Education",
    "EducationField",
    "EnvironmentSatisfaction",
    "Gender",
    "HourlyRate",
    "JobInvolvement",
    "JobLevel",
    "JobRole",
    "JobSatisfaction",
    "MaritalStatus",
    "MonthlyIncome",
    "MonthlyRate",
    "NumCompaniesWorked",
    "OverTime",
    "PercentSalaryHike",
    "PerformanceRating",
    "RelationshipSatisfaction",
    "StockOptionLevel",
    "TotalWorkingYears",
    "TrainingTimesLastYear",
    "WorkLifeBalance",
    "YearsAtCompany",
    "YearsInCurrentRole",
    "YearsSinceLastPromotion",
    "YearsWithCurrManager",
]

ENGAGEMENT_COLS = [
    "JobInvolvement",
    "JobSatisfaction",
    "EnvironmentSatisfaction",
    "RelationshipSatisfaction",
]


def _normalize_ordinal_series(s: pd.Series, valid_min=1, valid_max=4) -> pd.Series:
    """Normalize an ordinal (1-4) series to be within [valid_min, valid_max]."""
    s_numeric = pd.to_numeric(s, errors="coerce")
    # Clip to expected range and fill with median when missing.
    s_clean = s_numeric.clip(valid_min, valid_max)
    median = s_clean.median()
    return s_clean.fillna(median)


def compute_engagement_index(df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-employee engagement index (0–100) based on 4 satisfaction fields."""

    for col in ENGAGEMENT_COLS:
        if col in df.columns:
            df[col] = _normalize_ordinal_series(df[col])

    missing = [c for c in ENGAGEMENT_COLS if c not in df.columns]
    if missing:
        st.warning(
            f"Missing engagement dimension(s): {missing}. Engagement index will use available columns."
        )

    available = [c for c in ENGAGEMENT_COLS if c in df.columns]
    if not available:
        df["EngagementIndex"] = np.nan
        return df

    # Mean of the engagement dimensions (1–4), then scale to 0–100.
    df["EngagementMean"] = df[available].mean(axis=1)
    df["EngagementIndex"] = ((df["EngagementMean"] - 1) / 3) * 100
    df["EngagementIndex"] = df["EngagementIndex"].clip(0, 100)

    # Satisfaction stability: how much the four satisfaction metrics diverge.
    # Lower stddev = more stable; normalize to 0–100.
    df["EngagementStd"] = df[available].std(axis=1)
    max_std = 1.118  # theoretical max for 1-4 scale
    df["SatisfactionStabilityScore"] = (
        (1 - (df["EngagementStd"] / max_std)) * 100
    ).clip(0, 100, )

    return df


def compute_burnout_risk(df: pd.DataFrame) -> pd.DataFrame:
    """Compute burnout risk level and score based on overtime & work-life balance."""

    if "OverTime" in df.columns:
        overtime = df["OverTime"].astype(str).str.strip().str.lower()
        df["_OverTimeFlag"] = overtime.isin(["yes", "y", "true", "1"]).astype(int)
    else:
        df["_OverTimeFlag"] = 0

    if "WorkLifeBalance" in df.columns:
        wlb = pd.to_numeric(df["WorkLifeBalance"], errors="coerce")
        wlb = wlb.clip(1, 4)
        wlb = wlb.fillna(wlb.median())
        df["_LowWorkLifeBalance"] = (wlb <= 2).astype(int)
        df["WorkLifeBalance"] = wlb
    else:
        df["_LowWorkLifeBalance"] = 0

    df["BurnoutRiskScore"] = df["_OverTimeFlag"] + df["_LowWorkLifeBalance"]
    df["BurnoutRiskLevel"] = pd.Categorical(
        np.select(
            [df["BurnoutRiskScore"] == 0, df["BurnoutRiskScore"] == 1, df["BurnoutRiskScore"] >= 2],
            ["Low", "Medium", "High"],
            default="Unknown",
        ),
        categories=["Low", "Medium", "High"],
        ordered=True,
    )

    # Workload stress indicator: overtime + travel burden scale.
    travel_map = {"non-travel": 0, "travel rarely": 1, "travel frequently": 2}
    if "BusinessTravel" in df.columns:
        # Normalize to a predictable set of values (handles underscores/spacing/casing)
        travel = (
            df["BusinessTravel"]
            .astype(str)
            .str.strip()
            .str.replace("_", " ", regex=False)
            .str.lower()
            .map(travel_map)
            .fillna(0)
        )
    else:
        travel = pd.Series(0, index=df.index)

    df["WorkloadStressScore"] = (travel + df["_OverTimeFlag"]).clip(0, 3)
    df["WorkloadStressIndicator"] = (df["WorkloadStressScore"] / 3) * 100

    return df


def validate_columns(df: pd.DataFrame) -> tuple[bool, list[str]]:
    """Check if dataset contains required columns for analysis."""
    present = [c for c in REQUIRED_COLUMNS if c in df.columns]
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    ok = len(missing) == 0
    return ok, missing


def generate_sample_data(n: int = 500, random_state: int = 42) -> pd.DataFrame:
    """Generate a realistic-looking mock dataset for demonstration."""

    rng = np.random.default_rng(random_state)

    df = pd.DataFrame(
        {
            "Age": rng.integers(22, 60, size=n),
            "Attrition": rng.choice([0, 1], size=n, p=[0.82, 0.18]),
            "BusinessTravel": rng.choice(
                ["Non-Travel", "Travel_Rarely", "Travel_Frequently"], size=n, p=[0.6, 0.3, 0.1]
            ),
            "Department": rng.choice(["R&D", "Sales", "HR"], size=n, p=[0.65, 0.25, 0.1]),
            "DistanceFromHome": rng.integers(1, 50, size=n),
            "Education": rng.integers(1, 6, size=n),
            "EducationField": rng.choice(
                ["Life Sciences", "Medical", "Marketing", "Technical", "Other"], size=n
            ),
            "EnvironmentSatisfaction": rng.integers(1, 5, size=n),
            "Gender": rng.choice(["Male", "Female"], size=n),
            "HourlyRate": rng.integers(20, 100, size=n),
            "JobInvolvement": rng.integers(1, 5, size=n),
            "JobLevel": rng.integers(1, 6, size=n),
            "JobRole": rng.choice(
                [
                    "Sales Executive",
                    "Research Scientist",
                    "Laboratory Technician",
                    "Manufacturing Director",
                    "Healthcare Representative",
                    "Manager",
                    "Sales Representative",
                ],
                size=n,
            ),
            "JobSatisfaction": rng.integers(1, 5, size=n),
            "MaritalStatus": rng.choice(["Single", "Married", "Divorced"], size=n),
            "MonthlyIncome": rng.integers(2000, 20000, size=n),
            "MonthlyRate": rng.integers(1000, 25000, size=n),
            "NumCompaniesWorked": rng.integers(0, 6, size=n),
            "OverTime": rng.choice(["Yes", "No"], size=n, p=[0.32, 0.68]),
            "PercentSalaryHike": rng.integers(10, 25, size=n),
            "PerformanceRating": rng.choice([3, 4], size=n, p=[0.25, 0.75]),
            "RelationshipSatisfaction": rng.integers(1, 5, size=n),
            "StockOptionLevel": rng.integers(0, 4, size=n),
            "TotalWorkingYears": rng.integers(0, 40, size=n),
            "TrainingTimesLastYear": rng.integers(0, 6, size=n),
            "WorkLifeBalance": rng.integers(1, 5, size=n),
            "YearsAtCompany": rng.integers(0, 25, size=n),
            "YearsInCurrentRole": rng.integers(0, 15, size=n),
            "YearsSinceLastPromotion": rng.integers(0, 10, size=n),
            "YearsWithCurrManager": rng.integers(0, 15, size=n),
        }
    )

    # Introduce a consistent bias: employees with overtime tend to have lower work-life balance
    overtime_mask = df["OverTime"] == "Yes"
    df.loc[overtime_mask, "WorkLifeBalance"] = np.clip(
        df.loc[overtime_mask, "WorkLifeBalance"], 1, 3
    )

    # Add some missing values
    for col in ["JobSatisfaction", "EnvironmentSatisfaction", "RelationshipSatisfaction"]:
        missing_mask = rng.random(n) < 0.05
        df.loc[missing_mask, col] = np.nan

    # Normalize categories for BusinessTravel to expected naming
    df["BusinessTravel"] = df["BusinessTravel"].replace(
        {"Travel_Rarely": "Travel Rarely", "Travel_Frequently": "Travel Frequently"}
    )

    return df


# -----------------------------------------------------------------------------
# Visualization helpers
# -----------------------------------------------------------------------------


def bar_chart(df: pd.DataFrame, x: str, y: str, title: str, sort_desc: bool = True) -> alt.Chart:
    return (
        alt.Chart(df)
        .mark_bar()
        .encode(
            x=alt.X(f"{x}:N", sort="-y" if sort_desc else "x"),
            y=alt.Y(f"{y}:Q", title=y),
            tooltip=[x, y],
        )
        .properties(title=title)
        .interactive()
    )


def line_chart(df: pd.DataFrame, x: str, y: str, color: Optional[str] = None, title: Optional[str] = None):
    enc = {"x": alt.X(f"{x}:Q"), "y": alt.Y(f"{y}:Q", title=y)}
    if color:
        enc["color"] = alt.Color(f"{color}:N", legend=alt.Legend(title=color))
    return alt.Chart(df).mark_line(point=True).encode(**enc).properties(title=title).interactive()


# -----------------------------------------------------------------------------
# Dashboard
# -----------------------------------------------------------------------------


def main() -> None:
    st.set_page_config(
        page_title="Employee Engagement & Burnout Risk",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.title("🚦 Employee Engagement & Burnout Risk Dashboard")
    st.markdown(
        "This dashboard provides a preventive view into engagement, burnout risk, and career-stage dynamics."
    )

    # ----------- Sidebar inputs --------------
    st.sidebar.header("Data & Filters")
    upload = st.sidebar.file_uploader(
        "Upload employee dataset (CSV)", type=["csv"], accept_multiple_files=False
    )

    use_sample = st.sidebar.checkbox("Use synthetic example dataset", value=False)

    refresh_interval = st.sidebar.number_input(
        "Auto-refresh interval (seconds, 0=off)",
        min_value=0,
        max_value=3600,
        value=0,
        step=1,
        help="Reload the dashboard every N seconds (useful for live data feeds).",
    )

    if refresh_interval > 0:
        st.sidebar.markdown(
            f"<script>setTimeout(()=>window.location.reload(), {refresh_interval * 1000});</script>",
            unsafe_allow_html=True,
        )

    # Enable manual refresh when using a local CSV dataset
    if "refresh_counter" not in st.session_state:
        st.session_state.refresh_counter = 0

    if st.sidebar.button("Refresh data"):
        st.session_state.refresh_counter += 1
        st.experimental_rerun()

    df = None
    dataset_source = None
    last_modified_str = None

    if upload is not None:
        df = pd.read_csv(upload)
        dataset_source = "uploaded file"
        st.sidebar.success("Dataset loaded from upload")
    elif DEFAULT_DATA_PATH.exists():
        df = load_dataset(DEFAULT_DATA_PATH, st.session_state.refresh_counter)
        dataset_source = f"local file: {DEFAULT_DATA_FILE}"
        last_modified_str = datetime.fromtimestamp(DEFAULT_DATA_PATH.stat().st_mtime).strftime(
            "%Y-%m-%d %H:%M:%S"
        )
        st.sidebar.success(f"Dataset loaded from {dataset_source}")
    elif use_sample:
        df = generate_sample_data(900)
        dataset_source = "synthetic sample"
        st.sidebar.info("Using generated synthetic dataset")

    if df is None:
        st.info(
            "Upload a CSV dataset, place a file named\n" \
            f"'{DEFAULT_DATA_FILE}' in this folder, or enable the synthetic example dataset checkbox in the sidebar."
        )
        return

    if last_modified_str is not None:
        st.sidebar.markdown(f"**Last modified:** {last_modified_str}")

    # Validate required columns for core calculations
    ok, missing = validate_columns(df)
    if not ok:
        st.warning(
            "Dataset is missing some of the expected columns; analysis will still proceed with available fields."
        )
        st.write("Missing columns:", missing)

    df = compute_engagement_index(df)
    df = compute_burnout_risk(df)

    # Filters
    st.sidebar.markdown("---")
    st.sidebar.subheader("Filters")

    dept_choices = sorted(df["Department"].dropna().unique().astype(str)) if "Department" in df else []
    selected_depts = st.sidebar.multiselect("Department", options=dept_choices, default=dept_choices)

    role_choices = sorted(df["JobRole"].dropna().unique().astype(str)) if "JobRole" in df else []
    selected_roles = st.sidebar.multiselect("Job Role", options=role_choices, default=role_choices)

    overtime_choices = ["All"] + sorted(df["OverTime"].dropna().unique().astype(str)) if "OverTime" in df else ["All"]
    selected_overtime = st.sidebar.selectbox("Overtime", options=overtime_choices, index=0)

    min_eng, max_eng = float(df["EngagementIndex"].min()), float(df["EngagementIndex"].max())
    engagement_threshold = st.sidebar.slider(
        "Engagement index (min)", min_value=0.0, max_value=100.0, value=min_eng, step=1.0
    )

    year_min = int(df["YearsAtCompany"].min()) if "YearsAtCompany" in df else 0
    year_max = int(df["YearsAtCompany"].max()) if "YearsAtCompany" in df else 20
    years_range = st.sidebar.slider(
        "Years at company", min_value=year_min, max_value=year_max, value=(year_min, year_max)
    )

    # Apply filters
    df_filtered = df.copy()
    if selected_depts and "Department" in df:
        df_filtered = df_filtered[df_filtered["Department"].isin(selected_depts)]
    if selected_roles and "JobRole" in df:
        df_filtered = df_filtered[df_filtered["JobRole"].isin(selected_roles)]
    if selected_overtime != "All" and "OverTime" in df:
        df_filtered = df_filtered[df_filtered["OverTime"] == selected_overtime]
    df_filtered = df_filtered[df_filtered["EngagementIndex"] >= engagement_threshold]
    if "YearsAtCompany" in df:
        df_filtered = df_filtered[ df_filtered["YearsAtCompany"].between(years_range[0], years_range[1]) ]

    st.sidebar.markdown("---")
    st.sidebar.markdown(
        "**Tip:** Filters apply to all dashboard widgets. Use the engagement slider to focus on at-risk groups."
    )

    # ---------- KPI overview ----------
    st.subheader("Engagement Health Overview")
    kpi_cols = st.columns(4)

    avg_engagement = df_filtered["EngagementIndex"].mean() if not df_filtered.empty else np.nan
    avg_wlb = df_filtered["WorkLifeBalance"].mean() if "WorkLifeBalance" in df_filtered else np.nan
    burn_risk = df_filtered["BurnoutRiskLevel"].value_counts(normalize=True).mul(100).round(1)
    high_risk_pct = burn_risk.get("High", 0.0)

    kpi_cols[0].metric("Average engagement", f"{avg_engagement:.1f}/100")
    kpi_cols[1].metric("Average work-life balance", f"{avg_wlb:.1f}/4")
    kpi_cols[2].metric("High burnout risk", f"{high_risk_pct:.1f}%")
    kpi_cols[3].metric("Sample size", f"{len(df_filtered):,}")

    with st.expander("Show data sample"):
        st.dataframe(df_filtered.head(80))

    # Executive summary & export
    st.markdown("### Executive summary & export")

    filters = {
        "Departments": selected_depts,
        "Roles": selected_roles,
        "Overtime": selected_overtime,
        "Engagement min": engagement_threshold,
        "Tenure range": years_range,
        "Auto-refresh (s)": refresh_interval,
    }

    summary_md = build_executive_summary(
        df_filtered,
        dataset_source or "unknown",
        last_modified_str,
        filters,
    )

    st.download_button(
        "Download Markdown report",
        summary_md,
        file_name="employee_engagement_summary.md",
        mime="text/markdown",
    )

    # Engagement distribution
    st.markdown("### Engagement distribution")
    eng_hist = (
        alt.Chart(df_filtered)
        .mark_bar()
        .encode(
            alt.X("EngagementIndex:Q", bin=alt.Bin(maxbins=30), title="Engagement index"),
            y=alt.Y("count():Q", title="Employees"),
            tooltip=[alt.Tooltip("count():Q", title="Employees")],
        )
        .properties(width=700)
    )
    st.altair_chart(eng_hist, width="stretch")

    # Burnout risk distribution
    st.markdown("### Burnout risk levels")
    burn_dist = df_filtered["BurnoutRiskLevel"].value_counts().rename_axis("level").reset_index(name="count")
    burn_chart = bar_chart(burn_dist, x="level", y="count", title="Burnout risk distribution")
    st.altair_chart(burn_chart, width="stretch")

    # Workload & stress analysis
    st.markdown("### Workload & stress analysis")
    if "OverTime" in df_filtered:
        ot_eng = (
            df_filtered.groupby("OverTime")
            .agg(AvgEngagement=("EngagementIndex", "mean"), Count=("EngagementIndex", "count"))
            .reset_index()
        )
        st.altair_chart(
            bar_chart(ot_eng, x="OverTime", y="AvgEngagement", title="Engagement by overtime"),
            width="stretch",
        )

    if "BusinessTravel" in df_filtered:
        travel_eng = (
            df_filtered.groupby("BusinessTravel")
            .agg(AvgEngagement=("EngagementIndex", "mean"), Count=("EngagementIndex", "count"))
            .reset_index()
        )
        st.altair_chart(
            bar_chart(travel_eng, x="BusinessTravel", y="AvgEngagement", title="Engagement by travel frequency"),
            width="stretch",
        )

    # Career-stage analysis
    st.markdown("### Role & career stage analysis")
    if "JobLevel" in df_filtered:
        level_eng = (
            df_filtered.groupby("JobLevel")
            .agg(AvgEngagement=("EngagementIndex", "mean"), Count=("EngagementIndex", "count"))
            .reset_index()
        )
        st.altair_chart(
            bar_chart(level_eng, x="JobLevel", y="AvgEngagement", title="Engagement by job level"),
            width="stretch",
        )

    if "YearsAtCompany" in df_filtered and "EngagementIndex" in df_filtered:
        tenure_eng = (
            df_filtered.groupby(pd.cut(df_filtered["YearsAtCompany"], bins=[-0.1, 1, 3, 5, 10, 20, 40]))
            ["EngagementIndex"]
            .mean()
            .reset_index()
        )
        tenure_eng = tenure_eng.rename(columns={"YearsAtCompany": "TenureRange"})
        st.altair_chart(
            bar_chart(tenure_eng, x="TenureRange", y="EngagementIndex", title="Engagement by tenure")
                .properties(width=700),
            width="stretch",
        )

    # Engagement vs attrition
    st.markdown("### Engagement by attrition status")
    if "Attrition" in df_filtered:
        attr_map = {0: "Stayed", 1: "Left"}
        df_filtered["AttritionLabel"] = df_filtered["Attrition"].map(attr_map).fillna("Unknown")
        box = (
            alt.Chart(df_filtered)
            .mark_boxplot()
            .encode(
                x=alt.X("AttritionLabel:N", title="Attrition"),
                y=alt.Y("EngagementIndex:Q", title="Engagement index"),
                color=alt.Color("AttritionLabel:N", legend=None),
            )
        )
        st.altair_chart(box, width="stretch")

    st.markdown(
        "---\n" "**Note:** This dashboard is a diagnostic tool; it does not predict attrition, but highlights signals that correlate with disengagement and burnout."
    )


if __name__ == "__main__":
    main()
