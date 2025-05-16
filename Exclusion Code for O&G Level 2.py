import streamlit as st
import pandas as pd
import io
from io import BytesIO
import re
import numpy as np

#########################
# 1) HELPER FUNCTIONS
#########################

def flatten_multilevel_columns(df):
    """Flatten multi‐level column headers into single strings."""
    df.columns = [
        " ".join(str(level) for level in col).strip()
        for col in df.columns
    ]
    return df

def find_column(df, possible_matches, required=True):
    """Find the first column matching any item in possible_matches,
    preferring an exact match on the whole header before falling back to substring."""
    norm_map = {
        col: col.strip().lower().replace("\n", " ")
        for col in df.columns
    }
    # 1) exact
    for pattern in possible_matches:
        pat = pattern.strip().lower().replace("\n", " ")
        for col, col_norm in norm_map.items():
            if col_norm == pat:
                return col
    # 2) substring
    for col, col_norm in norm_map.items():
        for pattern in possible_matches:
            pat = pattern.strip().lower().replace("\n", " ")
            if pat in col_norm:
                return col
    if required:
        raise ValueError(
            f"Could not find a required column among {possible_matches}\n"
            f"Available columns: {list(df.columns)}"
        )
    return None

#########################
# 2) UPSTREAM EXCLUSION
#########################

def filter_upstream_companies(df):
    """Parses 'Upstream' sheet and applies custom exclusion logic."""
    df = flatten_multilevel_columns(df)

    # locate the three key columns
    resources_col = find_column(df,
        ["resources under development and field evaluation"],
        required=True)
    capex_col = find_column(df,
        ["exploration capex 3-year average", "exploration capex 3 year average"],
        required=True)
    short_term_col = find_column(df,
        ["short-term expansion ≥20 mmboe", "short term expansion"],
        required=True)

    # ensure they exist
    for col in (resources_col, capex_col, short_term_col):
        if col not in df.columns:
            df[col] = np.nan

    # convert numeric columns
    for c in (resources_col, capex_col):
        df[c] = (
            df[c].astype(str)
                .str.replace(",", "", regex=True)
        )
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # build exclusion flags
    df["Resources_Exclusion_Flag"] = df[resources_col].isna() | (df[resources_col] > 0)
    df["CAPEX_Exclusion_Flag"]    = df[capex_col].isna()    | (df[capex_col]    > 0)
    df["ShortTerm_Exclusion_Flag"] = (
        df[short_term_col]
          .astype(str)
          .str.strip()
          .str.lower()
          .eq("yes")
    )
    df["Excluded"] = (
        df[["Resources_Exclusion_Flag",
            "CAPEX_Exclusion_Flag",
            "ShortTerm_Exclusion_Flag"]]
        .any(axis=1)
    )

    # reasons
    def make_reason(row):
        reasons = []
        if row["Resources_Exclusion_Flag"]:
            reasons.append("Missing or >0 Resources under development")
        if row["CAPEX_Exclusion_Flag"]:
            reasons.append("Missing or >0 Exploration CAPEX 3-year avg")
        if row["ShortTerm_Exclusion_Flag"]:
            reasons.append("Short-Term Expansion ≥20 mmboe = Yes")
        return "; ".join(reasons)

    df["Exclusion Reason"] = df.apply(make_reason, axis=1)

    # split out
    excluded = df[df["Excluded"]].copy()
    retained = df[~df["Excluded"]].copy()

    # pick columns to show
    company_col = find_column(df, ["company"], required=False) or df.columns[0]
    out_cols = [
        company_col,
        resources_col,
        capex_col,
        short_term_col,
        "Exclusion Reason",
    ]

    return excluded[out_cols], retained[out_cols]

#########################
# 3) ALL-COMPANIES EXCLUSION (unchanged)
#########################

def rename_columns(df):
    df = flatten_multilevel_columns(df)
    df = df.loc[:, ~df.columns.str.lower().str.startswith("parent company")]
    df = df.iloc[1:].reset_index(drop=True)
    rename_map = {
        "Company": ["company"],
        "GOGEL Tab": ["gogel tab"],
        "BB Ticker": ["bb ticker", "bloomberg ticker"],
        "ISIN Equity": ["isin equity", "isin code"],
        "LEI": ["lei"],
        "Length of Pipelines under Development": ["length of pipelines", "pipeline under dev"],
        "Liquefaction Capacity (Export)": ["liquefaction capacity (export)", "lng export capacity"],
        "Regasification Capacity (Import)": ["regasification capacity (import)", "lng import capacity"],
        "Total Capacity under Development": ["total capacity under development", "total dev capacity"]
    }
    for new_col, patterns in rename_map.items():
        old = find_column(df, patterns, required=False)
        if old and old != new_col:
            df.rename(columns={old: new_col}, inplace=True)
    df = df.loc[:, ~df.columns.duplicated(keep='last')]
    return df

def filter_all_companies(df):
    df = rename_columns(df)
    required_columns = [
        "Company", "GOGEL Tab", "BB Ticker", "ISIN Equity", "LEI",
        "Length of Pipelines under Development",
        "Liquefaction Capacity (Export)",
        "Regasification Capacity (Import)",
        "Total Capacity under Development"
    ]
    for col in required_columns:
        if col not in df.columns:
            df[col] = None if col in ["Company", "GOGEL Tab", "BB Ticker", "ISIN Equity", "LEI"] else 0
    numeric_cols = [
        "Length of Pipelines under Development",
        "Liquefaction Capacity (Export)",
        "Regasification Capacity (Import)",
        "Total Capacity under Development"
    ]
    for c in numeric_cols:
        df[c] = (
            df[c].astype(str)
               .str.replace("%", "", regex=True)
               .str.replace(",", "", regex=True)
        )
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

    df["Upstream_Exclusion_Flag"] = df["GOGEL Tab"].str.contains("upstream", case=False, na=False)
    df["Midstream_Exclusion_Flag"] = (
        (df["Length of Pipelines under Development"] > 0)
        | (df["Liquefaction Capacity (Export)"] > 0)
        | (df["Regasification Capacity (Import)"] > 0)
        | (df["Total Capacity under Development"] > 0)
    )
    df["Excluded"] = df["Upstream_Exclusion_Flag"] | df["Midstream_Exclusion_Flag"]

    def get_exclusion_reason(row):
        reasons = []
        if row["Upstream_Exclusion_Flag"]:
            reasons.append("Upstream in GOGEL Tab")
        if row["Midstream_Exclusion_Flag"]:
            reasons.append("Midstream Expansion > 0")
        return "; ".join(reasons)

    df["Exclusion Reason"] = df.apply(get_exclusion_reason, axis=1)

    excluded_df = df[df["Excluded"]].copy()
    retained_df = df[~df["Excluded"]].copy()

    final_cols = [
        "Company", "BB Ticker", "ISIN Equity", "LEI",
        "GOGEL Tab",
        "Length of Pipelines under Development",
        "Liquefaction Capacity (Export)",
        "Regasification Capacity (Import)",
        "Total Capacity under Development",
        "Exclusion Reason"
    ]
    for c in final_cols:
        for d in (excluded_df, retained_df):
            if c not in d.columns:
                d[c] = None

    return excluded_df[final_cols], retained_df[final_cols]

#########################
# 4) STREAMLIT APP
#########################

def main():
    st.title("Level 2 Exclusion Filter for O&G")

    uploaded_file = st.file_uploader("Upload Excel file", type=["xlsx"])
    if not uploaded_file:
        return

    xls = pd.ExcelFile(uploaded_file)

    # All Companies
    if "All Companies" in xls.sheet_names:
        df_all = pd.read_excel(uploaded_file, sheet_name="All Companies", header=[3,4])
        excluded_all, retained_all = filter_all_companies(df_all)

        st.subheader("All Companies – Summary")
        total = len(excluded_all) + len(retained_all)
        st.write(f"**Total Companies Processed:** {total}")
        st.write(f"**Excluded (All):** {len(excluded_all)}")
        st.write(f"**Retained (All):** {len(retained_all)}")

        st.subheader("All Companies – Excluded")
        st.dataframe(excluded_all)
        st.subheader("All Companies – Retained")
        st.dataframe(retained_all)
    else:
        st.error("No sheet named 'All Companies'.")

    # Upstream
    if "Upstream" in xls.sheet_names:
        df_up = pd.read_excel(uploaded_file, sheet_name="Upstream", header=[3,4])
        excluded_up, retained_up = filter_upstream_companies(df_up)

        st.subheader("Upstream – Excluded")
        st.dataframe(excluded_up)
        st.subheader("Upstream – Retained")
        st.dataframe(retained_up)
    else:
        st.info("No sheet named 'Upstream' to process.")

    # Download all results in one file
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        if "All Companies" in xls.sheet_names:
            excluded_all.to_excel(writer, sheet_name="All Excluded", index=False)
            retained_all.to_excel(writer, sheet_name="All Retained", index=False)
        if "Upstream" in xls.sheet_names:
            excluded_up.to_excel(writer, sheet_name="Upstream Excluded", index=False)
            retained_up.to_excel(writer, sheet_name="Upstream Retained", index=False)
    output.seek(0)

    st.download_button(
        "Download Processed File",
        output,
        "O&G_Exclusion_Results.xlsx",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

if __name__ == "__main__":
    main()
