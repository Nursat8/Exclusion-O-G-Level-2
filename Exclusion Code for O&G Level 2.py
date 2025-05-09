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
    """Flatten multi-level column headers into single strings."""
    df.columns = [
        " ".join(str(level) for level in col).strip()
        for col in df.columns
    ]
    return df

def find_column(df, possible_matches, required=True):
    """Find the first column matching any item in possible_matches,
    preferring an exact match on the whole header before falling back to substring."""
    # build a map of normalized → actual
    norm_map = {
        col: col.strip().lower().replace("\n", " ")
        for col in df.columns
    }

    # 1) exact-match pass
    for pattern in possible_matches:
        pat = pattern.strip().lower().replace("\n", " ")
        for col, col_norm in norm_map.items():
            if col_norm == pat:
                return col

    # 2) substring pass (your old logic)
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


def rename_columns(df):
    """
    Flatten multi-level headers, drop the Parent Company column,
    then shift and rename.
    """
    # 1) flatten multi‐level into strings
    df = flatten_multilevel_columns(df)

    # 2) drop the Parent Company column so find_column("company") can only ever hit the real one
    df = df.loc[:, ~df.columns.str.lower().str.startswith("parent company")]

    # 3) shift up one row (Excel row 7 → pandas row 0)
    df = df.iloc[1:].reset_index(drop=True)

    rename_map = {
        "Company": ["company"],  
        "GOGEL Tab": ["GOGEL Tab"],  
        "BB Ticker": ["bb ticker", "bloomberg ticker"],
        "ISIN Equity": ["isin equity", "isin code"],
        "LEI": ["lei"],
        "Length of Pipelines under Development": ["length of pipelines", "pipeline under dev"],
        "Liquefaction Capacity (Export)": ["liquefaction capacity (export)", "lng export capacity", "Liquefaction Capacity Export"],
        "Regasification Capacity (Import)": ["regasification capacity (import)", "lng import capacity", "Regasification Capacity Import"],
        "Total Capacity under Development": ["total capacity under development", "total dev capacity"]
    }

    for new_col, patterns in rename_map.items():
        old_col = find_column(df, patterns, required=False)
        if old_col and old_col != new_col:
            df.rename(columns={old_col: new_col}, inplace=True)
    df = df.loc[:, ~df.columns.duplicated(keep='last')]
    return df

#########################
# 2) CORE EXCLUSION LOGIC
#########################

def filter_all_companies(df):
    """Parses 'All Companies' sheet, applies exclusion logic, and splits into categories."""
    
    # 1) Flatten headers, rename columns
    df = rename_columns(df)

    # 2) Ensure key columns exist
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

    # 3) Convert numeric columns
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

    # 4) Apply Exclusion Logic
    df["Upstream_Exclusion_Flag"] = df["GOGEL Tab"].str.contains("upstream", case=False, na=False)
    df["Midstream_Exclusion_Flag"] = (
        (df["Length of Pipelines under Development"] > 0)
        | (df["Liquefaction Capacity (Export)"] > 0)
        | (df["Regasification Capacity (Import)"] > 0)
        | (df["Total Capacity under Development"] > 0)
    )
    df["Excluded"] = df["Upstream_Exclusion_Flag"] | df["Midstream_Exclusion_Flag"]

    # 5) Build Exclusion Reason
    def get_exclusion_reason(row):
        reasons = []
        if row["Upstream_Exclusion_Flag"]:
            reasons.append("Upstream in GOGEL Tab")
        if row["Midstream_Exclusion_Flag"]:
            reasons.append("Midstream Expansion > 0")
        return "; ".join(reasons)
    
    df["Exclusion Reason"] = df.apply(get_exclusion_reason, axis=1)

    # 6) Move all "No Data" companies into Retained
    retained_df = df[~df["Excluded"]].copy()
    excluded_df = df[df["Excluded"]].copy()

    # 7) Keep only required columns, including Midstream Expansion data
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
        for d in [excluded_df, retained_df]:
            if c not in d.columns:
                d[c] = None

    return excluded_df[final_cols], retained_df[final_cols]

#########################
# 3) STREAMLIT APP
#########################

def main():
    st.title("Level 2 Exclusion Filter for O&G")

    uploaded_file = st.file_uploader("Upload Excel file", type=["xlsx"])

    if uploaded_file:
        xls = pd.ExcelFile(uploaded_file)

        if "All Companies" not in xls.sheet_names:
            st.error("No sheet named 'All Companies'.")
            return

        # Read with multi-level headers from rows 3 & 4 (0-based)
        df_all = pd.read_excel(
            uploaded_file,
            sheet_name="All Companies",
            header=[3,4]
        )

        excluded, retained = filter_all_companies(df_all)

        # Remove "Equity" string in the BB Ticker column for output file only
        def remove_equity_from_bb_ticker(df):
            df = df.copy()
            if "BB Ticker" in df.columns:
                df["BB Ticker"] = (
                    df["BB Ticker"]
                    .astype(str)
                    .str.replace(r"\u00A0", " ", regex=True)
                    .str.replace(r"(?i)\bEquity\b", "", regex=True)
                    .str.replace(r"\s+", " ", regex=True)
                    .str.strip()
                )
            return df

        excluded = remove_equity_from_bb_ticker(excluded)
        retained = remove_equity_from_bb_ticker(retained)

        # STATS
        total_companies = len(excluded) + len(retained)
        st.subheader("Summary Statistics")
        st.write(f"**Total Companies Processed:** {total_companies}")
        st.write(f"**Excluded Companies (Upstream & Midstream):** {len(excluded)}")
        st.write(f"**Retained Companies:** {len(retained)}")

        # Display DataFrames
        st.subheader("Excluded Companies")
        st.dataframe(excluded)

        st.subheader("Retained Companies")
        st.dataframe(retained)

        # Save to Excel
        output = BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            excluded.to_excel(writer, sheet_name="Excluded", index=False)
            retained.to_excel(writer, sheet_name="Retained", index=False)
        output.seek(0)

        st.download_button(
            "Download Processed File",
            output,
            "O&G Companies Level 2 Exclusion.xlsx",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

if __name__ == "__main__":
    main()
