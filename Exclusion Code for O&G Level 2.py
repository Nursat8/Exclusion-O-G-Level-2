import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO
import re

#########################
# 1) HELPER FUNCTIONS
#########################

def flatten_multilevel_columns(df):
    df.columns = [
        " ".join(str(level).strip() for level in col).strip()
        for col in df.columns
    ]
    return df

def find_column(df, possible_matches, required=True):
    norm_map = {}
    for col in df.columns:
        norm = col.strip().lower().replace("\n", " ")
        norm = re.sub(r"\s+", " ", norm)
        norm_map[col] = norm

    pats = []
    for pattern in possible_matches:
        p = pattern.strip().lower().replace("\n", " ")
        p = re.sub(r"\s+", " ", p)
        pats.append(p)

    # 1) exact
    for pat in pats:
        for col, col_norm in norm_map.items():
            if col_norm == pat:
                return col
    # 2) substring
    for col, col_norm in norm_map.items():
        for pat in pats:
            if pat in col_norm:
                return col

    if required:
        raise ValueError(f"Could not find a required column among {possible_matches}\nAvailable columns: {list(df.columns)}")
    return None

#########################
# 2) UPSTREAM EXCLUSION
#########################

def filter_upstream_companies(df):
    df = flatten_multilevel_columns(df)

    # locate the three key columns
    resources_col = find_column(df,
        ["Resources under Development and Field Evaluation"],
        required=True)
    capex_col = find_column(df,
        ["Exploration CAPEX 3-year average", "Exploration CAPEX 3 year average"],
        required=True)
    short_term_col = find_column(df,
        ["Short-Term Expansion ≥20 mmboe", "Short Term Expansion"],
        required=True)

    # coerce numeric columns
    for c in (resources_col, capex_col):
        df[c] = (
            df[c].astype(str)
                 .str.replace(",", "", regex=True)
        )
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # build exclusion flags
    df["Resources_Exclusion_Flag"] = df[resources_col].isna() | (df[resources_col] > 0)
    df["CAPEX_Exclusion_Flag"]    = df[capex_col].isna()    | (df[capex_col]    > 0)
    df["ShortTerm_Exclusion_Flag"]= (
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

    # assign the exact reason strings you requested
    def make_reason(row):
        reasons = []
        if row["Resources_Exclusion_Flag"]:
            reasons.append("Resources under development empty or >0")
        if row["CAPEX_Exclusion_Flag"]:
            reasons.append("Exploration CAPEX 3-year average empty or >0")
        if row["ShortTerm_Exclusion_Flag"]:
            reasons.append("Short-Term Expansion ≥20 mmboe = Yes")
        return "; ".join(reasons)

    df["Exclusion Reason"] = df.apply(make_reason, axis=1)

    # split into excluded vs retained
    excluded = df[df["Excluded"]].copy()
    retained = df[~df["Excluded"]].copy()

    # choose columns to output
    company_col = find_column(df, ["Company"], required=False) or df.columns[0]
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
    return df.loc[:, ~df.columns.duplicated(keep='last')]

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
            df[col] = None if col in ["Company","GOGEL Tab","BB Ticker","ISIN Equity","LEI"] else 0

    # convert to numeric
    for c in [
        "Length of Pipelines under Development",
        "Liquefaction Capacity (Export)",
        "Regasification Capacity (Import)",
        "Total Capacity under Development"
    ]:
        df[c] = (
            df[c].astype(str)
               .str.replace("%","",regex=True)
               .str.replace(",","",regex=True)
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

    def get_reason(row):
        reasons = []
        if row["Upstream_Exclusion_Flag"]:
            reasons.append("Upstream in GOGEL Tab")
        if row["Midstream_Exclusion_Flag"]:
            reasons.append("Midstream Expansion > 0")
        return "; ".join(reasons)

    df["Exclusion Reason"] = df.apply(get_reason, axis=1)
    excluded = df[df["Excluded"]].copy()
    retained = df[~df["Excluded"]].copy()

    final_cols = [
        "Company","BB Ticker","ISIN Equity","LEI","GOGEL Tab",
        "Length of Pipelines under Development",
        "Liquefaction Capacity (Export)",
        "Regasification Capacity (Import)",
        "Total Capacity under Development",
        "Exclusion Reason"
    ]
    for c in final_cols:
        for d in (excluded, retained):
            if c not in d.columns:
                d[c] = None

    return excluded[final_cols], retained[final_cols]

#########################
# 4) STREAMLIT UI
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
        exc_all, ret_all = filter_all_companies(df_all)
        st.subheader("All Companies – Summary")
        st.write(f"Total: {len(exc_all)+len(ret_all)} | Excluded: {len(exc_all)} | Retained: {len(ret_all)}")
        st.subheader("Excluded (All Companies)");   st.dataframe(exc_all)
        st.subheader("Retained (All Companies)");  st.dataframe(ret_all)
    else:
        st.error("No sheet named 'All Companies'.")

    # Upstream
    if "Upstream" in xls.sheet_names:
        df_up = pd.read_excel(uploaded_file, sheet_name="Upstream", header=[3,4])
        exc_up, ret_up = filter_upstream_companies(df_up)
        st.subheader("Upstream – Excluded");   st.dataframe(exc_up)
        st.subheader("Upstream – Retained");  st.dataframe(ret_up)
    else:
        st.info("No sheet named 'Upstream' to process.")

    # Download combined results
    out = BytesIO()
    with pd.ExcelWriter(out, engine="xlsxwriter") as writer:
        if "All Companies" in xls.sheet_names:
            exc_all.to_excel(writer, sheet_name="All Excluded",   index=False)
            ret_all.to_excel(writer, sheet_name="All Retained",  index=False)
        if "Upstream" in xls.sheet_names:
            exc_up.to_excel(writer, sheet_name="Upstream Excluded", index=False)
            ret_up.to_excel(writer, sheet_name="Upstream Retained",index=False)
    out.seek(0)

    st.download_button(
        "Download Results",
        out,
        "O&G_Exclusion_Results.xlsx",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

if __name__ == "__main__":
    main()
