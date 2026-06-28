import io
import pandas as pd
import numpy as np
import streamlit as st

st.set_page_config(page_title="SAS vs PySpark Output Compare", layout="wide")

# ── helpers ──────────────────────────────────────────────────────────────────

def load_file(uploaded) -> pd.DataFrame | None:
    if uploaded is None:
        return None
    name = uploaded.name.lower()
    if name.endswith(".csv"):
        return pd.read_csv(uploaded)
    if name.endswith((".xls", ".xlsx")):
        return pd.read_excel(uploaded)
    if name.endswith(".parquet"):
        return pd.read_parquet(io.BytesIO(uploaded.read()))
    st.error(f"Unsupported file type: {uploaded.name}")
    return None


def normalise_cols(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [c.strip().lower() for c in df.columns]
    return df


def badge(ok: bool, ok_text="PASS", fail_text="FAIL") -> str:
    colour = "green" if ok else "red"
    text = ok_text if ok else fail_text
    return f":{colour}[**{text}**]"


# ── section renderers ────────────────────────────────────────────────────────

def section_row_counts(sas: pd.DataFrame, spark: pd.DataFrame):
    st.subheader("Row Count Check")
    s, p = len(sas), len(spark)
    diff = s - p
    cols = st.columns(3)
    cols[0].metric("SAS rows", f"{s:,}")
    cols[1].metric("PySpark rows", f"{p:,}")
    cols[2].metric("Difference", f"{diff:,}", delta_color="off")
    st.write(badge(diff == 0), "Row counts match." if diff == 0
             else f"Row counts differ by **{abs(diff):,}**.")


def section_column_check(sas: pd.DataFrame, spark: pd.DataFrame):
    st.subheader("Column Check")
    sas_cols = set(sas.columns)
    spark_cols = set(spark.columns)
    common = sas_cols & spark_cols
    only_sas = sas_cols - spark_cols
    only_spark = spark_cols - sas_cols

    c1, c2, c3 = st.columns(3)
    c1.metric("SAS columns", len(sas_cols))
    c2.metric("PySpark columns", len(spark_cols))
    c3.metric("Common columns", len(common))

    if only_sas:
        st.warning(f"Columns **only in SAS** ({len(only_sas)}): `{sorted(only_sas)}`")
    if only_spark:
        st.warning(f"Columns **only in PySpark** ({len(only_spark)}): `{sorted(only_spark)}`")
    if not only_sas and not only_spark:
        st.write(badge(True), "All columns match.")

    # Data-type comparison for common columns
    if common:
        dtype_rows = []
        for col in sorted(common):
            st_dtype = str(sas[col].dtype)
            sp_dtype = str(spark[col].dtype)
            match = st_dtype == sp_dtype
            dtype_rows.append({"Column": col, "SAS dtype": st_dtype,
                                "PySpark dtype": sp_dtype, "Match": match})
        df_dtypes = pd.DataFrame(dtype_rows)
        mismatches = df_dtypes[~df_dtypes["Match"]]
        st.write(f"**Data-type comparison** ({len(mismatches)} mismatches):")
        st.dataframe(df_dtypes.style.apply(
            lambda row: ["background-color: #ffe6e6" if not row["Match"] else ""
                         for _ in row], axis=1),
            use_container_width=True)


def section_null_check(sas: pd.DataFrame, spark: pd.DataFrame):
    st.subheader("Null / Missing Value Check")
    common = sorted(set(sas.columns) & set(spark.columns))
    rows = []
    for col in common:
        sn = int(sas[col].isna().sum())
        pn = int(spark[col].isna().sum())
        rows.append({
            "Column": col,
            "SAS nulls": sn,
            "SAS null %": f"{sn / max(len(sas), 1) * 100:.2f}%",
            "PySpark nulls": pn,
            "PySpark null %": f"{pn / max(len(spark), 1) * 100:.2f}%",
            "Delta": sn - pn,
        })
    df_null = pd.DataFrame(rows)
    has_diff = (df_null["Delta"] != 0).any()
    st.write(badge(not has_diff, "No differences", "Differences found"))
    st.dataframe(df_null.style.apply(
        lambda row: ["background-color: #ffe6e6" if row["Delta"] != 0 else ""
                     for _ in row], axis=1),
        use_container_width=True)


def section_summary_stats(sas: pd.DataFrame, spark: pd.DataFrame):
    st.subheader("Summary Statistics (Numeric Columns)")
    common = sorted(set(sas.columns) & set(spark.columns))
    num_cols = [c for c in common
                if pd.api.types.is_numeric_dtype(sas[c])
                and pd.api.types.is_numeric_dtype(spark[c])]
    if not num_cols:
        st.info("No common numeric columns to compare.")
        return

    tol = st.number_input("Tolerance for numeric comparison (absolute)",
                          min_value=0.0, value=0.0, step=0.000001,
                          format="%.6f", key="stat_tol")

    stats = ["mean", "std", "min", "25%", "50%", "75%", "max"]
    sas_desc = sas[num_cols].describe().loc[stats]
    spark_desc = spark[num_cols].describe().loc[stats]

    rows = []
    for col in num_cols:
        for stat in stats:
            sv = sas_desc.at[stat, col]
            pv = spark_desc.at[stat, col]
            diff = abs(sv - pv) if pd.notna(sv) and pd.notna(pv) else np.nan
            rows.append({
                "Column": col, "Stat": stat,
                "SAS": round(float(sv), 6) if pd.notna(sv) else None,
                "PySpark": round(float(pv), 6) if pd.notna(pv) else None,
                "Abs diff": round(float(diff), 6) if pd.notna(diff) else None,
                "Within tol": diff <= tol if pd.notna(diff) else False,
            })
    df_stats = pd.DataFrame(rows)
    st.dataframe(df_stats.style.apply(
        lambda row: ["background-color: #ffe6e6" if not row["Within tol"] else ""
                     for _ in row], axis=1),
        use_container_width=True)


def section_duplicates(sas: pd.DataFrame, spark: pd.DataFrame):
    st.subheader("Duplicate Row Check")
    sd = int(sas.duplicated().sum())
    pd_ = int(spark.duplicated().sum())
    c1, c2 = st.columns(2)
    c1.metric("Duplicate rows in SAS", sd)
    c2.metric("Duplicate rows in PySpark", pd_)
    if sd:
        with st.expander("Show SAS duplicate rows"):
            st.dataframe(sas[sas.duplicated(keep=False)], use_container_width=True)
    if pd_:
        with st.expander("Show PySpark duplicate rows"):
            st.dataframe(spark[spark.duplicated(keep=False)], use_container_width=True)


def section_value_diff(sas: pd.DataFrame, spark: pd.DataFrame):
    st.subheader("Row-Level Value Difference Check")
    common = sorted(set(sas.columns) & set(spark.columns))

    key_cols = st.multiselect(
        "Select key column(s) to join on (must uniquely identify rows)",
        options=common, key="key_cols")

    if not key_cols:
        st.info("Select at least one key column above to run the row-level diff.")
        return

    tol = st.number_input("Numeric tolerance (absolute, 0 = exact)",
                          min_value=0.0, value=0.0, step=0.000001,
                          format="%.6f", key="diff_tol")

    compare_cols = [c for c in common if c not in key_cols]

    sas_k = sas[common].drop_duplicates(subset=key_cols)
    spark_k = spark[common].drop_duplicates(subset=key_cols)

    merged = sas_k.merge(spark_k, on=key_cols, how="outer",
                         suffixes=("_sas", "_spark"), indicator=True)

    only_in_sas = merged[merged["_merge"] == "left_only"]
    only_in_spark = merged[merged["_merge"] == "right_only"]
    both = merged[merged["_merge"] == "both"]

    c1, c2, c3 = st.columns(3)
    c1.metric("Rows only in SAS", len(only_in_sas))
    c2.metric("Rows only in PySpark", len(only_in_spark))
    c3.metric("Rows in both", len(both))

    if len(only_in_sas):
        with st.expander(f"Rows only in SAS ({len(only_in_sas)})"):
            st.dataframe(only_in_sas[key_cols + [c + "_sas" for c in compare_cols
                                                  if c + "_sas" in only_in_sas.columns]],
                         use_container_width=True)
    if len(only_in_spark):
        with st.expander(f"Rows only in PySpark ({len(only_in_spark)})"):
            st.dataframe(only_in_spark[key_cols + [c + "_spark" for c in compare_cols
                                                    if c + "_spark" in only_in_spark.columns]],
                         use_container_width=True)

    # Detect value differences in matched rows
    diff_rows = []
    for col in compare_cols:
        sc = col + "_sas"
        pc = col + "_spark"
        if sc not in both.columns or pc not in both.columns:
            continue
        sv = both[sc]
        pv = both[pc]
        if pd.api.types.is_numeric_dtype(sv) and pd.api.types.is_numeric_dtype(pv):
            mask = (sv - pv).abs() > tol
        else:
            mask = sv.astype(str).str.strip() != pv.astype(str).str.strip()
        mask = mask & sv.notna() & pv.notna()
        differing = both[mask]
        if len(differing):
            for _, r in differing.iterrows():
                diff_rows.append({
                    **{k: r[k] for k in key_cols},
                    "Column": col,
                    "SAS value": r[sc],
                    "PySpark value": r[pc],
                })

    if diff_rows:
        df_diff = pd.DataFrame(diff_rows)
        st.write(badge(False), f"**{len(df_diff)}** value difference(s) found:")
        st.dataframe(df_diff, use_container_width=True)
        csv = df_diff.to_csv(index=False).encode()
        st.download_button("Download differences as CSV", csv,
                           "value_differences.csv", "text/csv")
    else:
        st.write(badge(True), "No value differences found in matched rows.")


def section_distribution(sas: pd.DataFrame, spark: pd.DataFrame):
    st.subheader("Column Value Distribution")
    common = sorted(set(sas.columns) & set(spark.columns))
    col = st.selectbox("Select column", options=common, key="dist_col")
    if col:
        top_n = st.slider("Top N values", 5, 50, 20, key="dist_top")
        vc_sas = sas[col].astype(str).value_counts().head(top_n).rename("SAS count")
        vc_spark = spark[col].astype(str).value_counts().head(top_n).rename("PySpark count")
        df_dist = pd.concat([vc_sas, vc_spark], axis=1).fillna(0).astype(int)
        df_dist["Delta"] = df_dist["SAS count"] - df_dist["PySpark count"]
        st.dataframe(df_dist, use_container_width=True)
        st.bar_chart(df_dist[["SAS count", "PySpark count"]])


# ── main ─────────────────────────────────────────────────────────────────────

st.title("SAS vs PySpark Output Comparison")
st.markdown("Upload the SAS output and PySpark output files (CSV, Excel, or Parquet).")

col_a, col_b = st.columns(2)
with col_a:
    sas_file = st.file_uploader("SAS output", type=["csv", "xls", "xlsx", "parquet"],
                                key="sas_upload")
with col_b:
    spark_file = st.file_uploader("PySpark output", type=["csv", "xls", "xlsx", "parquet"],
                                  key="spark_upload")

normalise = st.checkbox("Normalise column names (lowercase + strip whitespace)", value=True)

if sas_file and spark_file:
    sas_df = load_file(sas_file)
    spark_df = load_file(spark_file)

    if sas_df is not None and spark_df is not None:
        if normalise:
            sas_df = normalise_cols(sas_df)
            spark_df = normalise_cols(spark_df)

        with st.expander("Preview SAS data (first 5 rows)"):
            st.dataframe(sas_df.head(), use_container_width=True)
        with st.expander("Preview PySpark data (first 5 rows)"):
            st.dataframe(spark_df.head(), use_container_width=True)

        st.divider()
        checks = st.multiselect(
            "Select checks to run",
            options=["Row Counts", "Column Check", "Null Check",
                     "Summary Statistics", "Duplicate Check",
                     "Value Differences", "Distribution"],
            default=["Row Counts", "Column Check", "Null Check",
                     "Summary Statistics", "Duplicate Check",
                     "Value Differences"],
        )

        if "Row Counts" in checks:
            st.divider()
            section_row_counts(sas_df, spark_df)

        if "Column Check" in checks:
            st.divider()
            section_column_check(sas_df, spark_df)

        if "Null Check" in checks:
            st.divider()
            section_null_check(sas_df, spark_df)

        if "Summary Statistics" in checks:
            st.divider()
            section_summary_stats(sas_df, spark_df)

        if "Duplicate Check" in checks:
            st.divider()
            section_duplicates(sas_df, spark_df)

        if "Value Differences" in checks:
            st.divider()
            section_value_diff(sas_df, spark_df)

        if "Distribution" in checks:
            st.divider()
            section_distribution(sas_df, spark_df)
else:
    st.info("Upload both files above to start comparing.")
