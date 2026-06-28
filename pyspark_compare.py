"""
SAS vs PySpark Output Comparison — PySpark Commands
----------------------------------------------------
Run:  spark-submit pyspark_compare.py
      or paste individual sections into a notebook / PySpark shell.

Replace SAS_PATH / SPARK_PATH with your actual file paths.
Supported formats: CSV, Parquet, ORC, JSON, Excel (via pandas bridge).
"""

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import NumericType

# ── 0. Start Spark ────────────────────────────────────────────────────────────

spark = (SparkSession.builder
         .appName("SAS_vs_PySpark_Compare")
         .getOrCreate())

spark.sparkContext.setLogLevel("WARN")

# ── 1. Load data ──────────────────────────────────────────────────────────────

SAS_PATH   = "sas_output.csv"       # change to your SAS export path
SPARK_PATH = "pyspark_output.csv"   # change to your PySpark output path

# CSV (inferSchema discovers data types automatically)
sas_df   = spark.read.csv(SAS_PATH,   header=True, inferSchema=True)
spark_df = spark.read.csv(SPARK_PATH, header=True, inferSchema=True)

# Parquet alternative:
# sas_df   = spark.read.parquet(SAS_PATH)
# spark_df = spark.read.parquet(SPARK_PATH)

# Normalise column names (strip spaces, lowercase) — optional
sas_df   = sas_df.toDF(*[c.strip().lower() for c in sas_df.columns])
spark_df = spark_df.toDF(*[c.strip().lower() for c in spark_df.columns])

print("\n" + "="*60)
print("      SAS vs PySpark Output Comparison Report")
print("="*60)


# ── 2. ROW COUNT CHECK ────────────────────────────────────────────────────────

print("\n[1] ROW COUNT CHECK")
print("-"*40)

sas_count   = sas_df.count()
spark_count = spark_df.count()
diff        = sas_count - spark_count

print(f"  SAS   rows  : {sas_count:,}")
print(f"  PySpark rows: {spark_count:,}")
print(f"  Difference  : {diff:,}")
print(f"  Result      : {'PASS' if diff == 0 else 'FAIL — counts differ'}")


# ── 3. COLUMN CHECK ───────────────────────────────────────────────────────────

print("\n[2] COLUMN CHECK")
print("-"*40)

sas_cols   = set(sas_df.columns)
spark_cols = set(spark_df.columns)
common     = sas_cols & spark_cols
only_sas   = sas_cols - spark_cols
only_spark = spark_cols - sas_cols

print(f"  SAS columns   : {len(sas_cols)}")
print(f"  PySpark cols  : {len(spark_cols)}")
print(f"  Common cols   : {len(common)}")

if only_sas:
    print(f"  Only in SAS   : {sorted(only_sas)}")
if only_spark:
    print(f"  Only in Spark : {sorted(only_spark)}")

# Data-type comparison for common columns
print("\n  Column dtype comparison:")
print(f"  {'Column':<30} {'SAS dtype':<20} {'PySpark dtype':<20} Match")
print(f"  {'-'*30} {'-'*20} {'-'*20} -----")

sas_schema   = {f.name: f.dataType for f in sas_df.schema}
spark_schema = {f.name: f.dataType for f in spark_df.schema}

dtype_mismatches = []
for col in sorted(common):
    st = str(sas_schema[col])
    pt = str(spark_schema[col])
    match = st == pt
    if not match:
        dtype_mismatches.append(col)
    flag = "OK" if match else "MISMATCH"
    print(f"  {col:<30} {st:<20} {pt:<20} {flag}")

print(f"\n  Dtype result: {'PASS' if not dtype_mismatches else f'FAIL — {len(dtype_mismatches)} mismatches: {dtype_mismatches}'}")


# ── 4. NULL / MISSING VALUE CHECK ────────────────────────────────────────────

print("\n[3] NULL / MISSING VALUE CHECK")
print("-"*40)

# Build a single aggregate expression per DataFrame
null_agg_sas   = [F.sum(F.col(c).isNull().cast("int")).alias(c) for c in common]
null_agg_spark = [F.sum(F.col(c).isNull().cast("int")).alias(c) for c in common]

sas_nulls   = sas_df.select(null_agg_sas).collect()[0].asDict()
spark_nulls = spark_df.select(null_agg_spark).collect()[0].asDict()

print(f"  {'Column':<30} {'SAS nulls':>12} {'Spark nulls':>12} {'Delta':>8}")
print(f"  {'-'*30} {'-'*12} {'-'*12} {'-'*8}")

null_diffs = []
for col in sorted(common):
    sn = sas_nulls[col]
    pn = spark_nulls[col]
    delta = sn - pn
    if delta != 0:
        null_diffs.append(col)
    flag = "" if delta == 0 else "  <-- DIFF"
    print(f"  {col:<30} {sn:>12,} {pn:>12,} {delta:>8,}{flag}")

print(f"\n  Null result : {'PASS' if not null_diffs else f'FAIL — differences in: {null_diffs}'}")


# ── 5. SUMMARY STATISTICS ─────────────────────────────────────────────────────

print("\n[4] SUMMARY STATISTICS (numeric columns)")
print("-"*40)

TOLERANCE = 0.0  # set > 0 to allow small floating-point differences

num_common = [c for c in sorted(common)
              if isinstance(sas_schema.get(c), NumericType)
              and isinstance(spark_schema.get(c), NumericType)]

if not num_common:
    print("  No common numeric columns found.")
else:
    # describe() returns: count, mean, stddev, min, max
    sas_stats   = sas_df.select(num_common).describe().toPandas().set_index("summary")
    spark_stats = spark_df.select(num_common).describe().toPandas().set_index("summary")

    # Add percentiles via approxQuantile
    QUANTILES = [0.25, 0.50, 0.75]
    sas_q   = sas_df.approxQuantile(num_common, QUANTILES, 0.001)
    spark_q = spark_df.approxQuantile(num_common, QUANTILES, 0.001)

    import pandas as pd
    q_labels = ["25%", "50%", "75%"]
    sas_q_df   = pd.DataFrame(sas_q,   index=num_common, columns=q_labels).T
    spark_q_df = pd.DataFrame(spark_q, index=num_common, columns=q_labels).T

    sas_full   = pd.concat([sas_stats, sas_q_df])
    spark_full = pd.concat([spark_stats, spark_q_df])

    stat_fail = []
    for col in num_common:
        print(f"\n  Column: {col}")
        print(f"  {'Stat':<10} {'SAS':>18} {'PySpark':>18} {'Abs diff':>14} Pass?")
        print(f"  {'-'*10} {'-'*18} {'-'*18} {'-'*14} -----")
        for stat in ["count", "mean", "stddev", "min", "25%", "50%", "75%", "max"]:
            try:
                sv = float(sas_full.at[stat, col])
                pv = float(spark_full.at[stat, col])
            except (KeyError, ValueError):
                continue
            diff = abs(sv - pv)
            ok   = diff <= TOLERANCE
            if not ok:
                stat_fail.append((col, stat))
            print(f"  {stat:<10} {sv:>18.4f} {pv:>18.4f} {diff:>14.6f} {'OK' if ok else 'FAIL'}")

    print(f"\n  Stats result: {'PASS' if not stat_fail else f'FAIL — {len(stat_fail)} stat(s) outside tolerance'}")


# ── 6. DUPLICATE CHECK ───────────────────────────────────────────────────────

print("\n[5] DUPLICATE CHECK")
print("-"*40)

sas_dup_count   = sas_df.count()   - sas_df.dropDuplicates().count()
spark_dup_count = spark_df.count() - spark_df.dropDuplicates().count()

print(f"  SAS    duplicates: {sas_dup_count:,}")
print(f"  PySpark duplicates: {spark_dup_count:,}")

if sas_dup_count > 0:
    print("  Sample SAS duplicate rows:")
    (sas_df
     .groupBy(sas_df.columns)
     .count()
     .filter(F.col("count") > 1)
     .drop("count")
     .show(5, truncate=False))

if spark_dup_count > 0:
    print("  Sample PySpark duplicate rows:")
    (spark_df
     .groupBy(spark_df.columns)
     .count()
     .filter(F.col("count") > 1)
     .drop("count")
     .show(5, truncate=False))

print(f"  Result: {'PASS — no duplicates in either dataset' if sas_dup_count == 0 and spark_dup_count == 0 else 'WARN — duplicates found'}")


# ── 7. ROW-LEVEL VALUE DIFF (key-based) ──────────────────────────────────────

print("\n[6] ROW-LEVEL VALUE DIFFERENCE CHECK")
print("-"*40)

# Set the column(s) that uniquely identify each row
KEY_COLS = ["id"]   # <-- change to your actual key column(s)

missing_keys = [k for k in KEY_COLS if k not in common]
if missing_keys:
    print(f"  Key columns not found in both datasets: {missing_keys}. Skipping.")
else:
    compare_cols = [c for c in sorted(common) if c not in KEY_COLS]

    sas_keyed   = sas_df.select(common).dropDuplicates(KEY_COLS)
    spark_keyed = spark_df.select(common).dropDuplicates(KEY_COLS)

    sas_alias   = sas_keyed.alias("sas")
    spark_alias = spark_keyed.alias("spark")

    joined = sas_alias.join(spark_alias, on=KEY_COLS, how="full_outer")

    # Rows only in SAS / only in PySpark
    only_in_sas   = joined.filter(
        F.col(f"spark.{compare_cols[0]}").isNull() if compare_cols else F.lit(False))
    only_in_spark = joined.filter(
        F.col(f"sas.{compare_cols[0]}").isNull() if compare_cols else F.lit(False))

    print(f"  Rows only in SAS    : {only_in_sas.count():,}")
    print(f"  Rows only in PySpark: {only_in_spark.count():,}")

    # Value differences in matched rows
    diff_conditions = []
    for col in compare_cols:
        sc = F.col(f"sas.{col}")
        pc = F.col(f"spark.{col}")
        if isinstance(sas_schema.get(col), NumericType):
            diff_conditions.append(F.abs(sc - pc) > TOLERANCE)
        else:
            diff_conditions.append(F.trim(sc.cast("string")) != F.trim(pc.cast("string")))

    if diff_conditions:
        any_diff = diff_conditions[0]
        for cond in diff_conditions[1:]:
            any_diff = any_diff | cond

        differing_rows = (joined
                          .filter(any_diff)
                          .select(
                              *KEY_COLS,
                              *[F.col(f"sas.{c}").alias(f"{c}_sas")   for c in compare_cols],
                              *[F.col(f"spark.{c}").alias(f"{c}_spark") for c in compare_cols],
                          ))

        diff_count = differing_rows.count()
        print(f"  Value differences   : {diff_count:,}")
        if diff_count > 0:
            print("  Sample differing rows:")
            differing_rows.show(10, truncate=False)
            # Save full diff report
            # differing_rows.write.csv("value_diff_report.csv", header=True, mode="overwrite")
        print(f"  Result: {'PASS' if diff_count == 0 else 'FAIL — see rows above'}")


print("\n" + "="*60)
print("  Comparison complete.")
print("="*60 + "\n")

spark.stop()
