%pip install tqdm


# 01: IMPORTS & CONFIG

from pyspark.sql import Row
from pyspark.sql import functions as F
from pyspark.sql.types import *
from datetime import date


table_name = "`medical-affairs-ma-datalake-tst`.edp_medical_affairs_stg.apttus_agreement"

s3_path = [
    "s3://gilead-medical-affairs-tst-us-west-2-curated/staging/APTTUS/APTTUS_AGREEMENT/pt_data_dt=20260115/pt_cycle_id=20260115140342672517/part-00000-fbfa9854-cb96-4e88-93cf-65b235f6d3f8-c000.snappy.parquet",
    "s3://gilead-medical-affairs-tst-us-west-2-curated/staging/APTTUS/APTTUS_AGREEMENT/pt_data_dt=20260109/pt_cycle_id=20260109114840395178/"
]

user_email = dbutils.notebook.entry_point.getDbutils().notebook().getContext().userName().get()
load_date_val = str(date.today())


# 02: LOAD RAW DATA (FOR SCHEMA CHECK)

df_table_raw = spark.table(table_name)
df_s3_raw    = spark.read.parquet(*s3_path)

df_table_raw.cache()
df_s3_raw.cache()

table_cnt = df_table_raw.count()
s3_cnt = df_s3_raw.count()

print("Databricks Table Rows :", table_cnt)
print("S3 File Rows          :", s3_cnt)


# 03: SCHEMA COMPARISON (RAW DATA)

def schema_map(schema):
    return {f.name.lower(): f.dataType.simpleString() for f in schema.fields}

a = schema_map(df_table_raw.schema)
b = schema_map(df_s3_raw.schema)

schema_rows = [
    (c, a.get(c), b.get(c), a.get(c) == b.get(c))
    for c in sorted(set(a) | set(b))
]

schema_df = spark.createDataFrame(
    schema_rows, ["column_name", "datatype_table", "datatype_s3", "is_match"]
)

matched = schema_df.filter("is_match = true").count()
total_cols = schema_df.count()
schema_match_pct = round((matched / total_cols) * 100, 2) if total_cols else 0

print(f"Schema Match Summary: {matched}/{total_cols} columns ({schema_match_pct}%)")
display(schema_df.orderBy(F.col("is_match").desc()))


# 04: NORMALIZATION (ONLY STRING COLUMNS)

def normalize_expr_dtype(field):
    c = field.name
    if isinstance(field.dataType, StringType):
        base = F.trim(
            F.regexp_replace(
                F.regexp_replace(F.col(c), r"[^\x00-\x7F]", ""),
                r"[^\w\s]", ""
            )
        )
        return (
            F.when(base.isin("", "null", "NULL", "NaN", "nan", "None", "NONE", "N/A", "n/a"), None)
             .otherwise(base)
             .alias(c)
        )
    else:
        return F.col(c)

df_table = df_table_raw.select(*[normalize_expr_dtype(f) for f in df_table_raw.schema.fields])
df_s3    = df_s3_raw.select(*[normalize_expr_dtype(f) for f in df_s3_raw.schema.fields])



# 05: COMPOSITE KEY

exclude_cols = ["pt_data_dt", "pt_cycle_id"]

common_cols = list(
    set(df_table.columns).intersection(df_s3.columns) - set(exclude_cols)
)

df_table = df_table.withColumn(
    "composite_key", F.concat_ws("||", *[F.col(c).cast("string") for c in common_cols])
)

df_s3 = df_s3.withColumn(
    "composite_key", F.concat_ws("||", *[F.col(c).cast("string") for c in common_cols])
)




# 06: DUPLICATE CHECK

def dup_count(df):
    return df.groupBy("composite_key").count().filter("count > 1").count()

dupe_table_cnt = dup_count(df_table)
dupe_s3_cnt = dup_count(df_s3)

print("Duplicate Keys in Table:", dupe_table_cnt)
print("Duplicate Keys in S3   :", dupe_s3_cnt)

if dupe_table_cnt > 0:
    df_table = df_table.dropDuplicates(["composite_key"])

if dupe_s3_cnt > 0:
    df_s3 = df_s3.dropDuplicates(["composite_key"])
    
    
    
    
# 07: JOIN

joined_df = (
    df_table.alias("table")
    .join(df_s3.alias("s3"), "composite_key", "inner")
    .cache()
)

joined_total = joined_df.count()
print("Joined Rows:", joined_total)



# 08: VALUE COMPARISON

from tqdm.notebook import tqdm
compare_cols = [c for c in df_table.columns if c in df_s3.columns and c != "composite_key"]

match_aggs = [
    F.count(
        F.when(
            (F.col(f"table.{c}").isNull() & F.col(f"s3.{c}").isNull()) |
            (F.col(f"table.{c}") == F.col(f"s3.{c}")),
            1
        )
    ).alias(c)
    for c in tqdm(compare_cols)
]

agg = joined_df.agg(*match_aggs).collect()[0].asDict()

# sample mismatches (1 per column)
mismatch_samples = {}
for c in tqdm(compare_cols):
    r = (
        joined_df
        .filter(~(
            (F.col(f"table.{c}").isNull() & F.col(f"s3.{c}").isNull()) |
            (F.col(f"table.{c}") == F.col(f"s3.{c}"))
        ))
        .select(
            "composite_key",
            F.col(f"table.{c}"),
            F.col(f"s3.{c}")
        )
        .limit(1)
        .collect()
    )
    mismatch_samples[c] = r[0] if r else (None, None, None)
    
    
    
# 09: FINAL SUMMARY DF

from tqdm.notebook import tqdm
summary = []
total_match = 0
composite_key_cols_str = ", ".join(common_cols)

dtype_t_map = {f.name: str(f.dataType) for f in df_table_raw.schema.fields}
dtype_s_map = {f.name: str(f.dataType) for f in df_s3_raw.schema.fields}

for c in tqdm(compare_cols):
    mc = agg[c]
    mm = joined_total - mc
    pct = round((mc / joined_total) * 100, 2) if joined_total else 0

    k, tv, sv = mismatch_samples[c]
    total_match += mc

    summary.append((
        c, dtype_t_map[c], dtype_s_map[c],
        mc, mm, joined_total, pct, 100 - pct,
        k, tv, sv, load_date_val, user_email, composite_key_cols_str
    ))

overall_pct = round((total_match / (joined_total * len(compare_cols))) * 100, 2) if joined_total else 0

sample_pk = (
    joined_df.select("composite_key").dropDuplicates().limit(1).collect()[0][0]
    if joined_total else "NO_PK"
)

summary.append((
    "OVERALL_SUMMARY",
    f"Total_Columns: {len(compare_cols)}",
    "-",
    total_match,
    joined_total * len(compare_cols) - total_match,
    joined_total,
    overall_pct,
    100 - overall_pct,
    sample_pk,
    "-",
    "-",
    load_date_val,
    user_email,
    composite_key_cols_str
))

summary_schema = StructType([
    StructField("column_name", StringType()),
    StructField("data_type_Databricks_Delta_table", StringType()),
    StructField("data_type_S3_Table", StringType()),
    StructField("match_count", DoubleType()),
    StructField("mismatch_count", DoubleType()),
    StructField("total_row_count", DoubleType()),
    StructField("match_percentage", DoubleType()),
    StructField("mismatch_percentage", DoubleType()),
    StructField("Sample_Primary_Key", StringType()),
    StructField("Sample_Databricks_Delta_Table_Value", StringType()),
    StructField("Sample_S3_Table_Value", StringType()),
    StructField("load_date", StringType()),
    StructField("userid", StringType()),
    StructField("Composite_Key_Columns", StringType())
])

final_summary_df = spark.createDataFrame(summary, summary_schema)
display(final_summary_df.orderBy(F.col("match_percentage").desc()))



# 10: SUMMARY COUNTS DF

summary_stats = [
    Row(Metric="Total Rows (Databricks Table)", Value=table_cnt),
    Row(Metric="Total Rows (S3 File)", Value=s3_cnt),
    Row(Metric="Joined Rows (Inner Join)", Value=joined_total),
    Row(Metric="Total Columns (Databricks Table)", Value=len(df_table_raw.columns)),
    Row(Metric="Total Columns (S3 File)", Value=len(df_s3_raw.columns)),
    Row(Metric="Common Columns Compared", Value=len(compare_cols)),
]

summary_counts_df = spark.createDataFrame(summary_stats)
display(summary_counts_df)


# 11: SOURCE & SCRIPT NAME

def extract_source_and_script_from_table(tname: str):
    cleaned = tname.replace("`", "")
    tbl = cleaned.split(".")[-1]
    return tbl.split("_")[0].upper(), tbl.lower()

source_name, script_name = extract_source_and_script_from_table(table_name)


# 12: S3 OUTPUT

base_path = (
    f"s3://gilead-medical-affairs-tst-us-west-2-raw/db_schema/"
    f"{source_name}/{script_name.upper()}_Comparison_Report"
)

final_summary_path = f"{base_path}/final_summary"
summary_counts_path = f"{base_path}/summary_counts"

final_summary_df.coalesce(1).write.mode("overwrite").parquet(final_summary_path)
summary_counts_df.coalesce(1).write.mode("overwrite").parquet(summary_counts_path)

print("Saved parquet outputs to S3.")



# 13: WRITE TO DELTA

catalog = "`medical-affairs-ma-datalake-tst`"
schema  = "edp_medical_affairs_stg"

table_final  = f"{catalog}.{schema}.comparison_final_summary"
table_counts = f"{catalog}.{schema}.comparison_summary_counts"

run_date = F.current_timestamp()

df_final_ext = (
    final_summary_df
        .withColumn("source_name", F.lit(source_name))
        .withColumn("script_name", F.lit(script_name))
        .withColumn("run_date", run_date)
)

df_counts_ext = (
    summary_counts_df
        .withColumn("source_name", F.lit(source_name))
        .withColumn("script_name", F.lit(script_name))
        .withColumn("run_date", run_date)
)

(
    df_final_ext.write
        .format("delta")
        .mode("append")
        .option("mergeSchema", "true")
        .partitionBy("source_name", "script_name", "run_date")
        .saveAsTable(table_final)
)

(
    df_counts_ext.write
        .format("delta")
        .mode("append")
        .option("mergeSchema", "true")
        .partitionBy("source_name", "script_name", "run_date")
        .saveAsTable(table_counts)
)

print("🎉 Migration output stored successfully!")
print("Final Summary Table :", table_final)
print("Summary Counts Table:", table_counts)
 
print("\nQuery your output using:")
print(f"""
SELECT *
FROM {table_final}
WHERE source_name = '{source_name}'
  AND script_name = '{script_name}'
ORDER BY run_date DESC;
""")