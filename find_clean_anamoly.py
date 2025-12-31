import pandas as pd
import numpy as np

# =====================================================
# CONFIG
# =====================================================
INPUT_FILE = "RawDataSet.csv"
OUTPUT_FILE = "AnamolyFreeDataSet.csv"

NUMERICAL_COLS = [
    "supplier_rating",
    "supplier_lead_time",
    "shipping_distance_km",
    "order_quantity",
    "unit_price",
    "total_order_value",
    "previous_on_time_rate"
]

IQR_MULTIPLIER = 1.5

# =====================================================
# LOAD DATA (SAFE)
# =====================================================
try:
    df = pd.read_csv(INPUT_FILE)
    print("✅ Dataset loaded successfully")
except FileNotFoundError:
    raise FileNotFoundError(f"❌ File not found: {INPUT_FILE}")

# Work on a copy
df_clean = df.copy()

print("\n🔍 Starting Anomaly Detection & Winsorization")
print("=" * 60)

# =====================================================
# FUNCTION: IQR CAPPING
# =====================================================
def cap_outliers_iqr(data, columns, multiplier=1.5):
    """
    Caps outliers using the IQR method (winsorization)
    Returns capped DataFrame and summary statistics
    """
    summary = []

    for col in columns:
        if col not in data.columns:
            print(f"⚠️ Column skipped (not found): {col}")
            continue

        series = data[col]

        # Skip non-numeric or constant columns
        if not np.issubdtype(series.dtype, np.number) or series.nunique() <= 1:
            print(f"⚠️ Column skipped (non-numeric/constant): {col}")
            continue

        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1

        lower = Q1 - multiplier * IQR
        upper = Q3 + multiplier * IQR

        low_cnt = (series < lower).sum()
        high_cnt = (series > upper).sum()

        data[col] = series.clip(lower=lower, upper=upper)

        summary.append({
            "Feature": col,
            "Lower_Bound": round(lower, 2),
            "Upper_Bound": round(upper, 2),
            "Lower_Anomalies": low_cnt,
            "Upper_Anomalies": high_cnt,
            "Total_Capped": low_cnt + high_cnt
        })

    return data, pd.DataFrame(summary)

# =====================================================
# APPLY CAPPING
# =====================================================
df_clean, cap_summary = cap_outliers_iqr(
    df_clean,
    NUMERICAL_COLS,
    multiplier=IQR_MULTIPLIER
)

# =====================================================
# SUMMARY REPORT
# =====================================================
total_capped = cap_summary["Total_Capped"].sum()

print("\n📊 CAPPING SUMMARY")
print(cap_summary.to_string(index=False))

print("\n" + "-" * 60)
print(f"✅ Total anomalies capped: {total_capped}")
print("✅ Cleaned data stored in 'df_clean'")

# =====================================================
# FINAL SAFETY CHECK
# =====================================================
df_clean.replace([np.inf, -np.inf], np.nan, inplace=True)
df_clean.fillna(df_clean.median(numeric_only=True), inplace=True)

# =====================================================
# SAVE OUTPUT
# =====================================================
df_clean.to_csv(OUTPUT_FILE, index=False)
print(f"\n📁 Cleaned dataset saved to: {OUTPUT_FILE}")
print("\nFirst 5 rows:")
print(df_clean.head())
