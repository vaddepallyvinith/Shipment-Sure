import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ===============================
# CONFIGURATION
# ===============================
DATA_FILE = "FeaturesImprovedDataSet.csv"
TARGET = "on_time_delivery"

# ===============================
# LOAD DATA (MEMORY SAFE)
# ===============================
df = pd.read_csv(DATA_FILE, engine="python", low_memory=True)

# OPTIONAL: sample to reduce clutter
df = df.sample(3000, random_state=42)

# ===============================
# SELECT IMPORTANT FEATURES
# ===============================
features = [
    "Promised_Lead_Time_Days",
    "shipping_distance_km",
    "supplier_rating",
    "Risk_Score",
    "Cost_Per_Km"
]

# ===============================
# BOX PLOTS (ONE PAGE)
# ===============================
plt.figure(figsize=(14, 8))

for i, feature in enumerate(features, 1):
    plt.subplot(2, 3, i)
    sns.boxplot(
        x=TARGET,
        y=feature,
        data=df,
        showfliers=False
    )
    plt.title(feature)
    plt.xlabel("On-Time Delivery (0 = Delayed, 1 = On Time)")

plt.suptitle("Feature Impact on On-Time Delivery (Box Plot Analysis)", fontsize=16)
plt.tight_layout()
plt.show()
