import kagglehub
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error
from mpl_toolkits.basemap import Basemap

# ============================================================
# 1. Load data
# ============================================================

path = kagglehub.dataset_download("sobhanmoosavi/us-accidents")
df = pd.read_csv(path + "/US_Accidents_March23.csv")
df = df.dropna(subset=["County", "State"])
print("Data loaded:", df.shape)

# ============================================================
# 2. Feature engineering
# ============================================================

df["Start_Time"] = pd.to_datetime(df["Start_Time"], errors="coerce")
df = df.dropna(subset=["Start_Time"])

df["hour"] = df["Start_Time"].dt.hour
df["dayofweek"] = df["Start_Time"].dt.dayofweek
df["is_night"] = ((df["hour"] < 6) | (df["hour"] >= 20)).astype(int)
df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)

num_cols = ["Severity", "Distance(mi)", "Temperature(F)", "Visibility(mi)", "Precipitation(in)"]
for c in num_cols:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

# ============================================================
# 3. Aggregate to county-level features
# ============================================================

group_cols = ["State", "County"]
agg_dict = {
    "ID": "count",
    "Severity": "mean",
    "Distance(mi)": "mean",
    "Temperature(F)": "mean",
    "Visibility(mi)": "mean",
    "Precipitation(in)": "mean",
    "is_night": "mean",
    "is_weekend": "mean"
}

cols_to_keep = list(set(group_cols + list(agg_dict.keys())))
df_small = df[cols_to_keep].copy()

county_df = df_small.groupby(group_cols).agg(agg_dict).reset_index()
county_df = county_df.rename(columns={"ID": "total_accidents"})

county_df = county_df.dropna(subset=["total_accidents"])
feature_cols = [c for c in county_df.columns if c not in ["State", "County", "total_accidents"]]
county_df = county_df.dropna(subset=feature_cols)
print("County data ready:", county_df.shape)

# ============================================================
# 4. Train-test split
# ============================================================

X = county_df[feature_cols].values
y = county_df["total_accidents"].values
X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
    X, y, county_df.index, test_size=0.2, random_state=42
)

# ============================================================
# 5. Train Random Forest
# ============================================================

rf = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)

y_pred_test = rf.predict(X_test)
print("R² on test:", r2_score(y_test, y_pred_test))
print("MAE on test:", mean_absolute_error(y_test, y_pred_test))

# ============================================================
# 6. Predict risk
# ============================================================

all_preds = rf.predict(X)
min_pred = all_preds.min()
max_pred = all_preds.max()
county_df["risk_score"] = 100 * (all_preds - min_pred) / (max_pred - min_pred + 1e-9)

# ============================================================
# 7. Top counties
# ============================================================

top_n = 20
top_risk = county_df.sort_values("risk_score", ascending=False).head(top_n)
print("\nTop 20 high-risk counties:")
print(top_risk[["State", "County", "total_accidents", "risk_score"]])

# ============================================================
# 8. Plot US bubble map with gradient circles
# ============================================================

# Compute mean lat/lon per county
county_coords = df.groupby(["State", "County"]).agg({
    "Start_Lat": "mean",
    "Start_Lng": "mean"
}).reset_index()

plot_df = county_df.merge(county_coords, on=["State", "County"])

# Handle extreme outliers
vmax = np.percentile(plot_df["risk_score"], 99)
colors = np.clip(plot_df["risk_score"], None, vmax)

plt.figure(figsize=(14,8))
m = Basemap(
     llcrnrlon=-125,  # moved a bit west
    llcrnrlat=20,    # moved a bit south to include Florida Keys
    urcrnrlon=-65,   # moved a bit east
    urcrnrlat=52,    # moved a bit north
    projection='lcc',
    lat_1=33, lat_2=45,
    lon_0=-95
)
m.drawcoastlines()
m.drawcountries()
m.drawstates()

x, y = m(plot_df["Start_Lng"].values, plot_df["Start_Lat"].values)

base_size = 150

# Normalize risk for colormap
norm = matplotlib.colors.Normalize(vmin=0, vmax=vmax)
cmap = plt.cm.Reds
colors_mapped = cmap(norm(colors))

ax = plt.gca()

# Plot gradient circles
for i in range(len(x)):
    for alpha, scale in zip([0.4, 0.2, 0.1], [1.0, 1.5, 2.0]):
        m.scatter(
            x[i], y[i],
            s=base_size * scale,
            color=colors_mapped[i],
            alpha=alpha,
            edgecolors='none'
        )

# Add colorbar
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cb = plt.colorbar(sm, ax=ax, label="Risk Score (0-100, clipped at 99th percentile)")

plt.title("US Accident Risk Bubble Map with Gradient Circles")
plt.show()