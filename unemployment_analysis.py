# ============================================================
# UNEMPLOYMENT ANALYSIS WITH PYTHON — CodeAlpha Task 2
# Dataset: Unemployment in India (Kaggle)
# URL: https://www.kaggle.com/datasets/gokulrajkmv/unemployment-in-india
# ============================================================

# ── 1. IMPORTS ──────────────────────────────────────────────
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings("ignore")

plt.style.use("seaborn-v0_8-whitegrid")
PALETTE = sns.color_palette("tab20", 20)
ACCENT = ["#C0392B", "#2980B9", "#27AE60", "#F39C12", "#8E44AD"]

print("=" * 62)
print("  UNEMPLOYMENT ANALYSIS — CodeAlpha Internship Task 2")
print("=" * 62)

# ── 2. LOAD DATA ─────────────────────────────────────────────
# Download from: https://www.kaggle.com/datasets/gokulrajkmv/unemployment-in-india
# File: 'Unemployment in India.csv'

df = pd.read_csv("Unemployment in India.csv")
print(f"\n📦 Dataset loaded: {df.shape[0]} rows × {df.shape[1]} columns")
print(f"   Columns: {list(df.columns)}")
print("\n🔍 First 5 rows:")
print(df.head())

# ── 3. DATA CLEANING ──────────────────────────────────────────
print("\n🧹 Cleaning Data...")
df.columns = [c.strip().replace(" ", "_") for c in df.columns]

# Parse date
date_col = [c for c in df.columns if "date" in c.lower() or "Date" in c][0]
df[date_col] = pd.to_datetime(df[date_col], dayfirst=True, errors="coerce")
df.dropna(subset=[date_col], inplace=True)
df.sort_values(date_col, inplace=True)
df.reset_index(drop=True, inplace=True)

# Identify key columns dynamically
unemp_col = [c for c in df.columns if "Estimated_Unemployment_Rate" in c or "Unemployment" in c][0]
labour_col = [c for c in df.columns if "Labour" in c or "labor" in c.lower()][0] if any("Labour" in c for c in df.columns) else None
region_col = [c for c in df.columns if "Region" in c or "State" in c][0]

print(f"   Date column      : {date_col}")
print(f"   Unemployment col : {unemp_col}")
print(f"   Region col       : {region_col}")
print(f"\n   Date range : {df[date_col].min().date()} → {df[date_col].max().date()}")
print(f"   Regions    : {df[region_col].nunique()} unique")
print(f"\n   Missing values:\n{df.isnull().sum().to_string()}")
df.dropna(subset=[unemp_col], inplace=True)

# ── 4. COVID PERIOD FLAG ──────────────────────────────────────
covid_start = pd.Timestamp("2020-03-01")
df["Period"] = df[date_col].apply(lambda d: "COVID Era" if d >= covid_start else "Pre-COVID")
print(f"\n🦠 COVID period flag added (from {covid_start.date()})")
pre = df[df["Period"] == "Pre-COVID"][unemp_col]
post = df[df["Period"] == "COVID Era"][unemp_col]
print(f"   Pre-COVID avg unemployment  : {pre.mean():.2f}%")
print(f"   COVID Era avg unemployment  : {post.mean():.2f}%")
print(f"   Increase                    : +{post.mean() - pre.mean():.2f} percentage points")

# Statistical significance test
t_stat, p_val = stats.ttest_ind(pre, post)
print(f"   T-test p-value              : {p_val:.6f} {'(significant ✅)' if p_val < 0.05 else '(not significant)'}")

# ── 5. EDA — NATIONAL TREND ───────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(16, 11))
fig.suptitle("Unemployment Analysis — National & Regional Trends", fontsize=15, fontweight="bold")

# 5a. National trend over time
national = df.groupby(date_col)[unemp_col].mean().reset_index()
axes[0][0].plot(national[date_col], national[unemp_col], color=ACCENT[1], linewidth=2)
axes[0][0].axvline(x=covid_start, color=ACCENT[0], linestyle="--", linewidth=2, label="COVID-19 Start (Mar 2020)")
axes[0][0].fill_between(national[date_col], national[unemp_col],
                         alpha=0.15, color=ACCENT[1])
axes[0][0].set_title("National Unemployment Rate Over Time", fontsize=12, fontweight="bold")
axes[0][0].set_xlabel("Date"); axes[0][0].set_ylabel("Unemployment Rate (%)")
axes[0][0].legend(); axes[0][0].xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))

# 5b. Pre-COVID vs COVID Era distribution
axes[0][1].hist(pre, bins=20, alpha=0.7, color=ACCENT[1], label="Pre-COVID", edgecolor="white")
axes[0][1].hist(post, bins=20, alpha=0.7, color=ACCENT[0], label="COVID Era", edgecolor="white")
axes[0][1].axvline(pre.mean(), color=ACCENT[1], linestyle="--", linewidth=2)
axes[0][1].axvline(post.mean(), color=ACCENT[0], linestyle="--", linewidth=2)
axes[0][1].set_title("Pre-COVID vs COVID Era Distribution", fontsize=12, fontweight="bold")
axes[0][1].set_xlabel("Unemployment Rate (%)"); axes[0][1].set_ylabel("Frequency")
axes[0][1].legend()

# 5c. Top 10 states by avg unemployment
state_avg = (df.groupby(region_col)[unemp_col].mean()
               .sort_values(ascending=False).head(10))
state_avg.plot(kind="barh", ax=axes[1][0],
               color=sns.color_palette("Reds_r", len(state_avg)), edgecolor="black")
axes[1][0].set_title("Top 10 Regions — Avg Unemployment Rate", fontsize=12, fontweight="bold")
axes[1][0].set_xlabel("Avg Unemployment Rate (%)")
axes[1][0].invert_yaxis()

# 5d. COVID impact by state (change in mean)
state_pre  = df[df["Period"] == "Pre-COVID"].groupby(region_col)[unemp_col].mean()
state_post = df[df["Period"] == "COVID Era"].groupby(region_col)[unemp_col].mean()
impact = (state_post - state_pre).dropna().sort_values(ascending=False).head(10)
impact.plot(kind="bar", ax=axes[1][1],
            color=[ACCENT[0] if v > 0 else ACCENT[2] for v in impact.values],
            edgecolor="black")
axes[1][1].axhline(0, color="black", linewidth=1)
axes[1][1].set_title("COVID Impact — Change in Unemployment Rate (Top 10)", fontsize=12, fontweight="bold")
axes[1][1].set_xlabel("Region"); axes[1][1].set_ylabel("Δ Unemployment Rate (%)")
axes[1][1].tick_params(axis="x", rotation=30)

plt.tight_layout()
plt.savefig("unemployment_trends.png", dpi=150, bbox_inches="tight")
plt.show()
print("\n✅ Trend plot saved → unemployment_trends.png")

# ── 6. HEATMAP — STATE × MONTH ────────────────────────────────
df["Month"] = df[date_col].dt.month
df["Year"] = df[date_col].dt.year
df["Year_Month"] = df[date_col].dt.to_period("M").astype(str)

pivot = df.pivot_table(values=unemp_col, index=region_col,
                        columns="Year_Month", aggfunc="mean")

fig, ax = plt.subplots(figsize=(18, 10))
sns.heatmap(pivot, ax=ax, cmap="YlOrRd", linewidths=0.3,
            cbar_kws={"label": "Unemployment Rate (%)"})
ax.set_title("Unemployment Rate Heatmap — Region × Month", fontsize=14, fontweight="bold")
ax.set_xlabel("Year-Month"); ax.set_ylabel("Region")
ax.tick_params(axis="x", rotation=45, labelsize=8)
plt.tight_layout()
plt.savefig("heatmap_region_month.png", dpi=150, bbox_inches="tight")
plt.show()
print("✅ Heatmap saved → heatmap_region_month.png")

# ── 7. SEASONAL ANALYSIS ──────────────────────────────────────
monthly_avg = df.groupby("Month")[unemp_col].mean()
month_names = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].bar(month_names, monthly_avg.values,
            color=sns.color_palette("coolwarm", 12), edgecolor="black")
axes[0].set_title("Seasonal Pattern — Monthly Avg Unemployment", fontsize=12, fontweight="bold")
axes[0].set_xlabel("Month"); axes[0].set_ylabel("Avg Unemployment Rate (%)")
axes[0].tick_params(axis="x", rotation=30)

# Year-wise trend
yearly = df.groupby("Year")[unemp_col].mean()
axes[1].plot(yearly.index, yearly.values, marker="o", linewidth=2.5,
             color=ACCENT[3], markersize=8, markeredgecolor="black")
axes[1].axvline(x=2020, color=ACCENT[0], linestyle="--", linewidth=2, label="COVID-19 (2020)")
axes[1].set_title("Year-wise Average Unemployment Rate", fontsize=12, fontweight="bold")
axes[1].set_xlabel("Year"); axes[1].set_ylabel("Avg Unemployment Rate (%)")
axes[1].legend(); axes[1].set_xticks(yearly.index)

plt.tight_layout()
plt.savefig("seasonal_yearly.png", dpi=150, bbox_inches="tight")
plt.show()
print("✅ Seasonal/yearly plot saved → seasonal_yearly.png")

# ── 8. LABOUR PARTICIPATION RATE (if available) ───────────────
if labour_col:
    corr_val = df[[unemp_col, labour_col]].corr().iloc[0, 1]
    print(f"\n📌 Correlation: Unemployment vs Labour Participation = {corr_val:.3f}")

# ── 9. INSIGHTS SUMMARY ───────────────────────────────────────
print("\n" + "=" * 62)
print("  📋 KEY INSIGHTS SUMMARY")
print("=" * 62)
print(f"\n1. National Avg Unemployment : {df[unemp_col].mean():.2f}%")
print(f"2. Pre-COVID Average         : {pre.mean():.2f}%")
print(f"3. COVID Era Average         : {post.mean():.2f}%")
print(f"4. COVID Impact              : +{post.mean() - pre.mean():.2f} pp increase")
print(f"5. Statistical Significance  : p = {p_val:.6f} ({'Significant' if p_val < 0.05 else 'Not Significant'})")
print(f"\n6. Top 3 Most Affected Regions by COVID:")
for i, (region, val) in enumerate(impact.head(3).items(), 1):
    print(f"   {i}. {region:30s} +{val:.2f}%")
print(f"\n7. Highest Avg Unemployment Region: {state_avg.idxmax()} ({state_avg.max():.2f}%)")
peak_month = monthly_avg.idxmax()
print(f"8. Peak Unemployment Month         : {month_names[peak_month - 1]}")
print("\n✅ Full analysis complete!")