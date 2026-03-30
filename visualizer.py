import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import os
from collections import defaultdict

# ── Output folder ──────────────────────────────────────────────────────────────
OUTPUT_DIR = "airbnb_plots"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def save(filename):
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {OUTPUT_DIR}/{filename}")

# ── Load & clean ───────────────────────────────────────────────────────────────
df = pd.read_csv("AB_NYC_2019.csv")
df["reviews_per_month"] = df["reviews_per_month"].fillna(0)
df["last_review"]       = df["last_review"].fillna("No reviews")
df = df.dropna(subset=["name", "host_name"])
df = df[df["price"] > 0]
Q1, Q3 = df["price"].quantile(0.25), df["price"].quantile(0.75)
df = df[df["price"] <= Q3 + 1.5 * (Q3 - Q1)]

print(f"\n{'='*55}")
print("  NYC Airbnb 2019 — Extended EDA")
print(f"  Dataset: {df.shape[0]:,} listings after cleaning")
print(f"{'='*55}\n")

# ── Colour palette ─────────────────────────────────────────────────────────────
BOROUGH_COLORS = {
    "Manhattan":    "#E45C3A",
    "Brooklyn":     "#3266AD",
    "Queens":       "#F5A623",
    "Bronx":        "#4CAF82",
    "Staten Island":"#9B59B6",
}
ROOM_COLORS = {
    "Entire home/apt": "#E45C3A",
    "Private room":    "#3266AD",
    "Shared room":     "#F5A623",
}

# 1. Top-10 hosts by number of listings
print("1. Top-10 hosts by listings")
top_hosts = df["host_name"].value_counts().head(10)
fig, ax = plt.subplots(figsize=(9, 5))
bars = ax.barh(top_hosts.index[::-1], top_hosts.values[::-1], color="#3266AD", edgecolor="none")
for bar, val in zip(bars, top_hosts.values[::-1]):
    ax.text(val + 0.5, bar.get_y() + bar.get_height() / 2,
            str(val), va="center", fontsize=10)
ax.set_xlabel("Number of Listings")
ax.set_title("Top 10 Hosts by Number of Listings")
ax.spines[["top", "right"]].set_visible(False)
save("01_top10_hosts.png")

# 2. Listings per neighbourhood group (pie)
print("2. Listings share by borough (pie)")
borough_counts = df["neighbourhood_group"].value_counts()
fig, ax = plt.subplots(figsize=(7, 7))
wedges, texts, autotexts = ax.pie(
    borough_counts,
    labels=borough_counts.index,
    autopct="%1.1f%%",
    colors=[BOROUGH_COLORS.get(b, "#888") for b in borough_counts.index],
    startangle=140, pctdistance=0.82,
    wedgeprops=dict(edgecolor="white", linewidth=2)
)
for t in autotexts:
    t.set_fontsize(10)
ax.set_title("Share of Listings by Borough", pad=20)
save("02_borough_pie.png")

# 3. Price violin plot by borough
print("3. Price violin by borough")
order = df.groupby("neighbourhood_group")["price"].median().sort_values(ascending=False).index
fig, ax = plt.subplots(figsize=(10, 6))
sns.violinplot(data=df, x="neighbourhood_group", y="price", order=order,
               palette=[BOROUGH_COLORS.get(b, "#888") for b in order],
               inner="quartile", linewidth=1.2, ax=ax)
ax.set_xlabel("Borough")
ax.set_ylabel("Price (USD)")
ax.set_title("Price Distribution by Borough (Violin)")
ax.spines[["top", "right"]].set_visible(False)
save("03_price_violin_borough.png")

# 4. Room-type share by borough (stacked 100% bar)
print("4. Room-type share by borough (stacked %)")
rt_borough = (df.groupby(["neighbourhood_group", "room_type"])
               .size().unstack(fill_value=0))
rt_pct = rt_borough.div(rt_borough.sum(axis=1), axis=0) * 100
rt_pct = rt_pct.loc[order]
fig, ax = plt.subplots(figsize=(10, 6))
bottom = np.zeros(len(rt_pct))
for room, color in ROOM_COLORS.items():
    if room in rt_pct.columns:
        ax.bar(rt_pct.index, rt_pct[room], bottom=bottom, label=room,
               color=color, edgecolor="white", linewidth=0.8)
        bottom += rt_pct[room].values
ax.set_ylabel("% of Listings")
ax.set_title("Room Type Mix by Borough (100% Stacked)")
ax.legend(loc="lower right", fontsize=9)
ax.spines[["top", "right"]].set_visible(False)
save("04_roomtype_stacked_borough.png")

# 5. Reviews per month distribution
print("5. Reviews per month distribution (fixed)")
active = df[df["reviews_per_month"] > 0]["reviews_per_month"]
log_bins = np.logspace(np.log10(active.min()), np.log10(active.max()), 45)

fig, ax = plt.subplots(figsize=(9, 5))
ax.hist(active, bins=log_bins, color="#E45C3A", edgecolor="white",
        linewidth=0.3, alpha=0.85)
ax.set_xscale("log")
ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
ax.set_xlabel("Reviews per Month (log scale)")
ax.set_ylabel("Number of Listings")
ax.set_title("Distribution of Reviews per Month (Active Listings)")
ax.spines[["top", "right"]].set_visible(False)
med_rpm = active.median()
ax.axvline(med_rpm, color="#333", linestyle="--", linewidth=1.2)
ax.text(med_rpm * 1.12, ax.get_ylim()[1] * 0.93,
        f"Median: {med_rpm:.2f}", fontsize=9)
save("05_reviews_per_month_dist.png")

# 6. Minimum-nights distribution (≤ 30 nights)
print("6. Minimum nights (≤ 30) distribution")
mn = df[df["minimum_nights"] <= 30]["minimum_nights"]
fig, ax = plt.subplots(figsize=(10, 5))
counts = mn.value_counts().sort_index()
ax.bar(counts.index, counts.values, color="#3266AD", edgecolor="none", width=0.8)
ax.set_xlabel("Minimum Nights Required")
ax.set_ylabel("Number of Listings")
ax.set_title("Minimum-Nights Distribution (capped at 30)")
ax.spines[["top", "right"]].set_visible(False)
save("06_min_nights_dist.png")

# 7. Availability 365 - KDE by room type
print("7. Availability KDE by room type")
fig, ax = plt.subplots(figsize=(9, 5))
for room, color in ROOM_COLORS.items():
    subset = df[df["room_type"] == room]["availability_365"]
    if len(subset) > 10:
        sns.kdeplot(subset, ax=ax, label=room, color=color, linewidth=2,
                    fill=True, alpha=0.15)
ax.set_xlabel("Available Days per Year")
ax.set_ylabel("Density")
ax.set_title("Availability Distribution by Room Type (KDE)")
ax.legend(fontsize=9)
ax.spines[["top", "right"]].set_visible(False)
save("07_availability_kde_roomtype.png")

# 8. Top 15 neighbourhoods by median price
print("8. Top 15 neighbourhoods by median price")
min_listings = 30
hood_price = (df.groupby("neighbourhood")["price"]
               .agg(["median", "count"])
               .query("count >= @min_listings")
               .sort_values("median", ascending=False)
               .head(15))
fig, ax = plt.subplots(figsize=(9, 6))
colors_h = [BOROUGH_COLORS.get(
    df[df["neighbourhood"] == h]["neighbourhood_group"].iloc[0], "#888")
    for h in hood_price.index]
bars = ax.barh(hood_price.index[::-1], hood_price["median"][::-1],
               color=colors_h[::-1], edgecolor="none")
for bar, val in zip(bars, hood_price["median"][::-1]):
    ax.text(val + 1, bar.get_y() + bar.get_height() / 2,
            f"${val:.0f}", va="center", fontsize=9)
ax.set_xlabel("Median Price (USD)")
ax.set_title(f"Top 15 Neighbourhoods by Median Price\n(min. {min_listings} listings)")
ax.spines[["top", "right"]].set_visible(False)
present = df[df["neighbourhood"].isin(hood_price.index)]["neighbourhood_group"].unique()
legend_patches = [plt.Rectangle((0,0),1,1, color=BOROUGH_COLORS.get(b,"#888")) for b in present]
ax.legend(legend_patches, present, fontsize=8, loc="lower right")
save("08_top15_neighbourhoods_price.png")

# 9. Price vs number_of_reviews scatter
print("9. Price vs reviews scatter (fixed)")
active_df = df[df["number_of_reviews"] > 0].copy()

rng = np.random.default_rng(42)
sample_parts = []
for room in ROOM_COLORS:
    sub = active_df[active_df["room_type"] == room]
    n = min(len(sub), 8000)
    sample_parts.append(sub.sample(n, random_state=42))
plot_df = pd.concat(sample_parts).reset_index(drop=True)

jitter_x = rng.uniform(-0.04, 0.04, size=len(plot_df))
jitter_y = rng.uniform(-0.02, 0.02, size=len(plot_df))
jittered_x = plot_df["number_of_reviews"].values * (10 ** jitter_x)
jittered_y = plot_df["price"].values * (10 ** jitter_y)

fig, ax = plt.subplots(figsize=(10, 6))
for room, color in ROOM_COLORS.items():
    mask = (plot_df["room_type"] == room).values
    ax.scatter(jittered_x[mask], jittered_y[mask],
               alpha=0.18, s=6, color=color, label=room, linewidths=0)
ax.set_xscale("log")
ax.set_yscale("log")
ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
ax.yaxis.set_major_formatter(mticker.ScalarFormatter())
ax.set_xlabel("Number of Reviews (log scale)")
ax.set_ylabel("Price USD (log scale)")
ax.set_title("Price vs Number of Reviews")
ax.legend(fontsize=9, markerscale=3)
ax.spines[["top", "right"]].set_visible(False)
save("09_price_vs_reviews_scatter.png")

# 10. Monthly review activity heatmap
print("10. Monthly review activity heatmap (fixed)")

rev_df = df[df["last_review"] != "No reviews"].copy()
rev_df["last_review"] = pd.to_datetime(rev_df["last_review"], errors="coerce")
rev_df = rev_df.dropna(subset=["last_review"])
rev_df = rev_df[rev_df["number_of_reviews"] > 0].copy()
rev_df["rpm"] = rev_df["reviews_per_month"].replace(0, 1.0)

monthly_counts = defaultdict(float)
for _, row in rev_df.iterrows():
    remaining = float(row["number_of_reviews"])
    rpm       = float(row["rpm"])
    period    = row["last_review"].to_period("M")
    while remaining > 0:
        alloc = min(rpm, remaining)
        monthly_counts[period] += alloc
        remaining -= alloc
        period -= 1

records = [(p.year, p.month, v) for p, v in monthly_counts.items()]
hm_df = pd.DataFrame(records, columns=["year", "month", "reviews"])
pivot = hm_df.pivot_table(index="year", columns="month",
                           values="reviews", aggfunc="sum", fill_value=0)
year_totals = pivot.sum(axis=1)
pivot = pivot[year_totals >= year_totals.max() * 0.01]
for m in range(1, 13):
    if m not in pivot.columns:
        pivot[m] = 0
pivot = pivot[[c for c in range(1, 13) if c in pivot.columns]]

month_labels = ["Jan","Feb","Mar","Apr","May","Jun",
                "Jul","Aug","Sep","Oct","Nov","Dec"]

fig, ax = plt.subplots(figsize=(14, max(4, len(pivot) * 0.6)))
sns.heatmap(pivot, cmap="YlOrRd", linewidths=0.4, ax=ax,
            cbar_kws={"label": "Estimated Reviews"},
            fmt=".0f", annot=(len(pivot) <= 6))
ax.set_xlabel("Month")
ax.set_ylabel("Year")
ax.set_title("Estimated Review Activity Heatmap (Year × Month)\n"
             "Reconstructed from reviews_per_month + last_review")
ax.set_xticklabels([month_labels[int(c)-1] for c in pivot.columns], rotation=0)
ax.set_yticklabels(pivot.index, rotation=0)
save("10_review_activity_heatmap.png")

# 11. Price CDF by room type
print("11. Price CDF by room type")
fig, ax = plt.subplots(figsize=(9, 5))
for room, color in ROOM_COLORS.items():
    prices = np.sort(df[df["room_type"] == room]["price"].values)
    cdf = np.arange(1, len(prices)+1) / len(prices)
    ax.plot(prices, cdf, label=room, color=color, linewidth=2)
ax.set_xlabel("Price (USD)")
ax.set_ylabel("Cumulative Fraction of Listings")
ax.set_title("Cumulative Distribution of Price by Room Type")
ax.legend(fontsize=9)
ax.spines[["top", "right"]].set_visible(False)
save("11_price_cdf_roomtype.png")

# 12. Geo scatter - listings density coloured by room type
print("12. Geo scatter by room type")
fig, ax = plt.subplots(figsize=(10, 8))
for room, color in ROOM_COLORS.items():
    sub = df[df["room_type"] == room]
    ax.scatter(sub["longitude"], sub["latitude"],
               c=color, alpha=0.25, s=3, label=room, rasterized=True)
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
ax.set_title("NYC Airbnb Listings by Room Type")
ax.legend(fontsize=9, markerscale=4, loc="upper left")
ax.spines[["top", "right"]].set_visible(False)
save("12_geo_scatter_roomtype.png")