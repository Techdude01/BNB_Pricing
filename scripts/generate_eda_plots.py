"""Generate curated EDA plots for the NYC Airbnb pricing project."""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CACHE_ROOT = Path(os.environ.get("TMPDIR", tempfile.gettempdir())) / "bnb_pricing_cache"
os.environ.setdefault("MPLCONFIGDIR", str(CACHE_ROOT / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(CACHE_ROOT / "xdg"))

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns

if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from pricing_lab import config
from pricing_lab.data import clean_listings_dataframe

DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "airbnb_plots"
BOROUGH_COLORS = {
    "Manhattan": "#E45C3A",
    "Brooklyn": "#3266AD",
    "Queens": "#F5A623",
    "Bronx": "#4CAF82",
    "Staten Island": "#9B59B6",
}
ROOM_COLORS = {
    "Entire home/apt": "#E45C3A",
    "Private room": "#3266AD",
    "Shared room": "#F5A623",
}


def _save_current_figure(output_dir: Path, filename: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / filename
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {output_path}")


def _load_plot_frame(csv_path: Path) -> pd.DataFrame:
    raw_frame = pd.read_csv(csv_path)
    cleaned_model_frame = clean_listings_dataframe(raw_frame)
    plot_frame = raw_frame.loc[cleaned_model_frame.index].copy()
    plot_frame["reviews_per_month"] = plot_frame["reviews_per_month"].fillna(0)
    plot_frame["last_review"] = plot_frame["last_review"].fillna("No reviews")
    return plot_frame


def _plot_top_hosts(frame: pd.DataFrame, output_dir: Path) -> None:
    print("1. Top-10 hosts by listings")
    top_hosts = frame["host_name"].value_counts().head(10)
    _, axis = plt.subplots(figsize=(9, 5))
    bars = axis.barh(top_hosts.index[::-1], top_hosts.values[::-1], color="#3266AD", edgecolor="none")
    for bar, value in zip(bars, top_hosts.values[::-1]):
        axis.text(value + 0.5, bar.get_y() + bar.get_height() / 2, str(value), va="center", fontsize=10)
    axis.set_xlabel("Number of Listings")
    axis.set_title("Top 10 Hosts by Number of Listings")
    axis.spines[["top", "right"]].set_visible(False)
    _save_current_figure(output_dir, "01_top10_hosts.png")


def _plot_borough_share(frame: pd.DataFrame, output_dir: Path) -> None:
    print("2. Listings share by borough")
    borough_counts = frame["neighbourhood_group"].value_counts()
    _, axis = plt.subplots(figsize=(7, 7))
    _, _, autotexts = axis.pie(
        borough_counts,
        labels=borough_counts.index,
        autopct="%1.1f%%",
        colors=[BOROUGH_COLORS.get(borough, "#888") for borough in borough_counts.index],
        startangle=140,
        pctdistance=0.82,
        wedgeprops={"edgecolor": "white", "linewidth": 2},
    )
    for text in autotexts:
        text.set_fontsize(10)
    axis.set_title("Share of Listings by Borough", pad=20)
    _save_current_figure(output_dir, "02_borough_pie.png")


def _plot_price_violin(frame: pd.DataFrame, output_dir: Path) -> pd.Index:
    print("3. Price violin by borough")
    borough_order = frame.groupby("neighbourhood_group")["price"].median().sort_values(ascending=False).index
    _, axis = plt.subplots(figsize=(10, 6))
    sns.violinplot(
        data=frame,
        x="neighbourhood_group",
        y="price",
        order=borough_order,
        palette=[BOROUGH_COLORS.get(borough, "#888") for borough in borough_order],
        inner="quartile",
        linewidth=1.2,
        ax=axis,
    )
    axis.set_xlabel("Borough")
    axis.set_ylabel("Price (USD)")
    axis.set_title("Price Distribution by Borough (Violin)")
    axis.spines[["top", "right"]].set_visible(False)
    _save_current_figure(output_dir, "03_price_violin_borough.png")
    return borough_order


def _plot_room_type_mix(frame: pd.DataFrame, borough_order: pd.Index, output_dir: Path) -> None:
    print("4. Room-type share by borough")
    room_borough = frame.groupby(["neighbourhood_group", "room_type"]).size().unstack(fill_value=0)
    room_percent = room_borough.div(room_borough.sum(axis=1), axis=0) * 100
    room_percent = room_percent.loc[borough_order]
    _, axis = plt.subplots(figsize=(10, 6))
    bottom = np.zeros(len(room_percent))
    for room_type, color in ROOM_COLORS.items():
        if room_type not in room_percent.columns:
            continue
        axis.bar(
            room_percent.index,
            room_percent[room_type],
            bottom=bottom,
            label=room_type,
            color=color,
            edgecolor="white",
            linewidth=0.8,
        )
        bottom += room_percent[room_type].values
    axis.set_ylabel("% of Listings")
    axis.set_title("Room Type Mix by Borough (100% Stacked)")
    axis.legend(loc="lower right", fontsize=9)
    axis.spines[["top", "right"]].set_visible(False)
    _save_current_figure(output_dir, "04_roomtype_stacked_borough.png")


def _plot_reviews_per_month(frame: pd.DataFrame, output_dir: Path) -> None:
    print("5. Reviews per month distribution")
    active_reviews = frame[frame["reviews_per_month"] > 0]["reviews_per_month"]
    log_bins = np.logspace(np.log10(active_reviews.min()), np.log10(active_reviews.max()), 45)
    _, axis = plt.subplots(figsize=(9, 5))
    axis.hist(active_reviews, bins=log_bins, color="#E45C3A", edgecolor="white", linewidth=0.3, alpha=0.85)
    axis.set_xscale("log")
    axis.xaxis.set_major_formatter(mticker.ScalarFormatter())
    axis.set_xlabel("Reviews per Month (log scale)")
    axis.set_ylabel("Number of Listings")
    axis.set_title("Distribution of Reviews per Month (Active Listings)")
    axis.spines[["top", "right"]].set_visible(False)
    median_reviews = active_reviews.median()
    axis.axvline(median_reviews, color="#333", linestyle="--", linewidth=1.2)
    axis.text(median_reviews * 1.12, axis.get_ylim()[1] * 0.93, f"Median: {median_reviews:.2f}", fontsize=9)
    _save_current_figure(output_dir, "05_reviews_per_month_dist.png")


def _plot_minimum_nights(frame: pd.DataFrame, output_dir: Path) -> None:
    print("6. Minimum nights distribution")
    minimum_nights = frame[frame["minimum_nights"] <= 30]["minimum_nights"]
    counts = minimum_nights.value_counts().sort_index()
    _, axis = plt.subplots(figsize=(10, 5))
    axis.bar(counts.index, counts.values, color="#3266AD", edgecolor="none", width=0.8)
    axis.set_xlabel("Minimum Nights Required")
    axis.set_ylabel("Number of Listings")
    axis.set_title("Minimum-Nights Distribution (capped at 30)")
    axis.spines[["top", "right"]].set_visible(False)
    _save_current_figure(output_dir, "06_min_nights_dist.png")


def _plot_availability(frame: pd.DataFrame, output_dir: Path) -> None:
    print("7. Availability KDE by room type")
    _, axis = plt.subplots(figsize=(9, 5))
    for room_type, color in ROOM_COLORS.items():
        subset = frame[frame["room_type"] == room_type]["availability_365"]
        if len(subset) <= 10:
            continue
        sns.kdeplot(subset, ax=axis, label=room_type, color=color, linewidth=2, fill=True, alpha=0.15)
    axis.set_xlabel("Available Days per Year")
    axis.set_ylabel("Density")
    axis.set_title("Availability Distribution by Room Type (KDE)")
    axis.legend(fontsize=9)
    axis.spines[["top", "right"]].set_visible(False)
    _save_current_figure(output_dir, "07_availability_kde_roomtype.png")


def _plot_top_neighbourhood_prices(frame: pd.DataFrame, output_dir: Path) -> None:
    print("8. Top 15 neighbourhoods by median price")
    min_listings = 30
    neighbourhood_price = (
        frame.groupby("neighbourhood")["price"]
        .agg(["median", "count"])
        .query("count >= @min_listings")
        .sort_values("median", ascending=False)
        .head(15)
    )
    colors = [
        BOROUGH_COLORS.get(frame[frame["neighbourhood"] == neighbourhood]["neighbourhood_group"].iloc[0], "#888")
        for neighbourhood in neighbourhood_price.index
    ]
    _, axis = plt.subplots(figsize=(9, 6))
    bars = axis.barh(
        neighbourhood_price.index[::-1],
        neighbourhood_price["median"][::-1],
        color=colors[::-1],
        edgecolor="none",
    )
    for bar, value in zip(bars, neighbourhood_price["median"][::-1]):
        axis.text(value + 1, bar.get_y() + bar.get_height() / 2, f"${value:.0f}", va="center", fontsize=9)
    axis.set_xlabel("Median Price (USD)")
    axis.set_title(f"Top 15 Neighbourhoods by Median Price\n(min. {min_listings} listings)")
    axis.spines[["top", "right"]].set_visible(False)
    present_boroughs = frame[frame["neighbourhood"].isin(neighbourhood_price.index)]["neighbourhood_group"].unique()
    legend_patches = [plt.Rectangle((0, 0), 1, 1, color=BOROUGH_COLORS.get(borough, "#888")) for borough in present_boroughs]
    axis.legend(legend_patches, present_boroughs, fontsize=8, loc="lower right")
    _save_current_figure(output_dir, "08_top15_neighbourhoods_price.png")


def _plot_price_vs_reviews(frame: pd.DataFrame, output_dir: Path) -> None:
    print("9. Price vs reviews scatter")
    active_frame = frame[frame["number_of_reviews"] > 0].copy()
    sample_parts = []
    for room_type in ROOM_COLORS:
        subset = active_frame[active_frame["room_type"] == room_type]
        sample_parts.append(subset.sample(min(len(subset), 8000), random_state=42))
    plot_frame = pd.concat(sample_parts).reset_index(drop=True)

    rng = np.random.default_rng(42)
    jitter_x = rng.uniform(-0.04, 0.04, size=len(plot_frame))
    jitter_y = rng.uniform(-0.02, 0.02, size=len(plot_frame))
    jittered_x = plot_frame["number_of_reviews"].to_numpy() * (10**jitter_x)
    jittered_y = plot_frame["price"].to_numpy() * (10**jitter_y)

    _, axis = plt.subplots(figsize=(10, 6))
    for room_type, color in ROOM_COLORS.items():
        mask = (plot_frame["room_type"] == room_type).to_numpy()
        axis.scatter(jittered_x[mask], jittered_y[mask], alpha=0.18, s=6, color=color, label=room_type, linewidths=0)
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.xaxis.set_major_formatter(mticker.ScalarFormatter())
    axis.yaxis.set_major_formatter(mticker.ScalarFormatter())
    axis.set_xlabel("Number of Reviews (log scale)")
    axis.set_ylabel("Price USD (log scale)")
    axis.set_title("Price vs Number of Reviews")
    axis.legend(fontsize=9, markerscale=3)
    axis.spines[["top", "right"]].set_visible(False)
    _save_current_figure(output_dir, "09_price_vs_reviews_scatter.png")


def _plot_review_activity_heatmap(frame: pd.DataFrame, output_dir: Path) -> None:
    print("10. Monthly review activity heatmap")
    review_frame = frame[frame["last_review"] != "No reviews"].copy()
    review_frame["last_review"] = pd.to_datetime(review_frame["last_review"], errors="coerce")
    review_frame = review_frame.dropna(subset=["last_review"])
    review_frame = review_frame[review_frame["number_of_reviews"] > 0].copy()
    review_frame["rpm"] = review_frame["reviews_per_month"].replace(0, 1.0)

    # Approximate monthly review activity by walking backward from each listing's last review date.
    monthly_counts: defaultdict[pd.Period, float] = defaultdict(float)
    for _, row in review_frame.iterrows():
        remaining = float(row["number_of_reviews"])
        reviews_per_month = float(row["rpm"])
        period = row["last_review"].to_period("M")
        while remaining > 0:
            allocation = min(reviews_per_month, remaining)
            monthly_counts[period] += allocation
            remaining -= allocation
            period -= 1

    records = [(period.year, period.month, reviews) for period, reviews in monthly_counts.items()]
    heatmap_frame = pd.DataFrame(records, columns=["year", "month", "reviews"])
    pivot = heatmap_frame.pivot_table(index="year", columns="month", values="reviews", aggfunc="sum", fill_value=0)
    year_totals = pivot.sum(axis=1)
    pivot = pivot[year_totals >= year_totals.max() * 0.01]
    for month in range(1, 13):
        if month not in pivot.columns:
            pivot[month] = 0
    pivot = pivot[[month for month in range(1, 13) if month in pivot.columns]]

    month_labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    _, axis = plt.subplots(figsize=(14, max(4, len(pivot) * 0.6)))
    sns.heatmap(
        pivot,
        cmap="YlOrRd",
        linewidths=0.4,
        ax=axis,
        cbar_kws={"label": "Estimated Reviews"},
        fmt=".0f",
        annot=(len(pivot) <= 6),
    )
    axis.set_xlabel("Month")
    axis.set_ylabel("Year")
    axis.set_title("Estimated Review Activity Heatmap (Year x Month)\nReconstructed from reviews_per_month + last_review")
    axis.set_xticklabels([month_labels[int(column) - 1] for column in pivot.columns], rotation=0)
    axis.set_yticklabels(pivot.index, rotation=0)
    _save_current_figure(output_dir, "10_review_activity_heatmap.png")


def _plot_price_cdf(frame: pd.DataFrame, output_dir: Path) -> None:
    print("11. Price CDF by room type")
    _, axis = plt.subplots(figsize=(9, 5))
    for room_type, color in ROOM_COLORS.items():
        prices = np.sort(frame[frame["room_type"] == room_type]["price"].to_numpy())
        cdf = np.arange(1, len(prices) + 1) / len(prices)
        axis.plot(prices, cdf, label=room_type, color=color, linewidth=2)
    axis.set_xlabel("Price (USD)")
    axis.set_ylabel("Cumulative Fraction of Listings")
    axis.set_title("Cumulative Distribution of Price by Room Type")
    axis.legend(fontsize=9)
    axis.spines[["top", "right"]].set_visible(False)
    _save_current_figure(output_dir, "11_price_cdf_roomtype.png")


def _plot_geo_scatter(frame: pd.DataFrame, output_dir: Path) -> None:
    print("12. Geo scatter by room type")
    _, axis = plt.subplots(figsize=(10, 8))
    for room_type, color in ROOM_COLORS.items():
        subset = frame[frame["room_type"] == room_type]
        axis.scatter(
            subset["longitude"],
            subset["latitude"],
            c=color,
            alpha=0.25,
            s=3,
            label=room_type,
            rasterized=True,
        )
    axis.set_xlabel("Longitude")
    axis.set_ylabel("Latitude")
    axis.set_title("NYC Airbnb Listings by Room Type")
    axis.legend(fontsize=9, markerscale=4, loc="upper left")
    axis.spines[["top", "right"]].set_visible(False)
    _save_current_figure(output_dir, "12_geo_scatter_roomtype.png")


def generate_plots(frame: pd.DataFrame, output_dir: Path) -> None:
    print(f"\n{'=' * 55}")
    print("  NYC Airbnb 2019 - Extended EDA")
    print(f"  Dataset: {len(frame):,} listings after shared cleaning")
    print(f"{'=' * 55}\n")
    _plot_top_hosts(frame, output_dir)
    _plot_borough_share(frame, output_dir)
    borough_order = _plot_price_violin(frame, output_dir)
    _plot_room_type_mix(frame, borough_order, output_dir)
    _plot_reviews_per_month(frame, output_dir)
    _plot_minimum_nights(frame, output_dir)
    _plot_availability(frame, output_dir)
    _plot_top_neighbourhood_prices(frame, output_dir)
    _plot_price_vs_reviews(frame, output_dir)
    _plot_review_activity_heatmap(frame, output_dir)
    _plot_price_cdf(frame, output_dir)
    _plot_geo_scatter(frame, output_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Regenerate the curated Airbnb EDA plot images.")
    parser.add_argument("--csv", type=Path, default=config.DATA_PATH, help="Path to AB_NYC_2019.csv.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directory for generated PNG files.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frame = _load_plot_frame(args.csv)
    generate_plots(frame, args.output_dir)


if __name__ == "__main__":
    main()
