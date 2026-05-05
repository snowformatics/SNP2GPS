import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

import folium
from folium.plugins import HeatMap, MarkerCluster, Fullscreen


def haversine(lat1, lon1, lat2, lon2):
    r = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return r * c


def _save_close(output_path):
    plt.savefig(output_path, bbox_inches="tight", dpi=300)
    plt.close()


def _safe_map(center, zoom=3):
    return folium.Map(location=center, zoom_start=zoom, tiles="CartoDB positron")


def read_performance_file(performance_file):
    metrics = {
        "r2_longitude": np.nan,
        "r2_latitude": np.nan,
        "mean_validation_error_km": np.nan,
        "median_validation_error_km": np.nan,
    }

    if not os.path.exists(performance_file):
        return metrics

    encodings_to_try = ["utf-8", "cp1252", "latin-1"]
    lines = None

    for enc in encodings_to_try:
        try:
            with open(performance_file, "r", encoding=enc) as f:
                lines = f.readlines()
            break
        except UnicodeDecodeError:
            continue

    if lines is None:
        raise UnicodeDecodeError(
            "read_performance_file",
            b"",
            0,
            1,
            f"Could not decode file: {performance_file}",
        )

    for line in lines:
        line = line.strip()

        if line.startswith("R² (Longitude):") or line.startswith("R2 (Longitude):"):
            metrics["r2_longitude"] = float(line.split(":", 1)[1].strip())
        elif line.startswith("R² (Latitude):") or line.startswith("R2 (Latitude):"):
            metrics["r2_latitude"] = float(line.split(":", 1)[1].strip())
        elif line.startswith("Mean Validation Error:"):
            metrics["mean_validation_error_km"] = float(line.split(":", 1)[1].replace("km", "").strip())
        elif line.startswith("Median Validation Error:"):
            metrics["median_validation_error_km"] = float(line.split(":", 1)[1].replace("km", "").strip())

    return metrics


def collect_training_histories(run_folder):
    history_files = sorted(Path(run_folder).glob("*_training_history.txt"))
    histories = []

    for fp in history_files:
        try:
            df = pd.read_csv(fp, sep="\t")
            if {"loss", "val_loss"}.issubset(df.columns):
                df = df.copy()
                df["epoch"] = np.arange(1, len(df) + 1)
                df["source_file"] = fp.name
                histories.append(df)
        except Exception as e:
            print(f"Could not read training history {fp}: {e}")

    return histories


def plot_best_training_history(histories, output_png):
    if not histories:
        return None

    best_idx = None
    best_val = np.inf
    for i, df in enumerate(histories):
        min_val = df["val_loss"].min()
        if min_val < best_val:
            best_val = min_val
            best_idx = i

    best_df = histories[best_idx]
    best_epoch = int(best_df["val_loss"].idxmin()) + 1

    plt.figure(figsize=(7, 4.5))
    plt.plot(best_df["epoch"], best_df["loss"], label="Train loss")
    plt.plot(best_df["epoch"], best_df["val_loss"], label="Validation loss")
    plt.axvline(best_epoch, linestyle="--", color="gray", label=f"Best epoch ({best_epoch})")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training History (Best Run)")
    plt.grid(alpha=0.3)
    plt.legend()
    _save_close(output_png)

    summary = {
        "history_plot": os.path.basename(output_png),
        "best_history_file": best_df["source_file"].iloc[0],
        "best_epoch": best_epoch,
        "best_val_loss": float(best_df["val_loss"].min()),
        "final_train_loss": float(best_df["loss"].iloc[-1]),
        "final_val_loss": float(best_df["val_loss"].iloc[-1]),
    }

    if "lr" in best_df.columns:
        lr_png = os.path.splitext(output_png)[0] + "_lr.png"
        plt.figure(figsize=(7, 3.5))
        plt.plot(best_df["epoch"], best_df["lr"])
        plt.xlabel("Epoch")
        plt.ylabel("Learning rate")
        plt.title("Learning Rate Schedule (Best Run)")
        plt.grid(alpha=0.3)
        _save_close(lr_png)
        summary["lr_plot"] = os.path.basename(lr_png)

    return summary


def load_validation_table(run_folder):
    run_folder = Path(run_folder)
    files = sorted(run_folder.glob("validation_set_run_*.csv"))

    if not files:
        print("No validation_set_run_*.csv found. Validation-only plots will be skipped.")
        return None

    dfs = []
    for fp in files:
        try:
            df = pd.read_csv(fp)
            df["source_validation_file"] = fp.name

            if "latitude" in df.columns and "longitude" in df.columns:
                df = df.rename(columns={
                    "latitude": "latitude_true",
                    "longitude": "longitude_true",
                })

            if "latitude_land" in df.columns and "longitude_land" in df.columns:
                df["latitude_plot"] = df["latitude_land"].where(
                    df["latitude_land"].notna(), df["latitude_centroid"]
                )
                df["longitude_plot"] = df["longitude_land"].where(
                    df["longitude_land"].notna(), df["longitude_centroid"]
                )
            else:
                df["latitude_plot"] = df["latitude_centroid"]
                df["longitude_plot"] = df["longitude_centroid"]

            if "distance_km" not in df.columns:
                df["distance_km"] = df.apply(
                    lambda r: haversine(
                        r["latitude_true"], r["longitude_true"],
                        r["latitude_plot"], r["longitude_plot"]
                    )
                    if pd.notna(r["latitude_true"]) and pd.notna(r["longitude_true"])
                    and pd.notna(r["latitude_plot"]) and pd.notna(r["longitude_plot"])
                    else np.nan,
                    axis=1,
                )

            dfs.append(df)
        except Exception as e:
            print(f"Could not read validation file {fp}: {e}")

    if not dfs:
        return None

    result = pd.concat(dfs, ignore_index=True)
    print(f"Loaded {len(result)} validation rows from {len(dfs)} validation file(s).")
    return result


def plot_heatmap(df, lat_col, lon_col, output_html, title):
    tmp = df.copy()
    tmp[lat_col] = pd.to_numeric(tmp[lat_col], errors="coerce")
    tmp[lon_col] = pd.to_numeric(tmp[lon_col], errors="coerce")
    tmp = tmp.dropna(subset=[lat_col, lon_col])

    if tmp.empty:
        return None

    center = [tmp[lat_col].mean(), tmp[lon_col].mean()]
    m = _safe_map(center, zoom=3)

    heat_data = tmp[[lat_col, lon_col]].values.tolist()
    HeatMap(heat_data, radius=25, blur=20, min_opacity=0.3).add_to(m)

    for _, row in tmp.iterrows():
        tooltip = str(row["sampleID"]) if "sampleID" in tmp.columns else None
        popup = None
        if "sampleID" in tmp.columns or "country" in tmp.columns:
            sid = row["sampleID"] if "sampleID" in tmp.columns else ""
            cty = row["country"] if "country" in tmp.columns else ""
            popup = f"{sid}<br>{cty}<br>lat={row[lat_col]:.4f}, lon={row[lon_col]:.4f}"

        folium.CircleMarker(
            location=[row[lat_col], row[lon_col]],
            radius=2,
            tooltip=tooltip,
            popup=popup,
            fill=True,
            fill_opacity=0.6,
        ).add_to(m)

    m.fit_bounds(tmp[[lat_col, lon_col]].values.tolist())

    title_html = f"""
    <div style="
        position: fixed;
        top: 10px;
        left: 50px;
        z-index:9999;
        background: white;
        padding: 8px 12px;
        border: 1px solid #999;
        border-radius: 6px;">
        <b>{title}</b>
    </div>
    """
    m.get_root().html.add_child(folium.Element(title_html))
    m.save(output_html)
    return os.path.basename(output_html)


def plot_prediction_map(pred_df, output_html):
    df = pred_df.copy()

    if "latitude_land" in df.columns and "longitude_land" in df.columns:
        df["plot_latitude"] = df["latitude_land"].where(df["latitude_land"].notna(), df["latitude_centroid"])
        df["plot_longitude"] = df["longitude_land"].where(df["longitude_land"].notna(), df["longitude_centroid"])
    else:
        df["plot_latitude"] = df["latitude_centroid"]
        df["plot_longitude"] = df["longitude_centroid"]

    df = df.dropna(subset=["plot_latitude", "plot_longitude"])
    if df.empty:
        return None

    center = [df["plot_latitude"].mean(), df["plot_longitude"].mean()]
    m = _safe_map(center, zoom=3)
    cluster = MarkerCluster().add_to(m)

    df["combined_sd"] = (df["latitude_sd"].fillna(0) + df["longitude_sd"].fillna(0)) / 2

    for _, row in df.iterrows():
        sd = row["combined_sd"]
        if sd <= 3:
            color = "green"
        elif sd >= 8:
            color = "red"
        else:
            color = "orange"

        popup = (
            f"<b>{row['sampleID']}</b><br>"
            f"Mean: {row['latitude_mean']:.4f}, {row['longitude_mean']:.4f}<br>"
            f"Centroid: {row['latitude_centroid']:.4f}, {row['longitude_centroid']:.4f}<br>"
            f"Land: {row['plot_latitude']:.4f}, {row['plot_longitude']:.4f}<br>"
            f"Country: {row.get('country', 'NA')}<br>"
            f"SD: {sd:.4f}<br>"
            f"Corrected: {row.get('was_corrected_to_land', 'NA')}<br>"
            f"Note: {row.get('correction_note', 'NA')}<br>"
            f"Land shift: {row.get('land_shift_km', np.nan):.2f} km"
        )

        folium.Marker(
            location=[row["plot_latitude"], row["plot_longitude"]],
            icon=folium.Icon(color=color),
            popup=folium.Popup(popup, max_width=320),
        ).add_to(cluster)

    m.save(output_html)
    return os.path.basename(output_html)


def plot_validation_map(validation_df, output_html):
    df = validation_df.dropna(
        subset=["latitude_true", "longitude_true", "latitude_plot", "longitude_plot", "distance_km"]
    ).copy()

    if df.empty:
        return None

    center = [df["latitude_plot"].mean(), df["longitude_plot"].mean()]
    m = _safe_map(center, zoom=4)
    Fullscreen(position="topright").add_to(m)

    for _, row in df.iterrows():
        if row["distance_km"] <= 100:
            color = "green"
        elif row["distance_km"] <= 300:
            color = "orange"
        elif row["distance_km"] >= 1000:
            color = "red"
        else:
            color = "black"

        popup = (
            f"<b>{row['sampleID']}</b><br>"
            f"True: {row['latitude_true']:.4f}, {row['longitude_true']:.4f}<br>"
            f"Predicted: {row['latitude_plot']:.4f}, {row['longitude_plot']:.4f}<br>"
            f"Distance: {row['distance_km']:.2f} km<br>"
            f"Correction: {row.get('correction_note', 'NA')}"
        )

        folium.CircleMarker(
            location=[row["latitude_true"], row["longitude_true"]],
            radius=4,
            color="blue",
            fill=True,
            fill_opacity=0.9,
            popup=f"True location: {row['sampleID']}",
        ).add_to(m)

        folium.Marker(
            location=[row["latitude_plot"], row["longitude_plot"]],
            icon=folium.Icon(color=color),
            popup=popup,
        ).add_to(m)

        folium.PolyLine(
            locations=[
                [row["latitude_true"], row["longitude_true"]],
                [row["latitude_plot"], row["longitude_plot"]],
            ],
            color="gray",
            weight=1,
            opacity=0.6,
        ).add_to(m)

    legend_html = """
    <div style="
        position: fixed;
        bottom: 50px;
        left: 50px;
        width: 220px;
        height: 145px;
        background-color: white;
        border:2px solid grey;
        z-index:9999;
        font-size:14px;
        padding: 10px;
        border-radius: 8px;">
    <b>Validation Error</b><br>
    <i style="background:green;width:10px;height:10px;display:inline-block;margin-right:8px;"></i> ≤ 100 km<br>
    <i style="background:orange;width:10px;height:10px;display:inline-block;margin-right:8px;"></i> 101–300 km<br>
    <i style="background:black;width:10px;height:10px;display:inline-block;margin-right:8px;"></i> 301–999 km<br>
    <i style="background:red;width:10px;height:10px;display:inline-block;margin-right:8px;"></i> ≥ 1000 km<br>
    <i style="background:blue;width:10px;height:10px;display:inline-block;margin-right:8px;"></i> True location
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))
    m.save(output_html)
    return os.path.basename(output_html)


def plot_error_histogram(validation_df, output_png):
    tmp = validation_df.dropna(subset=["distance_km"]).copy()
    if tmp.empty:
        return None

    plt.figure(figsize=(7, 4.5))
    plt.hist(tmp["distance_km"], bins=max(5, min(20, len(tmp))), edgecolor="black")
    plt.xlabel("Prediction error (km)")
    plt.ylabel("Count")
    plt.title("Validation Error Distribution")
    plt.grid(alpha=0.3)
    _save_close(output_png)
    return os.path.basename(output_png)


def plot_cumulative_error_curve(validation_df, output_png):
    tmp = validation_df.dropna(subset=["distance_km"]).copy()
    if tmp.empty:
        return None

    x = np.sort(tmp["distance_km"].values)
    y = np.arange(1, len(x) + 1) / len(x) * 100

    plt.figure(figsize=(7, 4.5))
    plt.plot(x, y)
    for thresh in [50, 100, 250, 500, 1000]:
        plt.axvline(thresh, linestyle="--", alpha=0.4)
    plt.xlabel("Prediction error threshold (km)")
    plt.ylabel("Samples within threshold (%)")
    plt.title("Cumulative Validation Error")
    plt.grid(alpha=0.3)
    _save_close(output_png)
    return os.path.basename(output_png)


def plot_distance_barchart_report(validation_df, output_png):
    tmp = validation_df.dropna(subset=["distance_km"]).copy()
    if tmp.empty:
        return None

    tmp = tmp.sort_values("distance_km")
    colors = tmp["distance_km"].apply(
        lambda x: "green" if x < 100 else ("orange" if x <= 1000 else "red")
    )

    if "was_corrected_to_land" in tmp.columns:
        labels = [
            f"{sid}*" if bool(corr) else str(sid)
            for sid, corr in zip(tmp["sampleID"], tmp["was_corrected_to_land"])
        ]
    else:
        labels = tmp["sampleID"].astype(str).tolist()

    plt.figure(figsize=(12, 5.5))
    plt.bar(labels, tmp["distance_km"], color=colors, edgecolor="black")
    plt.title("Distance between True GPS and Predicted Land-Corrected Position")
    plt.ylabel("Distance (km)")
    plt.xlabel("Validation sample ID (* = corrected to land)")
    plt.xticks(rotation=90, fontsize=8)
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.legend(handles=[
        Patch(color="green", label="< 100 km"),
        Patch(color="orange", label="100–1000 km"),
        Patch(color="red", label="> 1000 km"),
    ])
    _save_close(output_png)
    return os.path.basename(output_png)


def plot_land_shift_histogram(pred_df, output_png):
    if "land_shift_km" not in pred_df.columns:
        return None

    tmp = pred_df.dropna(subset=["land_shift_km"]).copy()
    if tmp.empty:
        return None

    plt.figure(figsize=(7, 4.5))
    plt.hist(tmp["land_shift_km"], bins=max(5, min(20, len(tmp))), edgecolor="black")
    plt.xlabel("Land correction shift (km)")
    plt.ylabel("Count")
    plt.title("Distribution of Land Correction Shift")
    plt.grid(alpha=0.3)
    _save_close(output_png)
    return os.path.basename(output_png)


def plot_land_correction_summary(pred_df, output_png):
    if "was_corrected_to_land" not in pred_df.columns:
        return None

    counts = pred_df["was_corrected_to_land"].fillna(False).value_counts()
    labels = ["On land", "Corrected to land"]
    values = [int(counts.get(False, 0)), int(counts.get(True, 0))]

    plt.figure(figsize=(5, 4.5))
    plt.bar(labels, values)
    plt.ylabel("Count")
    plt.title("Land Correction Summary")
    _save_close(output_png)
    return os.path.basename(output_png)


def plot_country_summary(pred_df, output_png, top_n=20):
    if "country" not in pred_df.columns:
        return None

    tmp = pred_df["country"].fillna("Unknown").value_counts().head(top_n)
    if tmp.empty:
        return None

    plt.figure(figsize=(8, 5.5))
    plt.barh(tmp.index[::-1], tmp.values[::-1])
    plt.xlabel("Predicted samples")
    plt.title(f"Top {top_n} Predicted Countries")
    _save_close(output_png)
    return os.path.basename(output_png)


def plot_lat_lon_scatter(validation_df, lat_png, lon_png):
    tmp = validation_df.dropna(
        subset=["latitude_true", "longitude_true", "latitude_plot", "longitude_plot"]
    ).copy()
    if tmp.empty:
        return None, None

    plt.figure(figsize=(5.5, 5))
    plt.scatter(tmp["latitude_true"], tmp["latitude_plot"], s=20, alpha=0.7)
    mn = min(tmp["latitude_true"].min(), tmp["latitude_plot"].min())
    mx = max(tmp["latitude_true"].max(), tmp["latitude_plot"].max())
    plt.plot([mn, mx], [mn, mx], linestyle="--")
    plt.xlabel("True latitude")
    plt.ylabel("Predicted latitude")
    plt.title("Validation set: true vs predicted latitude")
    plt.grid(alpha=0.3)
    _save_close(lat_png)

    plt.figure(figsize=(5.5, 5))
    plt.scatter(tmp["longitude_true"], tmp["longitude_plot"], s=20, alpha=0.7)
    mn = min(tmp["longitude_true"].min(), tmp["longitude_plot"].min())
    mx = max(tmp["longitude_true"].max(), tmp["longitude_plot"].max())
    plt.plot([mn, mx], [mn, mx], linestyle="--")
    plt.xlabel("True longitude")
    plt.ylabel("Predicted longitude")
    plt.title("Validation set: true vs predicted longitude")
    plt.grid(alpha=0.3)
    _save_close(lon_png)

    return os.path.basename(lat_png), os.path.basename(lon_png)


def plot_all_predictions_vs_metadata_scatter(pred_df, metadata_df, lat_png, lon_png):
    pred = pred_df.copy()
    meta = metadata_df.copy()

    if "sampleID" not in pred.columns or "sampleID" not in meta.columns:
        return None, None

    pred["sampleID"] = pred["sampleID"].astype(str)
    meta["sampleID"] = meta["sampleID"].astype(str)

    if "latitude_land" in pred.columns and "longitude_land" in pred.columns:
        pred["latitude_plot"] = pred["latitude_land"].where(
            pred["latitude_land"].notna(), pred["latitude_centroid"]
        )
        pred["longitude_plot"] = pred["longitude_land"].where(
            pred["longitude_land"].notna(), pred["longitude_centroid"]
        )
    else:
        pred["latitude_plot"] = pred["latitude_centroid"]
        pred["longitude_plot"] = pred["longitude_centroid"]

    if not {"x", "y"}.issubset(meta.columns):
        return None, None

    meta = meta.rename(columns={"x": "latitude_true", "y": "longitude_true"})
    meta["latitude_true"] = pd.to_numeric(meta["latitude_true"], errors="coerce")
    meta["longitude_true"] = pd.to_numeric(meta["longitude_true"], errors="coerce")

    merged = pd.merge(
        meta[["sampleID", "latitude_true", "longitude_true"]],
        pred[["sampleID", "latitude_plot", "longitude_plot"]],
        on="sampleID",
        how="inner",
    ).dropna(subset=["latitude_true", "longitude_true", "latitude_plot", "longitude_plot"])

    if merged.empty:
        return None, None

    plt.figure(figsize=(5.5, 5))
    plt.scatter(merged["latitude_true"], merged["latitude_plot"], s=14, alpha=0.6)
    mn = min(merged["latitude_true"].min(), merged["latitude_plot"].min())
    mx = max(merged["latitude_true"].max(), merged["latitude_plot"].max())
    plt.plot([mn, mx], [mn, mx], linestyle="--")
    plt.xlabel("Metadata latitude")
    plt.ylabel("Predicted latitude")
    plt.title("All matched samples: metadata vs predicted latitude")
    plt.grid(alpha=0.3)
    _save_close(lat_png)

    plt.figure(figsize=(5.5, 5))
    plt.scatter(merged["longitude_true"], merged["longitude_plot"], s=14, alpha=0.6)
    mn = min(merged["longitude_true"].min(), merged["longitude_plot"].min())
    mx = max(merged["longitude_true"].max(), merged["longitude_plot"].max())
    plt.plot([mn, mx], [mn, mx], linestyle="--")
    plt.xlabel("Metadata longitude")
    plt.ylabel("Predicted longitude")
    plt.title("All matched samples: metadata vs predicted longitude")
    plt.grid(alpha=0.3)
    _save_close(lon_png)

    return os.path.basename(lat_png), os.path.basename(lon_png)


def render_html_report(output_path, metrics, artifacts, pred_df, validation_df=None, training_summary=None):
    def card(title, value):
        if isinstance(value, float) and pd.isna(value):
            value = "NA"
        return f'<div class="card"><div class="label">{title}</div><div class="value">{value}</div></div>'

    def image_section(title, filename):
        if not filename:
            return ""
        return f'<section><h3>{title}</h3><img src="{filename}" class="plot"></section>'

    def iframe_section(title, filename, height=540):
        if not filename:
            return ""
        return f'<section><h3>{title}</h3><iframe src="{filename}" width="100%" height="{height}" class="mapframe"></iframe></section>'

    corrected_count = int(pred_df["was_corrected_to_land"].fillna(False).sum()) if "was_corrected_to_land" in pred_df.columns else 0
    total_predictions = len(pred_df)
    validation_count = len(validation_df) if validation_df is not None else 0

    if validation_df is not None and not validation_df.empty:
        val_preview = validation_df.sort_values("distance_km", na_position="last").head(200).to_html(index=False, classes="datatable")
    else:
        val_preview = "<p>No validation comparison table available.</p>"

    training_block = ""
    if training_summary:
        training_block = f"""
        <section>
          <h2>Training</h2>
          <div class="cards">
            {card('Best history file', training_summary.get('best_history_file'))}
            {card('Best epoch', training_summary.get('best_epoch'))}
            {card('Best validation loss', f"{training_summary.get('best_val_loss', np.nan):.4f}" if pd.notna(training_summary.get('best_val_loss', np.nan)) else 'NA')}
            {card('Final training loss', f"{training_summary.get('final_train_loss', np.nan):.4f}" if pd.notna(training_summary.get('final_train_loss', np.nan)) else 'NA')}
            {card('Final validation loss', f"{training_summary.get('final_val_loss', np.nan):.4f}" if pd.notna(training_summary.get('final_val_loss', np.nan)) else 'NA')}
          </div>
        </section>
        {image_section('Training history (best run)', training_summary.get('history_plot'))}
        {image_section('Learning rate schedule (best run)', training_summary.get('lr_plot'))}
        """

    html = f"""
<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>SNP2GPS Report</title>
<style>
body {{ font-family: Arial, sans-serif; margin: 24px; color: #222; }}
h1, h2, h3 {{ margin-top: 1.2em; }}
.cards {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 14px; margin: 16px 0; }}
.card {{ border: 1px solid #ddd; border-radius: 12px; padding: 14px; background: #fafafa; }}
.label {{ font-size: 12px; color: #666; margin-bottom: 8px; text-transform: uppercase; }}
.value {{ font-size: 22px; font-weight: bold; }}
.plot {{ width: 100%; max-width: 900px; border: 1px solid #ddd; border-radius: 12px; }}
.mapframe {{ border: 1px solid #ddd; border-radius: 12px; }}
.datatable {{ border-collapse: collapse; width: 100%; font-size: 12px; }}
.datatable td, .datatable th {{ border: 1px solid #ddd; padding: 6px; }}
.datatable th {{ background: #f2f2f2; }}
.small {{ color: #555; font-size: 13px; }}
</style>
</head>
<body>
<h1>SNP2GPS HTML Report</h1>
<p class="small">Automatically generated summary of model performance, geography, validation errors, land correction, and training history.</p>

<h2>Overview</h2>
<div class="cards">
  {card('R² longitude', f"{metrics['r2_longitude']:.4f}" if pd.notna(metrics['r2_longitude']) else 'NA')}
  {card('R² latitude', f"{metrics['r2_latitude']:.4f}" if pd.notna(metrics['r2_latitude']) else 'NA')}
  {card('Mean validation error', f"{metrics['mean_validation_error_km']:.2f} km" if pd.notna(metrics['mean_validation_error_km']) else 'NA')}
  {card('Median validation error', f"{metrics['median_validation_error_km']:.2f} km" if pd.notna(metrics['median_validation_error_km']) else 'NA')}
  {card('Predicted samples', total_predictions)}
  {card('Validation samples', validation_count)}
  {card('Corrected to land', corrected_count)}
</div>

{training_block}

<h2>All Predicted Samples (train/test + unknown predictions)</h2>
<p class="small">These plots summarize the complete prediction table from <code>final_averaged_predictions.csv</code>. They are not restricted to the held-out validation samples.</p>
{iframe_section('Prediction map', artifacts.get('prediction_map'))}
{iframe_section('Training heatmap', artifacts.get('training_heatmap'))}
{iframe_section('Predicted heatmap', artifacts.get('predicted_heatmap'))}
{image_section('All matched samples: metadata vs predicted latitude', artifacts.get('all_lat_scatter'))}
{image_section('All matched samples: metadata vs predicted longitude', artifacts.get('all_lon_scatter'))}

<h2>Validation Set Only</h2>
<p class="small">These plots use only the held-out samples from <code>validation_set_run_*.csv</code>, meaning samples with known true coordinates that were used for evaluation.</p>
{iframe_section('Validation comparison map (validation set only)', artifacts.get('validation_map'))}
{image_section('Validation error distribution', artifacts.get('error_histogram'))}
{image_section('Cumulative validation error', artifacts.get('cumulative_error_curve'))}
{image_section('Validation distance bar chart', artifacts.get('distance_barchart'))}
{image_section('Validation set: true vs predicted latitude', artifacts.get('lat_scatter'))}
{image_section('Validation set: true vs predicted longitude', artifacts.get('lon_scatter'))}

<h2>Prediction Geography and Land Correction (all predicted samples)</h2>
{image_section('Land correction summary', artifacts.get('land_correction_summary'))}
{image_section('Land shift histogram', artifacts.get('land_shift_histogram'))}
{image_section('Country summary', artifacts.get('country_summary'))}

<h2>Validation Table (validation set only, first 200 rows)</h2>
{val_preview}

</body>
</html>
"""
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)


def build_full_report(predictions_csv, metadata_tsv, run_folder, performance_txt=None, report_name="gps_report.html"):
    run_folder = Path(run_folder)
    run_folder.mkdir(parents=True, exist_ok=True)

    report_folder = run_folder / "report"
    report_folder.mkdir(parents=True, exist_ok=True)

    pred_df = pd.read_csv(predictions_csv)
    meta_df = pd.read_csv(metadata_tsv, sep="\t")

    perf_path = Path(performance_txt) if performance_txt else run_folder / "final_model_performance.txt"
    metrics = read_performance_file(str(perf_path))

    validation_df = load_validation_table(run_folder)

    artifacts = {}

    artifacts["prediction_map"] = plot_prediction_map(pred_df, report_folder / "prediction_map.html")

    artifacts["validation_map"] = None
    artifacts["error_histogram"] = None
    artifacts["cumulative_error_curve"] = None
    artifacts["distance_barchart"] = None
    artifacts["lat_scatter"] = None
    artifacts["lon_scatter"] = None

    if validation_df is not None and not validation_df.empty:
        artifacts["validation_map"] = plot_validation_map(validation_df, report_folder / "validation_map.html")
        artifacts["error_histogram"] = plot_error_histogram(validation_df, report_folder / "validation_error_distribution.png")
        artifacts["cumulative_error_curve"] = plot_cumulative_error_curve(validation_df, report_folder / "cumulative_validation_error.png")
        artifacts["distance_barchart"] = plot_distance_barchart_report(validation_df, report_folder / "validation_distance_barchart.png")

        lat_scatter, lon_scatter = plot_lat_lon_scatter(
            validation_df,
            report_folder / "validation_latitude_scatter.png",
            report_folder / "validation_longitude_scatter.png",
        )
        artifacts["lat_scatter"] = lat_scatter
        artifacts["lon_scatter"] = lon_scatter

    train_heat_df = meta_df.rename(columns={"x": "latitude_true", "y": "longitude_true"})
    artifacts["training_heatmap"] = plot_heatmap(
        train_heat_df,
        "latitude_true",
        "longitude_true",
        report_folder / "training_heatmap.html",
        "Training Data Heatmap",
    )

    pred_lat_col = "latitude_land" if "latitude_land" in pred_df.columns else "latitude_centroid"
    pred_lon_col = "longitude_land" if "longitude_land" in pred_df.columns else "longitude_centroid"
    artifacts["predicted_heatmap"] = plot_heatmap(
        pred_df,
        pred_lat_col,
        pred_lon_col,
        report_folder / "predicted_heatmap.html",
        "Predicted Data Heatmap",
    )

    all_lat_scatter, all_lon_scatter = plot_all_predictions_vs_metadata_scatter(
        pred_df,
        meta_df,
        report_folder / "all_samples_latitude_scatter.png",
        report_folder / "all_samples_longitude_scatter.png",
    )
    artifacts["all_lat_scatter"] = all_lat_scatter
    artifacts["all_lon_scatter"] = all_lon_scatter

    artifacts["land_shift_histogram"] = plot_land_shift_histogram(pred_df, report_folder / "land_shift_histogram.png")
    artifacts["land_correction_summary"] = plot_land_correction_summary(pred_df, report_folder / "land_correction_summary.png")
    artifacts["country_summary"] = plot_country_summary(pred_df, report_folder / "country_summary.png")

    histories = collect_training_histories(run_folder)
    training_summary = plot_best_training_history(histories, report_folder / "training_history_best_run.png")

    report_path = report_folder / report_name
    render_html_report(report_path, metrics, artifacts, pred_df, validation_df, training_summary)
    print(f"Report written to: {report_path}")
    return report_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build a full HTML report for SNP2GPS outputs.")
    parser.add_argument("--predictions_csv", required=True, help="Path to final_averaged_predictions.csv")
    parser.add_argument("--metadata_tsv", required=True, help="Path to metadata TSV with sampleID, x, y")
    parser.add_argument("--run_folder", required=True, help="Validation run folder, e.g. output/0")
    parser.add_argument("--performance_txt", default=None, help="Optional path to final_model_performance.txt")
    parser.add_argument("--report_name", default="gps_report.html", help="Output HTML report filename")
    args = parser.parse_args()

    build_full_report(
        predictions_csv=args.predictions_csv,
        metadata_tsv=args.metadata_tsv,
        run_folder=args.run_folder,
        performance_txt=args.performance_txt,
        report_name=args.report_name,
    )