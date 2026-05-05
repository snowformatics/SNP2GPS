import os
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import matplotlib.pyplot as plt


world = gpd.read_file("ne_110m_admin_0_countries.shp")


def validate_coordinates(df, lat_col, lon_col):
    df = df.copy()
    df[lat_col] = pd.to_numeric(df[lat_col], errors="coerce")
    df[lon_col] = pd.to_numeric(df[lon_col], errors="coerce")

    bad = df[df[lat_col].isna() | df[lon_col].isna() |
             ~df[lat_col].between(-90, 90) |
             ~df[lon_col].between(-180, 180)]

    if not bad.empty:
        raise ValueError(
            f"Invalid coordinates found:\n{bad[[lat_col, lon_col]].head(10)}"
        )

    return df


def get_country(lat, lon):
    point = Point(lon, lat)
    match = world.loc[world.intersects(point), "ADMIN"]
    return match.values[0] if not match.empty else ""


def normalize_country(name):
    aliases = {
        "usa": "united states of america",
        "uk": "united kingdom",
    }
    name = str(name).strip().lower()
    return aliases.get(name, name)


def check_country_mismatches(df, id_col, lat_col, lon_col, country_col,
                             mismatches_out="country_mismatches.csv"):
    df = df.dropna(subset=[lat_col, lon_col]).copy()
    df["COUNTRY_FROM_COORDS"] = df.apply(lambda r: get_country(r[lat_col], r[lon_col]), axis=1)

    csv_country = df[country_col].fillna("").map(normalize_country)
    coord_country = df["COUNTRY_FROM_COORDS"].fillna("").map(normalize_country)

    mismatches = df[csv_country != coord_country]

    if not mismatches.empty:
        mismatches[[id_col, country_col, "COUNTRY_FROM_COORDS"]].to_csv(mismatches_out, index=False)
        print(f"⚠️ {len(mismatches)} mismatches saved to '{mismatches_out}'.")
    else:
        print("✅ No mismatches found.")

    return mismatches


def save_points_in_water(df, id_col, lat_col, lon_col, outfile="points_in_water.csv"):
    df = df.dropna(subset=[lat_col, lon_col]).copy()
    land = world.unary_union

    def in_water(lat, lon):
        p = Point(float(lon), float(lat))
        return not (land.contains(p) or land.touches(p))

    df["IN_WATER"] = df.apply(lambda r: in_water(r[lat_col], r[lon_col]), axis=1)
    water_df = df[df["IN_WATER"]].copy()

    if not water_df.empty:
        water_df[[id_col, lat_col, lon_col]].to_csv(outfile, index=False)
        print(f"🌊 {len(water_df)} coordinates in water saved to '{outfile}'.")
    else:
        print("✅ No points in water found.")
    return water_df


def histogram_countries_from_df(df, lat_col, lon_col, output_png="country_histogram.png", return_counts=False):
    df = validate_coordinates(df, lat_col, lon_col)
    df = df.copy()

    df["country"] = df.apply(lambda r: get_country(r[lat_col], r[lon_col]), axis=1)
    df["country"] = df["country"].replace("", "Unknown")

    counts = df["country"].value_counts().sort_values(ascending=False)

    plt.figure(figsize=(10, max(4, 0.25 * len(counts))))
    ax = counts.plot(kind="bar")
    plt.xlabel("Country")
    plt.ylabel("Number of points")
    plt.title("GPS Points per Country")
    plt.xticks(rotation=90, ha="right")
    plt.tight_layout()

    for p in ax.patches:
        ax.annotate(
            str(int(p.get_height())),
            (p.get_x() + p.get_width() / 2., p.get_height()),
            ha="center", va="bottom",
            xytext=(0, 3), textcoords="offset points"
        )

    plt.savefig(output_png, dpi=150, bbox_inches="tight")
    plt.close()

    if return_counts:
        return counts


def run_preprocessing_pipeline(csv_path: str, id_col: str, lat_col: str, lon_col: str, country_col: str, out_path: str,
                               sep: str = "\t"):
    os.makedirs(out_path, exist_ok=True)

    df = pd.read_csv(csv_path, sep=sep)
    df = validate_coordinates(df, lat_col, lon_col)

    print("[1/4] Checking country mismatches...")
    check_country_mismatches(
        df, id_col, lat_col, lon_col, country_col,
        os.path.join(out_path, "country_mismatches.csv")
    )

    print("[2/4] Detecting GPS points in water...")
    save_points_in_water(
        df, id_col, lat_col, lon_col,
        os.path.join(out_path, "gps_in_water.csv")
    )

    print("[3/4] Creating country histogram...")
    histogram_countries_from_df(
        df, lat_col, lon_col,
        os.path.join(out_path, "country_histogram.png"),
        return_counts=True
    )

    print("[4/4] Plotting simple map...")
    plot_simple_map(
        csv_path, lat_col, lon_col, id_col, country_col,
        os.path.join(out_path, "map_simple.html"), sep
    )

    print("✓ Pipeline completed.")