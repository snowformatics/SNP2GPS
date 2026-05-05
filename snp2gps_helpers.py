import os
import math
import pickle
import h5py

import numpy as np
import pandas as pd
import geopandas as gpd
from scipy import stats
from shapely.geometry import Point

COUNTRY_COL = "ADMIN"
EARTH_RADIUS_KM = 6371.0
world = gpd.read_file("ne_110m_admin_0_countries.shp")


def validate_coordinates(df, lat_col="latitude", lon_col="longitude", sample_col="sampleID", name="data"):
    invalid_mask = ~df[lat_col].between(-90, 90) | ~df[lon_col].between(-180, 180)

    if invalid_mask.any():
        cols_to_show = [c for c in [sample_col, lat_col, lon_col] if c in df.columns]
        bad_rows = df.loc[invalid_mask, cols_to_show].head(20)

        raise ValueError(
            f"Invalid coordinates found in {name}. "
            f"This may indicate swapped latitude/longitude columns.\n"
            f"{bad_rows.to_string(index=False)}"
        )


def detect_possible_swap(df, lat_col="latitude", lon_col="longitude"):
    normal_valid = df[lat_col].between(-90, 90).sum() + df[lon_col].between(-180, 180).sum()
    swapped_valid = df[lon_col].between(-90, 90).sum() + df[lat_col].between(-180, 180).sum()
    return swapped_valid > normal_valid


def wrap_longitude(lon):
    return ((lon + 180) % 360) - 180


def calculate_centroid(latitudes, longitudes):
    latitudes = np.asarray(latitudes, dtype=float)
    longitudes = np.asarray(longitudes, dtype=float)

    x = np.cos(np.radians(latitudes)) * np.cos(np.radians(longitudes))
    y = np.cos(np.radians(latitudes)) * np.sin(np.radians(longitudes))
    z = np.sin(np.radians(latitudes))

    x_mean = np.mean(x)
    y_mean = np.mean(y)
    z_mean = np.mean(z)

    lon = np.degrees(np.arctan2(y_mean, x_mean))
    hyp = np.sqrt(x_mean * x_mean + y_mean * y_mean)
    lat = np.degrees(np.arctan2(z_mean, hyp))

    lon = wrap_longitude(lon)
    return lat, lon


def haversine(lat1, lon1, lat2, lon2):
    R = EARTH_RADIUS_KM

    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    return R * c


def get_location_geopandas(lat, lon):
    point = Point(lon, lat)
    country = world.loc[world.intersects(point), COUNTRY_COL]

    if not country.empty:
        return country.values[0], lat, lon
    else:
        return get_nearest_land_country(lat, lon)


def get_nearest_land_country(latitude, longitude, step_km=5, max_distance_km=10000, log_file=None):
    def calculate_new_coords(lat, lon, distance_km, bearing_deg):
        lat_rad = math.radians(lat)
        lon_rad = math.radians(lon)
        bearing_rad = math.radians(bearing_deg)

        angular_distance = distance_km / EARTH_RADIUS_KM

        new_lat = math.asin(
            math.sin(lat_rad) * math.cos(angular_distance) +
            math.cos(lat_rad) * math.sin(angular_distance) * math.cos(bearing_rad)
        )

        new_lon = lon_rad + math.atan2(
            math.sin(bearing_rad) * math.sin(angular_distance) * math.cos(lat_rad),
            math.cos(angular_distance) - math.sin(lat_rad) * math.sin(new_lat)
        )

        new_lat = math.degrees(new_lat)
        new_lon = wrap_longitude(math.degrees(new_lon))

        return new_lat, new_lon

    point = Point(longitude, latitude)
    country = world.loc[world.intersects(point), COUNTRY_COL]

    if not country.empty:
        return country.values[0], latitude, longitude

    bearings = [0, 45, 90, 135, 180, 225, 270, 315]

    distance = step_km
    while distance <= max_distance_km:
        for bearing in bearings:
            lat, lon = calculate_new_coords(latitude, longitude, distance, bearing)
            point = Point(lon, lat)
            country = world.loc[world.intersects(point), COUNTRY_COL]

            if not country.empty:
                found_country = country.values[0]

                if log_file is not None:
                    data = pd.DataFrame({
                        "Original Latitude": [latitude],
                        "Original Longitude": [longitude],
                        "New Latitude": [lat],
                        "New Longitude": [lon],
                        "Country": [found_country]
                    })
                    data.to_csv(
                        log_file,
                        mode="a",
                        header=not os.path.exists(log_file),
                        index=False
                    )

                return found_country, lat, lon

        distance += step_km

    return None, None, None


def calculate_average_position(predictions, nearest_land_log_file=None):
    validate_coordinates(predictions, lat_col="latitude", lon_col="longitude", name="predictions")

    stats_result = predictions.groupby("sampleID").agg(
        latitude_mean=("latitude", "mean"),
        longitude_mean=("longitude", "mean"),
        latitude_sd=("latitude", "std"),
        longitude_sd=("longitude", "std")
    ).reset_index()

    def centroid_series(df):
        lat, lon = calculate_centroid(df["latitude"].values, df["longitude"].values)
        return pd.Series({
            "latitude_centroid": lat,
            "longitude_centroid": lon
        })

    centroid_result = predictions.groupby("sampleID", group_keys=False).apply(centroid_series).reset_index()

    def get_land_info(row):
        country, lat_land, lon_land = get_location_geopandas(
            row["latitude_centroid"],
            row["longitude_centroid"]
        )

        if lat_land is None or lon_land is None:
            corrected = np.nan
            correction_note = "no_land_found"
            land_shift_km = np.nan
        else:
            corrected = not (
                np.isclose(row["latitude_centroid"], lat_land, atol=1e-6) and
                np.isclose(row["longitude_centroid"], lon_land, atol=1e-6)
            )

            correction_note = "corrected_to_land" if corrected else "original_on_land"

            land_shift_km = haversine(
                row["latitude_centroid"],
                row["longitude_centroid"],
                lat_land,
                lon_land
            )

        return pd.Series({
            "country": country,
            "latitude_land": lat_land,
            "longitude_land": lon_land,
            "was_corrected_to_land": corrected,
            "correction_note": correction_note,
            "land_shift_km": land_shift_km
        })

    land_info = centroid_result.apply(get_land_info, axis=1)
    centroid_result = pd.concat([centroid_result, land_info], axis=1)

    result = pd.merge(stats_result, centroid_result, on="sampleID")
    return result


def remove_outliers_with_zscore(predictions, num_runs, z_threshold, output_path=None):
    validate_coordinates(predictions, lat_col="latitude", lon_col="longitude", name="predictions")

    if num_runs < 5:
        return predictions.copy()

    def calculate_zscore(group):
        group = group.copy()

        coords = group[["longitude", "latitude"]].to_numpy(dtype=float)
        z_scores = stats.zscore(coords, axis=0, nan_policy="omit")
        z_scores = np.nan_to_num(z_scores, nan=0.0)

        group["zscore_lon"] = z_scores[:, 0]
        group["zscore_lat"] = z_scores[:, 1]
        group["is_outlier"] = (
            (np.abs(group["zscore_lon"]) > z_threshold) |
            (np.abs(group["zscore_lat"]) > z_threshold)
        )
        return group

    predictions = predictions.groupby("sampleID", group_keys=False).apply(calculate_zscore)

    if output_path is not None:
        all_outliers_path = output_path.replace("removed_outliers.csv", "all_outliers.csv")
        predictions.to_csv(all_outliers_path, index=False)

    outliers = predictions[predictions["is_outlier"]]
    cleaned_predictions = predictions[~predictions["is_outlier"]].copy()

    if output_path is not None and not outliers.empty:
        outliers.to_csv(output_path, index=False)
        print(f"Removed {len(outliers)} outliers, saved to {output_path}")

    return cleaned_predictions


def save_data(data, filename):
    ext = os.path.splitext(filename)[-1].lower()

    if ext == ".pkl":
        with open(filename, "wb") as f:
            pickle.dump(data, f)

    elif ext == ".npz":
        np.savez(filename, **data)

    elif ext == ".h5":
        with h5py.File(filename, "w") as f:
            for key, value in data.items():
                if isinstance(value, np.ndarray):
                    f.create_dataset(key, data=value)
                elif isinstance(value, (list, tuple)):
                    f.create_dataset(key, data=np.array(value))
                elif isinstance(value, (int, float, str, bytes, np.integer, np.floating)):
                    f.attrs[key] = value
                else:
                    f.attrs[key] = str(value)
    else:
        raise ValueError(f"Unsupported file format: {ext}")


def load_data(filename):
    ext = os.path.splitext(filename)[-1].lower()

    if ext == ".pkl":
        with open(filename, "rb") as f:
            data = pickle.load(f)

    elif ext == ".npz":
        data = dict(np.load(filename, allow_pickle=True))

    elif ext == ".h5":
        data = {}
        with h5py.File(filename, "r") as f:
            for key in f.keys():
                data[key] = f[key][()]
            for key in f.attrs.keys():
                data[key] = f.attrs[key]

    else:
        raise ValueError(f"Unsupported file format: {ext}")

    return data


def validate_passport_coordinates(df, lat_col="passport_latitude", lon_col="passport_longitude", sample_col="sampleID"):
    validate_coordinates(df, lat_col=lat_col, lon_col=lon_col, sample_col=sample_col, name="passport data")
# import os
# import math
# import pickle
# import h5py
#
# import numpy as np
# import pandas as pd
# import geopandas as gpd
# from scipy import stats
# from shapely.geometry import Point
#
# COUNTRY_COL = "ADMIN"
# EARTH_RADIUS_KM = 6371.0
# world = gpd.read_file("ne_110m_admin_0_countries.shp")
#
#
# def validate_coordinates(df, lat_col="latitude", lon_col="longitude", sample_col="sampleID", name="data"):
#     invalid_mask = ~df[lat_col].between(-90, 90) | ~df[lon_col].between(-180, 180)
#
#     if invalid_mask.any():
#         cols_to_show = [c for c in [sample_col, lat_col, lon_col] if c in df.columns]
#         bad_rows = df.loc[invalid_mask, cols_to_show].head(20)
#
#         raise ValueError(
#             f"Invalid coordinates found in {name}. "
#             f"This may indicate swapped latitude/longitude columns.\n"
#             f"{bad_rows.to_string(index=False)}"
#         )
#
#
# def detect_possible_swap(df, lat_col="latitude", lon_col="longitude"):
#     normal_valid = df[lat_col].between(-90, 90).sum() + df[lon_col].between(-180, 180).sum()
#     swapped_valid = df[lon_col].between(-90, 90).sum() + df[lat_col].between(-180, 180).sum()
#     return swapped_valid > normal_valid
#
#
# def wrap_longitude(lon):
#     return ((lon + 180) % 360) - 180
#
#
# def calculate_centroid(latitudes, longitudes):
#     latitudes = np.asarray(latitudes, dtype=float)
#     longitudes = np.asarray(longitudes, dtype=float)
#
#     x = np.cos(np.radians(latitudes)) * np.cos(np.radians(longitudes))
#     y = np.cos(np.radians(latitudes)) * np.sin(np.radians(longitudes))
#     z = np.sin(np.radians(latitudes))
#
#     x_mean = np.mean(x)
#     y_mean = np.mean(y)
#     z_mean = np.mean(z)
#
#     lon = np.degrees(np.arctan2(y_mean, x_mean))
#     hyp = np.sqrt(x_mean * x_mean + y_mean * y_mean)
#     lat = np.degrees(np.arctan2(z_mean, hyp))
#
#     lon = wrap_longitude(lon)
#     return lat, lon
#
#
# def haversine(lat1, lon1, lat2, lon2):
#     R = EARTH_RADIUS_KM
#
#     lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
#
#     dlat = lat2 - lat1
#     dlon = lon2 - lon1
#
#     a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
#     c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
#
#     return R * c
#
#
# def get_location_geopandas(lat, lon):
#     point = Point(lon, lat)
#     country = world.loc[world.intersects(point), COUNTRY_COL]
#
#     if not country.empty:
#         return country.values[0], lat, lon
#     else:
#         return get_nearest_land_country(lat, lon)
#
#
# def get_nearest_land_country(latitude, longitude, step_km=5, max_distance_km=10000, log_file=None):
#     def calculate_new_coords(lat, lon, distance_km, bearing_deg):
#         lat_rad = math.radians(lat)
#         lon_rad = math.radians(lon)
#         bearing_rad = math.radians(bearing_deg)
#
#         angular_distance = distance_km / EARTH_RADIUS_KM
#
#         new_lat = math.asin(
#             math.sin(lat_rad) * math.cos(angular_distance) +
#             math.cos(lat_rad) * math.sin(angular_distance) * math.cos(bearing_rad)
#         )
#
#         new_lon = lon_rad + math.atan2(
#             math.sin(bearing_rad) * math.sin(angular_distance) * math.cos(lat_rad),
#             math.cos(angular_distance) - math.sin(lat_rad) * math.sin(new_lat)
#         )
#
#         new_lat = math.degrees(new_lat)
#         new_lon = wrap_longitude(math.degrees(new_lon))
#
#         return new_lat, new_lon
#
#     point = Point(longitude, latitude)
#     country = world.loc[world.intersects(point), COUNTRY_COL]
#
#     if not country.empty:
#         return country.values[0], latitude, longitude
#
#     bearings = [0, 45, 90, 135, 180, 225, 270, 315]
#
#     distance = step_km
#     while distance <= max_distance_km:
#         for bearing in bearings:
#             lat, lon = calculate_new_coords(latitude, longitude, distance, bearing)
#             point = Point(lon, lat)
#             country = world.loc[world.intersects(point), COUNTRY_COL]
#
#             if not country.empty:
#                 found_country = country.values[0]
#
#                 if log_file is not None:
#                     data = pd.DataFrame({
#                         "Original Latitude": [latitude],
#                         "Original Longitude": [longitude],
#                         "New Latitude": [lat],
#                         "New Longitude": [lon],
#                         "Country": [found_country]
#                     })
#                     data.to_csv(
#                         log_file,
#                         mode="a",
#                         header=not os.path.exists(log_file),
#                         index=False
#                     )
#
#                 return found_country, lat, lon
#
#         distance += step_km
#
#     return None, None, None
#
#
# def calculate_average_position(predictions, nearest_land_log_file=None):
#     validate_coordinates(predictions, lat_col="latitude", lon_col="longitude", name="predictions")
#
#     stats_result = predictions.groupby("sampleID").agg(
#         latitude_mean=("latitude", "mean"),
#         longitude_mean=("longitude", "mean"),
#         latitude_sd=("latitude", "std"),
#         longitude_sd=("longitude", "std")
#     ).reset_index()
#
#     def centroid_series(df):
#         lat, lon = calculate_centroid(df["latitude"].values, df["longitude"].values)
#         return pd.Series({
#             "latitude_centroid": lat,
#             "longitude_centroid": lon
#         })
#
#     centroid_result = predictions.groupby("sampleID", group_keys=False).apply(centroid_series).reset_index()
#
#     centroid_result[["country", "latitude_land", "longitude_land"]] = centroid_result.apply(
#         lambda row: pd.Series(get_location_geopandas(row["latitude_centroid"], row["longitude_centroid"])),
#         axis=1
#     )
#
#     result = pd.merge(stats_result, centroid_result, on="sampleID")
#     return result
#
#
# def remove_outliers_with_zscore(predictions, num_runs, z_threshold, output_path=None):
#     validate_coordinates(predictions, lat_col="latitude", lon_col="longitude", name="predictions")
#
#     if num_runs < 5:
#         return predictions.copy()
#
#     def calculate_zscore(group):
#         group = group.copy()
#
#         coords = group[["longitude", "latitude"]].to_numpy(dtype=float)
#         z_scores = stats.zscore(coords, axis=0, nan_policy="omit")
#         z_scores = np.nan_to_num(z_scores, nan=0.0)
#
#         group["zscore_lon"] = z_scores[:, 0]
#         group["zscore_lat"] = z_scores[:, 1]
#         group["is_outlier"] = (
#             (np.abs(group["zscore_lon"]) > z_threshold) |
#             (np.abs(group["zscore_lat"]) > z_threshold)
#         )
#         return group
#
#     predictions = predictions.groupby("sampleID", group_keys=False).apply(calculate_zscore)
#
#     if output_path is not None:
#         all_outliers_path = output_path.replace("removed_outliers.csv", "all_outliers.csv")
#         predictions.to_csv(all_outliers_path, index=False)
#
#     outliers = predictions[predictions["is_outlier"]]
#     cleaned_predictions = predictions[~predictions["is_outlier"]].copy()
#
#     if output_path is not None and not outliers.empty:
#         outliers.to_csv(output_path, index=False)
#         print(f"Removed {len(outliers)} outliers, saved to {output_path}")
#
#     return cleaned_predictions
#
#
# def save_data(data, filename):
#     ext = os.path.splitext(filename)[-1].lower()
#
#     if ext == ".pkl":
#         with open(filename, "wb") as f:
#             pickle.dump(data, f)
#
#     elif ext == ".npz":
#         np.savez(filename, **data)
#
#     elif ext == ".h5":
#         with h5py.File(filename, "w") as f:
#             for key, value in data.items():
#                 if isinstance(value, np.ndarray):
#                     f.create_dataset(key, data=value)
#                 elif isinstance(value, (list, tuple)):
#                     f.create_dataset(key, data=np.array(value))
#                 elif isinstance(value, (int, float, str, bytes, np.integer, np.floating)):
#                     f.attrs[key] = value
#                 else:
#                     f.attrs[key] = str(value)
#     else:
#         raise ValueError(f"Unsupported file format: {ext}")
#
#
# def load_data(filename):
#     ext = os.path.splitext(filename)[-1].lower()
#
#     if ext == ".pkl":
#         with open(filename, "rb") as f:
#             data = pickle.load(f)
#
#     elif ext == ".npz":
#         data = dict(np.load(filename, allow_pickle=True))
#
#     elif ext == ".h5":
#         data = {}
#         with h5py.File(filename, "r") as f:
#             for key in f.keys():
#                 data[key] = f[key][()]
#             for key in f.attrs.keys():
#                 data[key] = f.attrs[key]
#
#     else:
#         raise ValueError(f"Unsupported file format: {ext}")
#
#     return data
#
#
# def validate_passport_coordinates(df, lat_col="passport_latitude", lon_col="passport_longitude", sample_col="sampleID"):
#     validate_coordinates(df, lat_col=lat_col, lon_col=lon_col, sample_col=sample_col, name="passport data")