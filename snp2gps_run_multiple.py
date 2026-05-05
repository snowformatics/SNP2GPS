import os
import gc
import random
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras.backend as K

from snp2gps_helpers import (
    calculate_average_position,
    remove_outliers_with_zscore,
    haversine,
)

from snps2gps_report import build_full_report

def snp2gps_launcher(
    path_zarr: str,
    gps_data: str,
    output_folder: str,
    run_id: str,
    min_minor_allele_count: int,
    apply_imputation: bool,
    train_ratio: float,
    num_layers: int,
    layer_width: int,
    dropout_rate: float,
    max_epochs: int,
    batch_size: int,
    num_runs: int,
    validation_size: int,
    validation_runs: int,
    genotype_npz_path: str = None,
    use_saved_npz: bool = False,
    z_threshold: float = 1.5,
    make_pca_plots: bool = False,
    make_tsne_plots: bool = False,
    target_mode: str = "latlon",
):
    from snp2gps import (
        load_genotypes,
        sort_samples,
        prepare_target_coordinates,
        filter_variants,
        setup_training_callbacks,
        split_train_test,
        build_neural_network,
        train_neural_network,
        predict_sample_locations,
        sphere_to_latlon,
    )

    def set_all_seeds(seed: int):
        np.random.seed(seed)
        random.seed(seed)
        tf.random.set_seed(seed)

    def validate_lat_lon_dataframe(df: pd.DataFrame, lat_col: str, lon_col: str, df_name: str):
        if lat_col not in df.columns or lon_col not in df.columns:
            raise ValueError(
                f"{df_name} must contain columns '{lat_col}' and '{lon_col}'. "
                f"Available columns: {list(df.columns)}"
            )

        lat = pd.to_numeric(df[lat_col], errors="coerce")
        lon = pd.to_numeric(df[lon_col], errors="coerce")

        bad = df[lat.isna() | lon.isna() | ~lat.between(-90, 90) | ~lon.between(-180, 180)]

        if not bad.empty:
            raise ValueError(
                f"Invalid coordinates detected in {df_name}.\n"
                f"Expected: {lat_col}=latitude in [-90,90], {lon_col}=longitude in [-180,180].\n"
                f"First problematic rows:\n{bad[[c for c in ['sampleID', lat_col, lon_col] if c in bad.columns]].head(10).to_string(index=False)}"
            )

    def validate_prediction_dataframe(df: pd.DataFrame, df_name: str = "prediction dataframe"):
        required = ["sampleID", "latitude", "longitude"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"{df_name} is missing required columns: {missing}")

        validate_lat_lon_dataframe(df, "latitude", "longitude", df_name)

    def get_run_folder(base_folder: str, val_run_idx: int) -> str:
        folder = os.path.join(base_folder, str(val_run_idx))
        os.makedirs(folder, exist_ok=True)
        return folder

    def load_input_genotypes():
        if use_saved_npz:
            if genotype_npz_path is None:
                raise ValueError("use_saved_npz=True but genotype_npz_path is None.")
            if not os.path.exists(genotype_npz_path):
                raise FileNotFoundError(f"NPZ file not found: {genotype_npz_path}")

            print(f"Loading genotype data from saved NPZ: {genotype_npz_path}")
            data = np.load(genotype_npz_path, allow_pickle=True)

            if "genotype_array" not in data or "samples" not in data:
                raise ValueError(
                    f"NPZ file must contain 'genotype_array' and 'samples'. "
                    f"Found keys: {list(data.keys())}"
                )

            genotype_array = data["genotype_array"]
            samples = data["samples"]
            return genotype_array, samples

        print(f"Loading genotype data from zarr: {path_zarr}")
        genotypes, samples = load_genotypes(path_zarr)
        genotype_array = filter_variants(genotypes, min_minor_allele_count, apply_imputation)
        return genotype_array, samples

    def build_set_info_dataframe(train_indices, test_indices, prediction_indices, samples):
        all_indices = np.concatenate([train_indices, test_indices, prediction_indices])
        set_types = (
            ["train"] * len(train_indices)
            + ["test"] * len(test_indices)
            + ["prediction"] * len(prediction_indices)
        )

        set_info = pd.DataFrame({
            "sample_index": all_indices,
            "set_type": set_types
        })

        if isinstance(samples, np.ndarray):
            set_info["sampleID"] = samples[set_info["sample_index"]]
        elif isinstance(samples, pd.DataFrame):
            if "sampleID" not in samples.columns:
                raise ValueError("If 'samples' is a DataFrame, it must contain a 'sampleID' column.")
            set_info["sampleID"] = samples.iloc[set_info["sample_index"]]["sampleID"].values
        else:
            raise TypeError("Unsupported type for 'samples'. Expected numpy array or pandas DataFrame.")

        return set_info[["sample_index", "sampleID", "set_type"]]

    if validation_runs is None:
        validation_runs = 1
    if validation_runs < 1:
        raise ValueError("validation_runs must be >= 1")

    if num_runs < 1:
        raise ValueError("num_runs must be >= 1")

    os.makedirs(output_folder, exist_ok=True)

    genotype_array, samples = load_input_genotypes()
    print(f"Loaded genotype matrix with shape: {genotype_array.shape}")
    print(f"Loaded {len(samples)} samples")
    print(f"Target mode: {target_mode}")

    for val_runs in range(validation_runs):
        print(f"\n{'=' * 80}")
        print(f"Starting validation run {val_runs + 1}/{validation_runs}")
        print(f"{'=' * 80}\n")

        run_folder = get_run_folder(output_folder, val_runs)

        gps_df, sample_coordinates, validation_samples = sort_samples(samples, gps_data, validation_size)

        if isinstance(sample_coordinates, np.ndarray):
            temp_df = pd.DataFrame(sample_coordinates, columns=["x", "y"])
            validate_lat_lon_dataframe(temp_df.dropna(), "x", "y", "sample_coordinates")

        if validation_samples is not None and not validation_samples.empty:
            if isinstance(validation_samples, pd.DataFrame):
                validate_lat_lon_dataframe(validation_samples.dropna(subset=["x", "y"]), "x", "y", "validation_samples")

        transform_info, target_coordinates = prepare_target_coordinates(sample_coordinates, target_mode=target_mode)

        all_predicted_dfs = []
        all_r2_longitude = []
        all_r2_latitude = []
        all_mean_errors = []
        all_median_errors = []

        predicted_csv_path = os.path.join(run_folder, "predicted_gps.csv")

        with open(predicted_csv_path, "w") as output_file:
            for i in range(num_runs):
                print(f"\n### Running Model {i + 1}/{num_runs} with seed {i * 1000} ###\n")

                random_seed = i * 1000
                set_all_seeds(random_seed)

                (
                    train_indices,
                    test_indices,
                    train_genotypes,
                    test_genotypes,
                    train_coords,
                    test_coords,
                    prediction_indices,
                    prediction_genotypes
                ) = split_train_test(
                    genotype_array,
                    target_coordinates,
                    train_ratio,
                    random_seed
                )

                if make_pca_plots:
                    pca_path = os.path.join(run_folder, f"PCA_training_set_{i}.png")
                    plot_pca(genotype_array, train_indices, test_indices, pca_path)

                if make_tsne_plots:
                    tsne_path = os.path.join(run_folder, f"TSNE_training_set_{i}.png")
                    plot_tsne(genotype_array, train_indices, test_indices, tsne_path)

                set_info = build_set_info_dataframe(train_indices, test_indices, prediction_indices, samples)
                set_info_path = os.path.join(run_folder, f"training_set_info_run_{i}.csv")
                set_info.to_csv(set_info_path, index=False)
                print(f"Sample set info saved to: {set_info_path}")

                checkpoint_callback, early_stopping_callback, reduce_lr_callback = setup_training_callbacks(
                    run_folder,
                    f"{run_id}_run{i}",
                    patience=50
                )

                output_dim = 2 if target_mode == "latlon" else 3

                model = build_neural_network(
                    train_genotypes.shape[1],
                    num_layers,
                    layer_width,
                    dropout_rate,
                    output_dim=output_dim
                )

                history, model = train_neural_network(
                    model,
                    train_genotypes,
                    test_genotypes,
                    train_coords,
                    test_coords,
                    max_epochs,
                    batch_size,
                    checkpoint_callback,
                    early_stopping_callback,
                    reduce_lr_callback,
                    run_folder,
                    f"{run_id}_run{i}"
                )

                (
                    distance_errors,
                    predicted_df,
                    r2_longitude,
                    r2_latitude,
                    mean_error,
                    median_error
                ) = predict_sample_locations(
                    model,
                    prediction_genotypes,
                    transform_info,
                    test_coords,
                    prediction_indices,
                    samples,
                    test_genotypes,
                    run_folder,
                    f"{run_id}_run{i}",
                    output_file,
                    history,
                    verbose=False
                )

                validate_prediction_dataframe(predicted_df, f"predicted_df run {i}")

                all_predicted_dfs.append(predicted_df)
                all_r2_longitude.append(r2_longitude)
                all_r2_latitude.append(r2_latitude)
                all_mean_errors.append(mean_error)
                all_median_errors.append(median_error)

                try:
                    K.clear_session()
                    del model
                    gc.collect()
                    try:
                        tf.config.experimental.reset_memory_stats("GPU:0")
                    except Exception:
                        pass
                except Exception:
                    pass

        if len(all_predicted_dfs) == 0:
            raise RuntimeError("No prediction data frames were produced.")

        combined_predictions = pd.concat(all_predicted_dfs, ignore_index=True)
        validate_prediction_dataframe(combined_predictions, "combined_predictions before outlier removal")

        outliers_path = os.path.join(run_folder, "removed_outliers.csv")
        combined_predictions = remove_outliers_with_zscore(
            combined_predictions,
            num_runs,
            z_threshold=1.5,
            output_path=outliers_path
        )

        validate_prediction_dataframe(combined_predictions, "combined_predictions after outlier removal")

        final_averaged_predictions = calculate_average_position(combined_predictions)
        final_output_path = os.path.join(run_folder, "final_averaged_predictions.csv")
        final_averaged_predictions.to_csv(final_output_path, index=False)

        if validation_samples is not None and not validation_samples.empty:
            validation_samples = validation_samples.copy().reset_index(drop=True)
            validation_samples = validation_samples.rename(columns={"x": "latitude", "y": "longitude"})
            validate_lat_lon_dataframe(validation_samples, "latitude", "longitude", "validation_samples renamed")

            merged_validation = pd.merge(
                validation_samples,
                final_averaged_predictions,
                on="sampleID",
                how="left"
            )

            def safe_distance(row):
                pred_lat = row["latitude_land"] if pd.notna(row.get("latitude_land", np.nan)) else row[
                    "latitude_centroid"]
                pred_lon = row["longitude_land"] if pd.notna(row.get("longitude_land", np.nan)) else row[
                    "longitude_centroid"]

                needed = ["latitude", "longitude"]
                if any(pd.isna(row[col]) for col in needed) or pd.isna(pred_lat) or pd.isna(pred_lon):
                    return np.nan

                return haversine(
                    row["latitude"],
                    row["longitude"],
                    pred_lat,
                    pred_lon
                )

            merged_validation["distance_km"] = merged_validation.apply(safe_distance, axis=1)
            merged_validation = merged_validation.sort_values(by="distance_km", na_position="last")

            val_path = os.path.join(run_folder, f"validation_set_run_{val_runs}.csv")
            merged_validation.to_csv(val_path, index=False)


        final_r2_longitude = float(np.mean(all_r2_longitude)) if all_r2_longitude else np.nan
        final_r2_latitude = float(np.mean(all_r2_latitude)) if all_r2_latitude else np.nan
        final_mean_error = float(np.mean(all_mean_errors)) if all_mean_errors else np.nan
        final_median_error = float(np.mean(all_median_errors)) if all_median_errors else np.nan

        print("\n### Final Model Performance Metrics Across Runs ###\n")
        print(f"R² (Longitude): {final_r2_longitude:.4f}")
        print(f"R² (Latitude): {final_r2_latitude:.4f}")
        print(f"Mean Validation Error: {final_mean_error:.4f} km")
        print(f"Median Validation Error: {final_median_error:.4f} km")

        performance_output_path = os.path.join(run_folder, "final_model_performance.txt")
        with open(performance_output_path, "w", encoding="utf-8") as f:
            f.write(f"R2 (Longitude): {final_r2_longitude:.4f}\n")
            f.write(f"R2 (Latitude): {final_r2_latitude:.4f}\n")
            f.write(f"Mean Validation Error: {final_mean_error:.4f} km\n")
            f.write(f"Median Validation Error: {final_median_error:.4f} km\n")

        build_full_report(
            predictions_csv=os.path.join(run_folder, "final_averaged_predictions.csv"),
            metadata_tsv=gps_data,
            run_folder=run_folder,
            performance_txt=os.path.join(run_folder, "final_model_performance.txt"),
            report_name="gps_report.html",
        )

        print(f"\n### Final Model Performance Metrics Saved to: {performance_output_path} ###\n")
        print(f"Finished validation run {val_runs + 1}/{validation_runs}\n")