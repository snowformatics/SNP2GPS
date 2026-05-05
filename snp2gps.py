import argparse
import os

import allel
import zarr
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K

from snp2gps_run_multiple import snp2gps_launcher
from snp2gps_preprocessing import run_preprocessing_pipeline


def vcf_to_zarr(vcf_file: str, zarr_output: str):
    file_name = os.path.basename(vcf_file).split(".")[0]
    zarr_output = os.path.join(zarr_output, f"{file_name}.zarr")

    os.makedirs(os.path.dirname(zarr_output), exist_ok=True)

    if not os.path.exists(zarr_output):
        print("Starting conversion, this might take a while...")
        allel.vcf_to_zarr(vcf_file, zarr_output, fields="*", overwrite=True)
        print(f"Successfully converted {vcf_file} to {zarr_output}")
    else:
        print(f"Zarr output already exists at {zarr_output}, skipping conversion.")

    return zarr_output


def load_genotypes(path_zarr: str):
    callset = zarr.open_group(path_zarr, mode="r")
    gt = callset["calldata/GT"]
    genotypes = allel.GenotypeArray(gt[:])
    samples = callset["samples"][:]
    return genotypes, samples


def create_validation_set(validation_size, gps_df, samples, sample_coordinates, *, rng=None):
    if rng is None:
        rng = np.random.default_rng()

    available_samples = gps_df.loc[gps_df["x"].notna() & gps_df["y"].notna()]
    if len(available_samples) < validation_size:
        raise ValueError(
            f"Not enough samples with valid coordinates. Requested {validation_size}, "
            f"available {len(available_samples)}."
        )

    seed = int(rng.integers(0, 2**31 - 1))
    selected_samples = available_samples.sample(n=validation_size, random_state=seed)

    gps_df = gps_df.copy()
    gps_df.loc[selected_samples.index, ["x", "y"]] = np.nan

    sample_to_index = {str(sample_id): idx for idx, sample_id in enumerate(samples.astype(str))}
    for sample_id in selected_samples.index.astype(str):
        if sample_id in sample_to_index:
            sample_coordinates[sample_to_index[sample_id]] = [np.nan, np.nan]

    return gps_df, sample_coordinates, selected_samples


def sort_samples(samples, gps_data, validation_size):
    gps_df = pd.read_csv(gps_data, sep="\t").copy()

    required_cols = {"sampleID", "x", "y"}
    missing_cols = required_cols - set(gps_df.columns)
    if missing_cols:
        raise ValueError(f"GPS file is missing required columns: {sorted(missing_cols)}")

    gps_df["x"] = pd.to_numeric(gps_df["x"], errors="coerce")
    gps_df["y"] = pd.to_numeric(gps_df["y"], errors="coerce")

    gps_df["sampleID2"] = gps_df["sampleID"].astype(str)
    gps_df.set_index("sampleID", inplace=True)

    samples = samples.astype(str)
    gps_df = gps_df.reindex(np.array(samples))

    if len(samples) != len(gps_df):
        raise ValueError(
            f"Sample mismatch: VCF/Zarr has {len(samples)} samples but metadata has {len(gps_df)} rows "
            "after reindexing. All VCF IDs must be present in metadata; missing entries should be NA."
        )

    if not all(gps_df["sampleID2"].iloc[i] == samples[i] for i in range(len(samples))):
        raise ValueError("Sample ordering failed. Check that metadata sample IDs match the VCF/Zarr sample order.")

    sample_coordinates = np.array(gps_df[["x", "y"]], dtype=float)

    if validation_size:
        gps_df, sample_coordinates, selected_samples = create_validation_set(
            validation_size,
            gps_df,
            samples,
            sample_coordinates
        )
        selected_samples = selected_samples.drop(columns=["sampleID2"], errors="ignore").copy()
        selected_samples.reset_index(inplace=True)
    else:
        selected_samples = None

    return gps_df, sample_coordinates, selected_samples


def normalize_coordinates(sample_coordinates):
    mean_x = np.nanmean(sample_coordinates[:, 0])
    std_x = np.nanstd(sample_coordinates[:, 0])
    mean_y = np.nanmean(sample_coordinates[:, 1])
    std_y = np.nanstd(sample_coordinates[:, 1])

    if std_x == 0 or np.isnan(std_x):
        raise ValueError("Standard deviation of x (latitude) is zero or NaN; cannot normalize.")
    if std_y == 0 or np.isnan(std_y):
        raise ValueError("Standard deviation of y (longitude) is zero or NaN; cannot normalize.")

    normalized_coordinates = np.array([
        [(x[0] - mean_x) / std_x, (x[1] - mean_y) / std_y]
        for x in sample_coordinates
    ], dtype=float)

    return mean_x, std_x, mean_y, std_y, normalized_coordinates


def latlon_to_sphere(sample_coordinates):
    lat = np.radians(sample_coordinates[:, 0])
    lon = np.radians(sample_coordinates[:, 1])

    sphere_x = np.cos(lat) * np.cos(lon)
    sphere_y = np.cos(lat) * np.sin(lon)
    sphere_z = np.sin(lat)

    sphere_coords = np.column_stack([sphere_x, sphere_y, sphere_z]).astype(float)

    nan_mask = np.isnan(sample_coordinates[:, 0]) | np.isnan(sample_coordinates[:, 1])
    sphere_coords[nan_mask] = np.nan

    return sphere_coords


def sphere_to_latlon(sphere_coords):
    sphere_coords = np.asarray(sphere_coords, dtype=float)

    sx = sphere_coords[:, 0]
    sy = sphere_coords[:, 1]
    sz = sphere_coords[:, 2]

    norms = np.sqrt(sx**2 + sy**2 + sz**2)
    norms[norms == 0] = 1.0

    sx = sx / norms
    sy = sy / norms
    sz = sz / norms

    lon = np.degrees(np.arctan2(sy, sx))
    hyp = np.sqrt(sx**2 + sy**2)
    lat = np.degrees(np.arctan2(sz, hyp))

    return np.column_stack([lat, lon])


def prepare_target_coordinates(sample_coordinates, target_mode="latlon"):
    if target_mode == "latlon":
        mean_x, std_x, mean_y, std_y, target_coordinates = normalize_coordinates(sample_coordinates)
        transform_info = {
            "target_mode": "latlon",
            "mean_x": mean_x,
            "std_x": std_x,
            "mean_y": mean_y,
            "std_y": std_y
        }
        return transform_info, target_coordinates

    elif target_mode == "sphere":
        target_coordinates = latlon_to_sphere(sample_coordinates)
        transform_info = {
            "target_mode": "sphere"
        }
        return transform_info, target_coordinates

    else:
        raise ValueError(f"Unsupported target_mode: {target_mode}")


def impute_missing_data(genotypes):
    print("Imputing missing genotype data...")
    allele_counts = genotypes.count_alleles()[:, 1]
    alt_allele_counts = genotypes.to_allele_counts()[:, :, 1]
    missing_data_mask = genotypes.is_missing()
    num_samples = np.array([np.sum(~x) for x in missing_data_mask])

    allele_freqs = np.array([
        allele_counts[i] / (2 * num_samples[i]) if num_samples[i] > 0 else 0.0
        for i in range(len(num_samples))
    ])

    for i in range(alt_allele_counts.shape[0]):
        for j in range(alt_allele_counts.shape[1]):
            if missing_data_mask[i, j]:
                alt_allele_counts[i, j] = np.random.binomial(2, allele_freqs[i])

    return alt_allele_counts


def filter_variants(genotypes, min_minor_allele_count, apply_imputation):
    print("Filtering SNPs...")
    allele_counts = genotypes.count_alleles()
    biallelic_mask = allele_counts.is_biallelic()
    genotypes = genotypes[biallelic_mask, :, :]

    if min_minor_allele_count > 1:
        derived_counts = genotypes.count_alleles()[:, 1]
        filter_mask = np.array([x >= min_minor_allele_count for x in derived_counts])
        genotypes = genotypes[filter_mask, :, :]

    if apply_imputation:
        genotype_array = impute_missing_data(genotypes)
    else:
        genotype_array = genotypes.to_allele_counts()[:, :, 1]

    print(f"Processing {len(genotype_array)} genotypes after filtering.")
    return genotype_array


def setup_training_callbacks(output_prefix, run_id, patience=50):
    os.makedirs(output_prefix, exist_ok=True)
    checkpoint_path = os.path.join(output_prefix, f"{run_id}_weights.hdf5")

    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        verbose=1,
        save_best_only=True,
        save_weights_only=True,
        monitor="val_loss",
        save_freq="epoch"
    )

    early_stopping_callback = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        min_delta=0,
        patience=patience,
        verbose=1,
        restore_best_weights=True
    )

    reduce_lr_callback = tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=max(1, int(patience / 6)),
        verbose=1,
        mode="auto",
        min_delta=0,
        cooldown=0,
        min_lr=0
    )

    return checkpoint_callback, early_stopping_callback, reduce_lr_callback


def split_train_test(genotype_data, coordinates, train_ratio, random_seed):
    if not (0 < train_ratio <= 1):
        raise ValueError(f"train_ratio must be in (0, 1], got {train_ratio}")

    rng = np.random.default_rng(random_seed)

    train_pool = np.flatnonzero(~np.isnan(coordinates[:, 0]))

    if len(train_pool) == 0:
        raise ValueError("No samples with known coordinates available for training.")

    n_test = int(round((1 - train_ratio) * len(train_pool)))
    if len(train_pool) > 1:
        n_test = min(max(n_test, 1), len(train_pool) - 1)
    else:
        n_test = 0

    if n_test > 0:
        test_indices = np.sort(rng.choice(train_pool, size=n_test, replace=False))
        train_indices = np.setdiff1d(train_pool, test_indices, assume_unique=False)
    else:
        test_indices = np.array([], dtype=int)
        train_indices = train_pool

    train_genotypes = genotype_data[:, train_indices].T
    test_genotypes = genotype_data[:, test_indices].T

    prediction_indices = np.arange(coordinates.shape[0], dtype=int)
    prediction_genotypes = genotype_data[:, prediction_indices].T

    train_coords = coordinates[train_indices]
    test_coords = coordinates[test_indices]

    print(f"Training samples: {len(train_indices)}, Test samples: {len(test_indices)}")
    print(f"Training genotypes: {len(train_genotypes)}, Test genotypes: {len(test_genotypes)}")
    print(f"Training coordinates: {len(train_coords)}, Test coordinates: {len(test_coords)}")
    print(f"Prediction samples: {len(prediction_indices)}, Prediction genotypes: {len(prediction_genotypes)}")

    return (
        train_indices,
        test_indices,
        train_genotypes,
        test_genotypes,
        train_coords,
        test_coords,
        prediction_indices,
        prediction_genotypes,
    )


def build_neural_network(input_shape, num_layers, layer_width, dropout_rate, output_dim=2):
    def euclidean_loss(y_true, y_pred):
        return K.sqrt(K.sum(K.square(y_pred - y_true), axis=-1))

    tf.keras.backend.clear_session()

    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Input(shape=(input_shape,)))
    model.add(tf.keras.layers.BatchNormalization())

    for _ in range(int(np.floor(num_layers / 2))):
        model.add(tf.keras.layers.Dense(layer_width, activation="elu"))

    model.add(tf.keras.layers.Dropout(dropout_rate))

    for _ in range(int(np.ceil(num_layers / 2))):
        model.add(tf.keras.layers.Dense(layer_width, activation="elu"))

    model.add(tf.keras.layers.Dense(output_dim))
    model.compile(optimizer="Adam", loss=euclidean_loss)

    return model


def train_neural_network(
    model,
    train_genotypes,
    test_genotypes,
    train_coords,
    test_coords,
    max_epochs,
    batch_size,
    checkpoint,
    early_stop,
    reduce_lr,
    output_prefix,
    run_id
):
    callbacks = [checkpoint, early_stop, reduce_lr]

    if len(test_genotypes) > 0 and len(test_coords) > 0:
        history = model.fit(
            train_genotypes,
            train_coords,
            epochs=max_epochs,
            batch_size=batch_size,
            shuffle=True,
            verbose=1,
            validation_data=(test_genotypes, test_coords),
            callbacks=callbacks
        )
    else:
        history = model.fit(
            train_genotypes,
            train_coords,
            epochs=max_epochs,
            batch_size=batch_size,
            shuffle=True,
            verbose=1,
            callbacks=callbacks
        )

    model_path = os.path.join(output_prefix, f"{run_id}_weights.hdf5")
    print(f"Training complete. Loading best weights from: {model_path}")
    model.load_weights(model_path)

    return history, model


def _haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return R * c


def decode_predictions_to_latlon(predictions, transform_info):
    target_mode = transform_info["target_mode"]

    if target_mode == "latlon":
        latlon = np.array([
            [x[0] * transform_info["std_x"] + transform_info["mean_x"],
             x[1] * transform_info["std_y"] + transform_info["mean_y"]]
            for x in predictions
        ], dtype=float)
        return latlon

    elif target_mode == "sphere":
        return sphere_to_latlon(predictions)

    else:
        raise ValueError(f"Unsupported target_mode: {target_mode}")

def sanitize_lat_lon_array(latlon):
    lat = latlon[:, 0]
    lon = latlon[:, 1]

    # clip latitude
    lat = np.clip(lat, -90, 90)

    # wrap longitude
    lon = ((lon + 180) % 360) - 180

    return np.column_stack([lat, lon])

def predict_sample_locations(
    model,
    genotype_predictions,
    transform_info,
    test_locations,
    prediction_indices,
    sample_ids,
    test_genotypes,
    output_prefix,
    run_id,
    output_file,
    training_history,
    verbose=True
):
    if verbose:
        print("Predicting sample locations...")

    os.makedirs(output_prefix, exist_ok=True)

    predictions = model.predict(genotype_predictions, verbose=0)
    #predictions_latlon = decode_predictions_to_latlon(predictions, transform_info)
    predictions_latlon = decode_predictions_to_latlon(predictions, transform_info)
    predictions_latlon = sanitize_lat_lon_array(predictions_latlon)

    predicted_df = pd.DataFrame(predictions_latlon, columns=["latitude", "longitude"])
    predicted_df["sampleID"] = np.array(sample_ids)[prediction_indices]
    predicted_df.to_csv(os.path.join(output_prefix, f"{run_id}_predicted_locations.txt"), index=False)

    if len(test_genotypes) > 0 and len(test_locations) > 0:
        test_locations_latlon = decode_predictions_to_latlon(test_locations, transform_info)

        test_predictions = model.predict(test_genotypes, verbose=0)
        #test_predictions_latlon = decode_predictions_to_latlon(test_predictions, transform_info)
        test_predictions_latlon = decode_predictions_to_latlon(test_predictions, transform_info)
        test_predictions_latlon = sanitize_lat_lon_array(test_predictions_latlon)

        if len(test_predictions_latlon) >= 2:
            r2_latitude = np.corrcoef(test_predictions_latlon[:, 0], test_locations_latlon[:, 0])[0, 1] ** 2
            r2_longitude = np.corrcoef(test_predictions_latlon[:, 1], test_locations_latlon[:, 1])[0, 1] ** 2
        else:
            r2_latitude = np.nan
            r2_longitude = np.nan

        distance_errors = [
            _haversine_km(
                test_predictions_latlon[i, 0], test_predictions_latlon[i, 1],
                test_locations_latlon[i, 0], test_locations_latlon[i, 1]
            )
            for i in range(len(test_predictions_latlon))
        ]
        mean_error = float(np.mean(distance_errors)) if len(distance_errors) > 0 else np.nan
        median_error = float(np.median(distance_errors)) if len(distance_errors) > 0 else np.nan
    else:
        distance_errors = []
        r2_latitude = np.nan
        r2_longitude = np.nan
        mean_error = np.nan
        median_error = np.nan

    if verbose:
        output_file.write(
            f"R2(latitude)={r2_latitude}\tR2(longitude)={r2_longitude}\t"
            f"Mean validation error {mean_error}\tMedian validation error {median_error}\n"
        )
        print(
            f"R2(latitude)={r2_latitude}\n"
            f"R2(longitude)={r2_longitude}\n"
            f"Mean validation error {mean_error}\n"
            f"Median validation error {median_error}\n"
        )

    history_df = pd.DataFrame(training_history.history)
    history_df.to_csv(
        os.path.join(output_prefix, f"{run_id}_training_history.txt"),
        sep="\t",
        index=False
    )

    return distance_errors, predicted_df, r2_longitude, r2_latitude, mean_error, median_error


def main():
    parser = argparse.ArgumentParser(description="Command-line tool for genomic data processing.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    parser_convert = subparsers.add_parser("convert", help="Convert VCF to Zarr format.")
    parser_convert.add_argument("--vcf", type=str, required=True, help="Path to the input VCF file")
    parser_convert.add_argument("--zarr", type=str, required=True, help="Path to the output Zarr directory")

    parser_check = subparsers.add_parser("check", help="Quality check of GPS coordinates.")
    parser_check.add_argument("--gps_data", type=str, required=True, help="Path to GPS data file")
    parser_check.add_argument("--output_folder", type=str, required=True, help="Folder for output files")

    min_minor_allele_count = 2
    apply_imputation = True
    train_ratio = 0.75
    num_layers, layer_width, dropout_rate = 10, 256, 0.25
    max_epochs, batch_size = 200, 32
    num_runs = 10
    run_id = "test"

    parser_filter = subparsers.add_parser("snp2gps", help="Start SNP2GPS_private run.")
    parser_filter.add_argument("--path_zarr", type=str, required=True, help="Path to Zarr file")
    parser_filter.add_argument("--gps_data", type=str, required=True, help="Path to GPS data file")
    parser_filter.add_argument("--output_folder", type=str, required=True, help="Folder for output files")
    parser_filter.add_argument("--run_id", type=str, default=run_id, help="Run ID for logging")
    parser_filter.add_argument("--min_minor_allele_count", type=int, default=min_minor_allele_count,
                               help="Minimum minor allele count for filtering")
    parser_filter.add_argument("--apply_imputation", dest="apply_imputation", action="store_true",
                               help="Enable imputation for missing values")
    parser_filter.add_argument("--no_imputation", dest="apply_imputation", action="store_false",
                               help="Disable imputation for missing values")
    parser_filter.set_defaults(apply_imputation=apply_imputation)
    parser_filter.add_argument("--train_ratio", type=float, default=train_ratio, help="Train/test split ratio")
    parser_filter.add_argument("--num_layers", type=int, default=num_layers, help="Number of neural network layers")
    parser_filter.add_argument("--layer_width", type=int, default=layer_width, help="Width of each layer")
    parser_filter.add_argument("--dropout_rate", type=float, default=dropout_rate,
                               help="Dropout rate for regularization")
    parser_filter.add_argument("--max_epochs", type=int, default=max_epochs, help="Maximum number of training epochs")
    parser_filter.add_argument("--batch_size", type=int, default=batch_size, help="Batch size during training")
    parser_filter.add_argument("--num_runs", type=int, default=num_runs, help="Number of runs with different seeds")
    parser_filter.add_argument("--validation_size", type=int, default=None,
                               help="Number of samples to include in each validation set")
    parser_filter.add_argument("--validation_runs", type=int, default=None,
                               help="Number of validation sets to generate")
    parser_filter.add_argument("--use_saved_npz", action="store_true",
                               help="Load genotype_array and samples from a saved NPZ file instead of Zarr")
    parser_filter.add_argument("--genotype_npz_path", type=str, default=None,
                               help="Path to NPZ file containing genotype_array and samples")
    parser_filter.add_argument("--target_mode", type=str, default="latlon", choices=["latlon", "sphere"],
                               help="Prediction target mode: latlon (default) or sphere")

    args = parser.parse_args()

    if args.command == "convert":
        vcf_to_zarr(args.vcf, args.zarr)

    elif args.command == "check":
        run_preprocessing_pipeline(
            args.gps_data,
            "ID",
            "LATITUDE",
            "LONGITUDE",
            "COUNTRY_NAME",
            args.output_folder,
            "\t"
        )

    elif args.command == "snp2gps":
        snp2gps_launcher(
            path_zarr=args.path_zarr,
            gps_data=args.gps_data,
            output_folder=args.output_folder,
            run_id=args.run_id,
            min_minor_allele_count=args.min_minor_allele_count,
            apply_imputation=args.apply_imputation,
            train_ratio=args.train_ratio,
            num_layers=args.num_layers,
            layer_width=args.layer_width,
            dropout_rate=args.dropout_rate,
            max_epochs=args.max_epochs,
            batch_size=args.batch_size,
            num_runs=args.num_runs,
            validation_size=args.validation_size,
            validation_runs=args.validation_runs,
            genotype_npz_path=args.genotype_npz_path,
            use_saved_npz=args.use_saved_npz,
            target_mode=args.target_mode
        )


if __name__ == "__main__":
    main()