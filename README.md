# SNP2GPS

![SNP2GPS Logo](https://github.com/snowformatics/SNP2GPS/blob/37b24ed5a46ad5e75ea5fbb47651088a977fd311/logo.PNG)

SNP2GPS is a bioinformatics tool for genomic data processing. It enables the conversion of VCF files to Zarr format, handles genotype and phenotype data, and performs machine learning–based prediction of sample geographic locations based on SNP data. The tool includes data processing, imputation, filtering, model training using neural networks, validation, visualization, and generation of an HTML report.

---

## 🚀 **Features**

- Convert VCF files to Zarr and preprocess genotype/GPS data
- Filter SNPs and impute missing genotypes
- GPS quality check
- Train neural networks to predict geographic locations in latlon or sphere mode
- Evaluate predictions with random validation sets and GPS quality checks
- Generate visualizations, interactive maps, and an HTML summary report
- Fully implemented in Python with a simplified command-line interface
- Includes predicted-vs-actual plots, model statistics, outlier handling, confidence intervals, and land-correction summaries
---

## 📦 **Installation**

### 1. **Create a Conda Environment**

It is recommended to create a clean Conda environment to avoid conflicts:

```bash
conda remove -n geo_tf_env --all -y
conda create -n geo_tf_env python=3.9 pip -y
conda activate geo_tf_env
```

### 2. **Install Dependencies**

```bash
pip install numpy==1.26.4
pip install pandas scipy matplotlib tensorflow==2.10.1 h5py zarr numcodecs scikit-allel geopandas shapely pyproj pyogrio fiona folium scikit-learn
```

---

## 🛠️ **Usage**

### 1. **Convert VCF to Zarr**

```bash
python snp2gps.py convert --vcf path/to/input.vcf.gz --zarr path/to/output_folder/
```

### 2. **Check GPS Coordinates**

```bash
python snp2gps.py check --gps_data path/to/gps_data.txt --output_folder path/to/output/
```

### 3. **Run SNP2GPS**

```bash
python snp2gps.py snp2gps \
    --path_zarr path/to/input.zarr \
    --gps_data path/to/gps_data.txt \
    --output_folder path/to/output/ \
    --run_id test_run \
    --min_minor_allele_count 2 \
    --apply_imputation \
    --train_ratio 0.75 \
    --num_layers 10 \
    --layer_width 256 \
    --dropout_rate 0.25 \
    --max_epochs 200 \
    --batch_size 32 \
    --num_runs 10 \
    --validation_size 100 \
    --validation_runs 5 \
    --target_mode latlon
```

### 4. **Run SNP2GPS without Imputation**

```bash
python snp2gps.py snp2gps \
    --path_zarr path/to/input.zarr \
    --gps_data path/to/gps_data.txt \
    --output_folder path/to/output/ \
    --no_imputation
```

### 5. **Run SNP2GPS using a saved NPZ file**

```bash
python snp2gps.py snp2gps \
    --gps_data path/to/gps_data.txt \
    --output_folder path/to/output/ \
    --use_saved_npz \
    --genotype_npz_path path/to/genotypes.npz
```

---

## 📋 **Arguments**

### `convert`

| Argument | Type | Description | Default |
|----------|------|-------------|---------|
| `--vcf` | str | Path to the input VCF file | None |
| `--zarr` | str | Path to the output Zarr directory | None |

### `check`

| Argument | Type | Description | Default |
|----------|------|-------------|---------|
| `--gps_data` | str | Path to GPS data file | None |
| `--output_folder` | str | Folder for output files | None |

### `snp2gps`

| Argument | Type | Description | Default |
|----------|------|-------------|---------|
| `--path_zarr` | str | Path to Zarr file | None |
| `--gps_data` | str | Path to GPS data file | None |
| `--output_folder` | str | Folder for output files | None |
| `--run_id` | str | Unique ID for the run | `"test"` |
| `--min_minor_allele_count` | int | Minimum minor allele count for filtering | `2` |
| `--apply_imputation` | flag | Enable imputation for missing values | enabled by default |
| `--no_imputation` | flag | Disable imputation for missing values | disabled |
| `--train_ratio` | float | Train/test split ratio | `0.75` |
| `--num_layers` | int | Number of neural network layers | `10` |
| `--layer_width` | int | Width of each layer | `256` |
| `--dropout_rate` | float | Dropout rate for regularization | `0.25` |
| `--max_epochs` | int | Maximum number of training epochs | `200` |
| `--batch_size` | int | Batch size during training | `32` |
| `--num_runs` | int | Number of runs with different seeds | `10` |
| `--validation_size` | int | Number of samples in each validation set | `None` |
| `--validation_runs` | int | Number of validation sets to generate | `None` |
| `--use_saved_npz` | flag | Load genotype array and samples from a saved NPZ file instead of Zarr | `False` |
| `--genotype_npz_path` | str | Path to NPZ file containing genotype array and samples | `None` |
| `--target_mode` | str | Prediction target mode: `latlon` or `sphere` | `"latlon"` |

### `build_full_report.py`

| Argument | Type | Description | Default |
|----------|------|-------------|---------|
| `--predictions_csv` | str | Path to `final_averaged_predictions.csv` | required |
| `--metadata_tsv` | str | Path to metadata TSV with `sampleID`, `x`, and `y` | required |
| `--run_folder` | str | Validation run folder, for example `output/0` | required |
| `--performance_txt` | str | Optional path to `final_model_performance.txt` | `None` |
| `--report_name` | str | Output HTML report filename | `"gps_report.html"` |

---

## 🧪 **Examples**

### Example 1: Convert VCF to Zarr

```bash
python snp2gps.py convert --vcf example.vcf.gz --zarr example_output/
```

### Example 2: Run SNP2GPS with default parameters

```bash
python snp2gps.py snp2gps \
    --path_zarr example_output/example.zarr \
    --gps_data example_gps.txt \
    --output_folder results/
```

### Example 3: Run SNP2GPS with validation sets

```bash
python snp2gps.py snp2gps \
    --path_zarr example_output/example.zarr \
    --gps_data example_gps.txt \
    --output_folder results/ \
    --validation_size 100 \
    --validation_runs 5
```

### Example 4: Run SNP2GPS in sphere mode

```bash
python snp2gps.py snp2gps \
    --path_zarr example_output/example.zarr \
    --gps_data example_gps.txt \
    --output_folder results/ \
    --target_mode sphere
```

### Example 5: Build the HTML report (generated by default)

```bash
python build_full_report.py \
    --predictions_csv results/final_averaged_predictions.csv \
    --metadata_tsv example_gps.txt \
    --run_folder results/0
```
---
## 🧭 GPS Quality Check (`check` command)

SNP2GPS includes a preprocessing step to **validate and clean GPS metadata** before model training.

### What it does

The `check` command runs a quality control pipeline on your GPS file to ensure that coordinates are consistent, valid, and usable for training.

It performs:

- Validation of required columns (`sampleID`, coordinates, metadata)  
- Detection of missing or invalid coordinates  
- Conversion of coordinate formats to numeric values  
- Identification of problematic entries (e.g. swapped latitude/longitude)  
- Basic consistency checks between samples and metadata  

### Why this is important

Accurate geographic prediction strongly depends on the quality of input coordinates. Errors in GPS data can lead to:

- Incorrect model training  
- Inflated prediction errors  
- Misleading validation results  

Running the GPS check step ensures your dataset is **clean, aligned, and reliable** before training.

---

# Creating the Validation Set

## How the Validation Set is Created

The validation set in `snp2gps` is generated through **random sampling** based on two parameters:

- **`--validation_size`** → Number of samples per validation set  
- **`--validation_runs`** → Number of validation sets  

### Process

1. Randomly select `validation_size` samples from the dataset  
2. Repeat this process `validation_runs` times  
3. Each validation set is independent of the train/test split  

## Why Create Multiple Validation Sets?

- **More reliable evaluation** → reduces bias from a single split  
- **Better model tuning** → improves hyperparameter optimization  
- **Stable performance estimates** → reduces variance due to sampling  

---

## 🌍 Sphere Mode (Geographic Representation)

SNP2GPS supports two target representations for geographic prediction:

- `latlon` (default) → predicts normalized latitude and longitude  
- `sphere` → predicts 3D coordinates on the unit sphere  

### Why use sphere mode?

Geographic coordinates (latitude/longitude) have inherent limitations:
- Discontinuities at ±180° longitude  
- Distortions near the poles  
- Non-Euclidean distances on a spherical surface  

To address these issues, **sphere mode** transforms coordinates into 3D Cartesian space:

- Latitude/longitude → *(x, y, z)* on a unit sphere  
- Enables smooth, continuous learning without boundary artifacts  
- Provides a more natural representation for global-scale prediction  

### When to use it

- Global datasets spanning large geographic regions  
- Data crossing longitude boundaries (e.g. ±180°)  
- Samples near polar regions  
- When training stability or convergence is an issue  

### How to enable

```bash
--target_mode sphere
```
---

## 📄 **Input Data Format**

### 1. **VCF File**

The VCF file should follow the standard format:

```text
##fileformat=VCFv4.2
##source=MyTool
#CHROM  POS     ID      REF     ALT     QUAL    FILTER  INFO    FORMAT  SAMPLE1 SAMPLE2 ...
1       10176   rs541   A       C       .       .       .       GT      0/0     0/1 ...
```

### 2. **GPS Data File**

Tab-separated `.txt` file with required columns:

- **sampleID**  
- **x**  
- **y**  

Example:

```text
sampleID	x	y
ERX2583486	31.5	121.09
ERX2583566	35.46	116.48
ERX2583489	31.4	120.17
```

---

## 📊 **Output Files**

Depending on the selected workflow, SNP2GPS can generate:

- Zarr genotype data  
- Predicted locations tables  
- Training history files  
- Model weight files  
- Validation set files  
- Performance summary files  
- Static plots in PNG format  
- Interactive maps in HTML format  
- A full HTML report summarizing model performance and predictions  

### HTML report contents

The HTML report can include:

- Overview cards with model metrics  
- Training history plot and learning-rate schedule  
- Prediction map for all samples  
- Training and predicted heatmaps  
- Validation comparison map  
- Validation error histogram  
- Cumulative validation error curve  
- Validation distance bar chart  
- Latitude and longitude scatter plots  
- Land-correction summary  
- Land-shift histogram  
- Country summary  
- Validation preview table  

---

## 🧬 **References**

> Battey CJ, Ralph PL, Kern AD (2020)  
> Predicting geographic location from genetic variation with deep neural networks  
> https://doi.org/10.7554/eLife.54507  

---

## 📄 **License**

This project is licensed under the GNU General Public License v3.0 (GPL-3.0)

---

## 👩‍💻 **Contributors**

- [Stefanie Lueck](https://github.com/snowformatics)
