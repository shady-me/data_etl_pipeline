# Data ETL Pipeline

This project implements a configurable Data ETL (Extract, Transform, Load) pipeline designed for preparing datasets for machine learning tasks. The pipeline reads raw data, preprocesses it, and outputs a clean, structured dataset suitable for training and evaluation.

## 📁 Project Structure

```
data_etl_pipeline/
├── config.yaml              # Configuration file for the ETL process
├── dataset.csv              # Sample dataset with 100 records (for testing)
├── logs/
│   └── etl_pipeline.log     # Log file for ETL process
├── data/
│   ├── raw/                 # Raw data input directory
│   └── processed/           # Processed data output directory
└── README.md                # Project documentation
```

## ⚙️ Configuration - `config.yaml`

The pipeline behavior is controlled through a YAML config file, which defines the following:

### Data

- **`source_path`**: Path to the raw CSV dataset.
- **`output_dir`**: Directory to save the processed output.
- **`target_column`**: Name of the target variable.
- **`required_columns`**: List of mandatory columns to validate.

### Preprocessing

- **Numerical Features**: `age`, `income`, `years_of_experience`
- **Categorical Features**: `education`, `marital_status`, `employment_type`
- **Imputation**:
  - Numeric: median
  - Categorical: most frequent
- **Scaler**: StandardScaler (standard normalization)
- **Train-Test Split**: 80% train, 20% test (random_state: 42)
- **Feature Engineering** (Optional):
  - Interaction Features: disabled
  - Polynomial Features: disabled
  - Feature Selection: disabled by default

### Logging

- Logs are stored in `logs/etl_pipeline.log`
- Default logging level: `INFO`

## 📊 Dataset - `dataset.csv`

The sample dataset consists of 100 rows and includes the following fields:

| Column             | Type        | Description                        |
|--------------------|-------------|------------------------------------|
| `age`              | Numeric     | Age of the individual              |
| `income`           | Numeric     | Annual income                      |
| `education`        | Categorical | Education level                    |
| `marital_status`   | Categorical | Marital status                     |
| `employment_type`  | Categorical | Type of employment                 |
| `years_of_experience` | Numeric  | Total years of work experience     |
| `target`           | Numeric     | Target label (binary classification) |

## 🚀 Usage

1. Update the `config.yaml` file with your data paths and preferences.
2. Run the ETL pipeline script (not included here) using Python.
3. Processed data will be saved to the `data/processed` directory.
4. Logs will be stored in `logs/etl_pipeline.log`.

## 📝 Notes

- Ensure all required Python libraries (e.g., pandas, sklearn, yaml) are installed.
- The ETL script should read and apply the configuration defined in `config.yaml`.

## 📬 Contact

For queries or contributions, feel free to reach out or open an issue.