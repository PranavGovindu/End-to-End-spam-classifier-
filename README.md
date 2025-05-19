# DVC ML Pipeline

## Overview

This project implements an end-to-end Machine Learning pipeline using DVC (Data Version Control) for reproducible experimentation. The pipeline performs text classification using TF-IDF features and a Random Forest classifier, with the model exported in ONNX format.

## Pipeline Stages

1. **Data Ingestion** (`src/data_ingestion.py`)
   - Loads and splits the raw data into train/test sets
   - Configurable test size (default: 10%)

2. **Data Preprocessing** (`src/data_preprocessing.py`)
   - Processes raw text data
   - Prepares interim datasets

3. **Feature Engineering** (`src/feature_engineering.py`)
   - Applies TF-IDF vectorization
   - Configurable maximum features (default: 20)

4. **Model Building** (`src/model_building.py`)
   - Trains a Random Forest Classifier
   - Configurable parameters:
     - n_estimators: 22
     - random_state: 2
   - Exports model in ONNX format

5. **Model Evaluation** (`src/model_evaluation.py`)
   - Evaluates model performance
   - Current metrics:
     - Accuracy: 1.0
     - Precision: 1.0
     - Recall: 1.0

## Project Structure

```plaintext
├── data/
│   ├── interim/          # Preprocessed data
│   ├── processed/        # Feature engineered data
│   └── raw/             # Original dataset
├── models/              # Saved models (ONNX format)
├── reports/            # Performance metrics
├── src/               # Source code
└── logs/              # Pipeline execution logs
```

## Installation

1. Clone the repository:

```bash
git clone <repository-url>
cd ML-Pipeline-full
```

1. Install dependencies:

```bash
pip install -r requirements.txt
```

1. Run the pipeline:

```bash
dvc repro  # Run the complete pipeline
# OR
dvc exp run  # Run as an experiment
```

## Configuration

The pipeline parameters can be configured in `params.yaml`:

```yaml
data_ingestion:
  test_size: 0.10
feature_engineering:
  max_features: 20
model_building:
  n_estimators: 22
  random_state: 2
```

## Monitoring and Visualization

- Pipeline metrics are tracked using DVCLive

- View metrics and plots using:

```bash
dvc metrics show
dvc plots show
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License.
