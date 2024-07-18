# Traffic Incident Severity Classification

This repository contains scripts and data for analyzing traffic incident severity using various transformer-based language models and machine learning techniques. The repository is organized into several folders to handle different datasets (UK, Queensland, USA, USA with using  descriptions only), and consists of scripts for feature extraction and model evaluation. Below is an overview of the repository’s structure and detailed instructions on how to use the provided scripts.

## Repository Structure

```plaintext
.
├── USA
│   ├── B-encode-USA-wo-desc.ipynb
│   ├── B-encode-USA.ipynb
│   ├── C-eval-USA.ipynb
│   ├── compress_weights_npy_npz.py
│   ├── results-LLM-flat-even-severity-final-nodesc.csv
│   ├── results-LLM-flat-even-severity-final.csv
│   ├── S-hist.pdf
│   ├── subset-USA-even-bin.csv
│   ├── subset-USA-even-fulltext.csv
├── Q
│   ├── B-encode-Q.ipynb
│   ├── C-eval-Q.ipynb
│   ├── compress_weights_npy_npz.py
│   ├── results-LLM-32-even-severity-final-BL-nonorm.csv
│   ├── results-LLM-32-even-severity-final-Q.csv
│   ├── results-LLM-32-even-severity-flat.csv
│   ├── S-hist.pdf
│   ├── q-merged-sampled-bin.csv
│   ├── q-merged-sampled-fulltext.csv
├── UK
│   ├── B-encode-UK.ipynb
│   ├── C-eval-UK.ipynb
│   ├── compress_weights_npy_npz.py
│   ├── results-LLM-32-even-severity-final-UK.csv
│   ├── S-hist.pdf
│   ├── UK-results-LLM-32-even-severity-final.csv
│   ├── dft-merged-sampled-bin.csv
│   ├── dft-merged-sampled-fulltext.csv
├── DESC-only
│   ├── B-encode-usa-desc-speed.ipynb
│   ├── B-encode-usa-desc.ipynb
│   ├── C-eval-desc-only-PCA-components.ipynb
│   ├── C-eval-desc-only.ipynb
│   ├── DESC-results-LLM-flat-even-severity.csv
│   ├── inference_samples_per_minute.pdf
│   ├── language_models_samples_per_minute.pdf
│   ├── S-hist.pdf
│   ├── tokenization_samples_per_minute.pdf
│   ├── total_samples_per_minute.pdf
│   ├── subset-USA-even-bin.csv
│   ├── subset-USA-even-fulltext.csv
│   ├── subset-USA-even.csv
├── results_csv
│   ├── DESC-results-LLM-flat-even-severity.csv
│   ├── Q-results-LLM-32-even-severity-final.csv
│   ├── UK-results-LLM-32-even-severity-final.csv
│   ├── USA-results-LLM-32-even-severity-final.csv
├── LICENSE
```

## Setup

### Prerequisites

To run the scripts, you need to have Python installed. We recommend creating a virtual environment to manage dependencies. Install required packages using pip.

```sh
pip install -r requirements.txt
```

### Directory Structure

Ensure that you have the following structure, with folders corresponding to each dataset (UK, USA, DESC-only), each containing the necessary scripts and data files.

## How to Use

### 1. Feature Extraction

The `B-encode-*.ipynb` notebooks are designed for feature extraction using transformer-based models.

#### Steps:
1. Open one of the `B-encode-*.ipynb` notebooks (e.g., `B-encode-UK.ipynb`).
2. Ensure you have the corresponding `subset-*.csv` file in the same directory.
3. Run the notebook cells sequentially. The process will:
   - Load and preprocess the data.
   - Set up transformer models.
   - Extract features from the text.
   - Save the extracted features as `.npz` files.

### 2. Model Evaluation

The `C-eval-*.py` scripts are intended for the evaluation of machine learning models using the extracted features.

#### Steps:
1. Run the evaluation script from the terminal:
   ```sh
   python C-eval-[dataset].py
   ```
   Example for the UK dataset:
   ```sh
   python C-eval-UK.py
   ```
2. This script will:
   - Load the previously saved features.
   - Perform Exploratory Data Analysis (EDA).
   - Set up and train various machine learning models.
   - Conduct cross-validation.
   - Save the results to `results_csv` directory.

### 3. Results

After running the evaluation scripts, results will be saved in the `results_csv` directory within each dataset folder. These results include performance metrics such as accuracy, F1 score, precision, and recall for classification tasks, and appropriate metrics for regression tasks.

### Visualizations

The scripts also generate visualizations to help in understanding the model performance and feature importance. The visualizations are saved in the appropriate directories.

## Detailed Description of Scripts

### B-encode-*.ipynb

This Jupyter Notebook contains:
- Loading and preprocessing of text data.
- Setup and loading of transformer-based pretrained models (BERT, GPT-2, RoBERTa, etc.).
- Extraction of features from the text data.
- Timing metrics for processing.
- Saving extracted features.

### C-eval-*.py

This Python script includes:
- Loading previously extracted features and the dataset.
- Exploratory Data Analysis.
- Setting up machine learning models (LightGBM, KNN, RandomForest, etc.).
- Cross-validation for classification and regression tasks.
- Evaluation using various metrics (Accuracy, F1 score, Precision, Recall, RMSE, MAPE).
- Saving performance results to CSV files.
- Visualizing feature importances and distributions.

## Notes

- Ensure that your data files are named correctly (`subset-*.csv`) and placed in the appropriate directories.
- Adjust paths in the scripts if needed based on your local setup.
- Use a GPU for faster processing of transformer models, if available.
