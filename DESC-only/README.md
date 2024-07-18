The B-encode-usa-desc.ipynb Python script is designed to perform the following tasks:

1. **Load and Preprocess Data**: 
   - It reads a CSV file (`subset-*.csv`) containing textual data using the `pandas` library, extracts the "FullText" and "Description" columns, and then displays the data.
  
2. **Set Up and Load Transformer-based Pretrained Models**:
   - It defines a function `load_model` that accepts a model name and loads the corresponding tokenizer and model from the Hugging Face `transformers` library.
   - The supported models include various versions of BERT, GPT-2, RoBERTa, XLNet, and mT5.

3. **Feature Extraction Using Pretrained Models**:
   - It has a function `extract_features` which takes a list of texts, a model name, and a batch size, then encodes and tokenizes the texts.
   - The function processes the text in batches to avoid memory issues and utilizes GPU if available.
   - The average time taken per batch and the total processing time for extracting features from the text are calculated and printed.

4. **Batch Processing**:
   - Features for all the texts are extracted using the specified pretrained model, and the features are saved as a NumPy array for future use.

5. **Statistics and Visualization**:
   - The script computes and prints the duration distribution statistics of the texts (such as percentiles and Empirical Cumulative Distribution Function - ECDF) using the `Duration` column.
   - The ECDF is plotted for visualizing the duration distribution.

6. **Main Workflow**:
   - A list of model names is defined, and for each model, the features of the text are extracted and saved. The models specified include 'distilroberta' and 'bert-cased'.
   
7. **Potential for Extension**:
   - The script can be extended by adding more models to the list and processing more types of texts. Additionally, other statistical analyses and visualizations can be added.



The B-encode-usa-desc-speed.py Python script is designed to perform multiple tasks with a focus on text analysis and feature extraction using various transformer-based pretrained models. Here's an overview of its functionality:

1. **Load and Preprocess Data**:
   - It reads a CSV file (`subset-LA-even.csv`) using `pandas`, creating a DataFrame `dt`. This DataFrame is assumed to contain text data, particularly the columns "FullText" and "Description".

2. **Set Up and Load Transformer-based Pretrained Models**:
   - The `load_model` function accepts a model name and loads the corresponding tokenizer and model from Hugging Face's `transformers` library. Supported models include various versions like BERT, GPT-2, RoBERTa, XLNet, mT5, and ALBERT.
   - Models are moved to a GPU if available using the `torch` library.

3. **Feature Extraction Using Pretrained Models**:
   - The `extract_features` function tokenizes and encodes a list of texts, then processes them in batches to avoid memory issues.
   - It supports models like BERT, GPT-2, RoBERTa, XLNet, and mT5, extracting features for these texts while tracking the time taken for each batch.

4. **Measure and Print Timing Statistics**:
   - The script measures average and total time taken for tokenizing and inferring text features. These timings are collected and printed for each transformer model.
   - It also calculates and prints the average time per batch, per sample, and samples processed per second and per minute.

5. **Save Extracted Features**:
   - The function `extract_and_save_features` uses `extract_features` to get features of the text and saves them as NumPy arrays for later use.

6. **Batch Processing Across Different Models**:
   - `extract_features_average_batch` performs feature extraction on multiple batches and computes the timing statistics for each batch, per sample, and overall.
   - The script iterates over a list of predefined model names, performs feature extraction, and stores the timing results in a DataFrame.

7. **Statistical Analysis and Visualization**:
   - Empirical Cumulative Distribution Function (ECDF) is used to visualize the duration distribution in the dataset. It computes and plots the ECDF for the 'Duration' column in the DataFrame.
   - Percentile tables for the 'Duration' column are computed and printed for more detailed statistical insights.

8. **Visualization of Performance Metrics**:
   - After collecting timing metrics, the script uses `seaborn` and `matplotlib` for visualizing the samples processed per minute for each model in bar plots.
   - It produces and saves PDF files for these visualizations to provide a clear comparison of the models' performance.



The C-eval-desc-only.py Python script is designed for machine learning tasks involving text data and additional non-textual features. It utilizes various transformer-based language models to extract features and employs multiple machine learning models for classification and regression tasks. Here is an overall description of the code:

### 1. **Loading and Preparing Feature Data**

- **Loading Features**: The `load_features` function loads saved features extracted previously from different language models and stored in `.npz` files.
- **Reading Dataset**: The script reads a dataset from `subset-LA-even.csv` which contains accident-related data with columns such as "FullText", "Description", "Severity", "Duration", and others.

### 2. **Exploratory Data Analysis (EDA)**

- **Plotting Severity Distribution**: The script generates a histogram showing the distribution of accident severity classes.
- **Cross-Validation Procedure**: The `cross_validate_classification` function performs cross-validation and calculates accuracy, F1 score, precision, and recall scores for classification models.
  
### 3. **Model Initialization and Feature Extraction**

- **Loading Multiple Models**: The script supports multiple models such as LightGBM, K-Nearest Neighbors, RandomForestClassifier, XGBoostClassifier, etc.
- **Principal Component Analysis (PCA)**: The `perform_pca` function is used to reduce the dimensionality of the extracted features from language models.
- **Standardization**: Features are standardized using `StandardScaler` from `sklearn` to ensure better performance of ML models.

### 4. **Training and Evaluation**

- **Cross-Validation**: The script performs cross-validation for each combination of language model features and machine learning models. It evaluates combinations like "NLP features", "Report features", and combined "NLP + Report features".
- **Evaluation Metrics**: The results are evaluated using metrics like accuracy, F1 score, precision, and recall.

### 5. **Saving and Displaying Results**

- **Results DataFrame**: The script stores the evaluation results in a DataFrame and writes it to a CSV file (`DESC-results-LLM-flat-even-severity-2.csv`).
- **Improved Readability for Results**: The script customizes the calculated averages and results to make the interpretation easy by converting the internal representations into human-readable formats.

### 6. **Visualization and Feature Importance**

- **Feature Importance Plot**: It shows the importance of various features using RandomForestClassifier's feature importance attribute.
- **Histogram of Duration**: Another histogram shows the distribution of accident durations.
  
### 7. **Regression Tasks**

- **Model Setup**: For regression tasks, various models like Linear Regression, ElasticNet, DecisionTree, XGBoost, etc., are used.
- **Cross-Validation for Regression**: The script performs K-Fold Cross-validation for regression models, evaluating with RMS Error and Mean Absolute Percentage Error (MAPE).

### 8. **Result Aggregation and File Outputs**

- **Aggregating Results**: It calculates the average performance metrics across different folds and stores these in a DataFrame, then saves the results to disk.
- **Adjust Result Naming**: The script adjusts the names of feature sets to improve the readability of results.
  
### 9. **Handling DataFrame Operations**

- **Data Preprocessing**: It preprocesses the DataFrame by handling missing values, encoding categorical variables using LabelEncoder, and transforms boolean columns to integers.