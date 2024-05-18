
# AnalysisBot

AnalysisBot is a comprehensive data analysis project focused on customer support case data. I built this as a project to learn more about machine learning and data science using python and jupyter. It includes data loading, preprocessing, exploratory data analysis (EDA), feature engineering, text processing, and predictive modeling using machine learning techniques. The project utilizes various Python libraries and tools to extract insights and build predictive models.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Features](#features)
- [Functionality Details](#functionality-details)
  - [Data Loading and Preprocessing](#data-loading-and-preprocessing)
  - [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
  - [Feature Engineering and Text Processing](#feature-engineering-and-text-processing)
  - [Correlation Analysis](#correlation-analysis)
  - [Predictive Modeling](#predictive-modeling)
  - [Advanced Analysis](#advanced-analysis)
- [Contributing](#contributing)
- [License](#license)

## Installation

To get started with this project, you'll need to install the necessary dependencies: 

- **pandas: For data manipulation and analysis.**
- **numpy: For numerical operations.**
- **matplotlib: For plotting and visualization.**
- **seaborn: For statistical data visualization.**
- **scikit-learn: For machine learning algorithms and tools.**
- **spacy: For Natural Language Processing (NLP).**
- **wordcloud: For generating word cloud visualizations.**
- **textblob: For text processing and sentiment analysis.**
- **plotly: For interactive visualizations.**
- **bokeh: For interactive visualizations.**
- **faker: For generating fake data.**
- **scipy: For scientific and technical computing (used for correlation with categorical data).**

You can install them using the following commands:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn spacy wordcloud textblob plotly bokeh faker
python -m spacy download en_core_web_md
```
Or within the notebook by creating a new cell above the first with the following content:
```bash
# Install necessary libraries
!pip install pandas
!pip install numpy
!pip install matplotlib
!pip install seaborn
!pip install scikit-learn
!pip install spacy
!pip install wordcloud
!pip install textblob
!pip install plotly
!pip install bokeh
!pip install faker
!pip install scipy

# Install the larger spaCy model
!python -m spacy download en_core_web_md
```


## Usage

1. Clone the repository:
    ```bash
    git clone https://github.com/thejerrod/analysisBot.git
    cd analysisBot
    ```

2. Open the Jupyter notebook:
    ```bash
    jupyter notebook analysisBot.ipynb
    ```

3. Run the cells sequentially to perform data analysis and machine learning tasks.

## Features

- **Data Loading and Preprocessing**: Load data from `values.csv`, handle missing values, and preprocess the data. faker is used to generate values if values.csv is missing
- **Exploratory Data Analysis (EDA)**: Perform initial data exploration, visualization, and summary statistics.
- **Feature Engineering**: Generate new features and process text data using NLP techniques.
- **Correlation Analysis**: Compute and visualize various types of correlation matrices.
- **Predictive Modeling**: Train and evaluate machine learning models to predict resolution times.
- **Advanced Analysis**: Perform clustering, anomaly detection, dimensionality reduction, and feature importance analysis.

## Functionality Details

### Data Loading and Preprocessing

- **Loading Data**: The project starts by loading the dataset from `values.csv`. This dataset contains various fields related to customer support cases, such as product type, product version, customer description, engineer description, root cause, and more.
  
- **Handling Missing Values**: Missing values are handled by filling them with placeholders or by removing columns that contain only one unique value, ensuring that the data is clean and ready for analysis.

- **Initial Data Exploration**: Summary statistics and initial rows of the data are displayed to understand the basic structure and contents of the dataset.

### Exploratory Data Analysis (EDA)

- **Distribution Plots**: Visualize the distribution of categorical and numerical data using count plots and histograms to understand the frequency and spread of different variables.

- **Group Analysis**: Group the data by specific columns (e.g., Reason for contact) and calculate aggregate statistics (e.g., mean time to resolve) to identify patterns and insights.

### Feature Engineering and Text Processing

- **N-grams Extraction**: Identify frequent bigrams and trigrams in textual data (e.g., customer descriptions) to generate patterns that can be used for further analysis.

- **NLP Techniques**: Perform various Natural Language Processing (NLP) tasks including:
  - **POS Tagging**: Identify parts of speech in customer descriptions.
  - **Dependency Parsing**: Analyze grammatical relationships between words.
  - **Noun Phrase Extraction**: Extract key noun phrases.
  - **Text Summarization**: Generate summaries of customer descriptions.
  - **Sentiment Analysis**: Assess the sentiment of customer descriptions using TextBlob.
  - **Entity Recognition**: Identify named entities using spaCy.
  - **Text Cleaning**: Clean text data by removing stop words and punctuation.

### Correlation Analysis

- **Pearson Correlation Matrix**: Calculate and visualize the Pearson correlation coefficients between numerical variables to identify linear relationships.
  
- **Spearman and Kendall Correlation Matrices**: Compute Spearman and Kendall correlation coefficients to identify non-linear relationships and rank correlations.
  
- **Categorical Correlations**: Compute Cramér's V to measure the association between categorical variables and visualize the results in a heatmap.

### Predictive Modeling

- **Model Training**: Train a RandomForestRegressor model using the numerical features of the dataset to predict the resolution time of support cases.

- **Model Evaluation**: Evaluate the model's performance using Mean Squared Error (MSE) to understand its accuracy and predictive power.

### Advanced Analysis

- **Clustering**: Apply KMeans clustering to group similar support cases and visualize the clusters using scatter plots.
  
- **Anomaly Detection**: Use IsolationForest to detect anomalies in the dataset, identifying unusual or unexpected cases that may require special attention.

- **Dimensionality Reduction**: Perform Principal Component Analysis (PCA) to reduce the dimensionality of the data for visualization and analysis of high-dimensional features.

- **Feature Importance**: Determine and visualize the most important features for predicting the resolution time using the trained RandomForestRegressor model.

## Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue if you have any suggestions or improvements.

1. Fork the repository.
2. Create your feature branch (`git checkout -b feature/your-feature`).
3. Commit your changes (`git commit -m 'Add some feature'`).
4. Push to the branch (`git push origin feature/your-feature`).
5. Open a pull request.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
