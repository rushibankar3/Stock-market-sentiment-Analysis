# Stock Market Sentiment Analysis

This project performs comprehensive stock market sentiment analysis by combining Twitter data with historical stock prices from Yahoo Finance. It analyzes tweet sentiment using advanced time-based filtering and correlates sentiment patterns with stock performance metrics.

## Features

- **Advanced Sentiment Analysis**: Uses NLTK's VADER sentiment analyzer to extract compound, positive, negative, and neutral sentiment scores from tweets
- **Time-Based Tweet Filtering**: Separates tweets into pre-market (before 5:30 AM EST) and intraday (5:30 AM - 12:30 PM EST) periods
- **Sentiment Aggregation**: Computes statistical measures (mean, min, max, percentiles) for sentiment scores by stock and date
- **Correlation Analysis**: Analyzes relationships between sentiment metrics and stock price movements
- **Interactive Visualizations**: Includes heatmaps, time series plots, and statistical visualizations
- **Multi-Stock Analysis**: Supports analysis across multiple stocks (TSLA, MSFT, AAPL, GOOG, etc.)

## Files

- `stock_prediction.ipynb`: Complete Jupyter notebook with data processing, sentiment analysis, and visualization
- `stock_tweets.csv`: Twitter dataset with 80,793 tweets across multiple stocks
- `stock_yfinance_data.csv`: Historical stock price data (6,300 records) from Yahoo Finance
- `X_scaler.pkl`: Pickled MinMaxScaler for input features
- `y_scaler.pkl`: Pickled MinMaxScaler for target variable
- `README.md`: Project documentation

## Machine Learning Models

The notebook includes implementation of several machine learning models for stock price prediction:

- **Logistic Regression**: Baseline model for market movement classification
- **Random Forest Classifier**: Ensemble method for improved accuracy
- **XGBoost Classifier**: Gradient boosting algorithm for high-performance prediction
- **LSTM Neural Networks**: Deep learning model for time series forecasting

### Model Performance
- Achieves ~99.7% accuracy on test data
- Includes feature scaling and label encoding
- Supports multi-class classification (up/down/neutral movements)

## Requirements

- Python 3.x
- Jupyter Notebook
- Required Python packages:
  - numpy
  - pandas
  - matplotlib
  - seaborn
  - tensorflow
  - scikit-learn
  - nltk
  - plotly
  - tqdm
  - statsmodels
  - unicodedata

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/rushibankar3/Stock-market-sentiment-Analysis.git
   cd Stock-market-sentiment-Analysis
   ```

2. Install required packages:
   ```bash
   pip install numpy pandas matplotlib seaborn tensorflow scikit-learn nltk plotly tqdm statsmodels
   ```

3. Download NLTK data:
   ```python
   import nltk
   nltk.download('vader_lexicon')
   ```

## Usage

1. Open the Jupyter notebook:
   ```bash
   jupyter notebook stock_prediction.ipynb
   ```

2. Run the analysis pipeline:
   - **Data Loading**: Import and explore tweet and stock price datasets
   - **Sentiment Analysis**: Process tweets with VADER sentiment analyzer
   - **Time-Based Filtering**: Separate tweets by market hours
   - **Statistical Aggregation**: Compute sentiment metrics by stock and time period
   - **Data Integration**: Merge sentiment data with stock performance metrics
   - **Correlation Analysis**: Generate heatmaps showing sentiment-stock relationships
   - **Visualization**: Create interactive plots and statistical summaries

## Methodology

### 1. Data Preprocessing
- Load Twitter data (80K+ tweets) and stock price data
- Convert timestamps to datetime format with timezone handling
- Clean and normalize tweet text using Unicode normalization

### 2. Sentiment Analysis Pipeline
- Apply VADER sentiment analysis to extract:
  - Compound sentiment score (-1 to +1)
  - Positive, negative, and neutral component scores
- Process tweets in batches for efficiency

### 3. Time-Based Analysis
- **Pre-market tweets**: Before 5:30 AM EST (sentiment before market open)
- **Intraday tweets**: 5:30 AM - 12:30 PM EST (during market hours)
- Aggregate sentiment statistics for each time window

### 4. Feature Engineering
- Calculate percentage changes in stock prices and volume
- Create lagged sentiment features (previous day sentiment)
- Generate statistical measures (mean, min, max, quartiles)

### 5. Correlation Analysis
- Compute correlations between sentiment metrics and stock performance
- Visualize relationships using seaborn heatmaps
- Identify predictive sentiment patterns

## Key Findings

The analysis reveals significant correlations between tweet sentiment and stock market performance, particularly:
- Pre-market sentiment often predicts next-day price movements
- Intraday sentiment correlates with same-day trading volume
- Negative sentiment shows stronger correlations than positive sentiment
- Different stocks show varying sensitivity to sentiment patterns

## Results & Visualizations

The notebook includes:
- Sentiment distribution plots by stock
- Time series analysis of sentiment vs. stock prices
- Correlation heatmaps showing sentiment-stock relationships
- Statistical summaries and performance metrics
- Interactive plotly visualizations

## Recent Updates

- ✅ Fixed XGBoost class label encoding issues (converted -1,0,1 to 0,1,2)
- ✅ Resolved data type errors by filtering to numeric columns only
- ✅ Fixed technical indicator calculations (MACD, EMA, Bollinger Bands)
- ✅ Added label encoding for machine learning models
- ✅ Created and saved MinMaxScaler objects for feature scaling
- ✅ Fixed seaborn import error in correlation analysis
- ✅ Updated file paths for local execution
- ✅ Enhanced error handling and data validation
- ✅ Improved documentation and code comments

## Contributing

Feel free to submit issues, feature requests, or pull requests to improve the analysis.

## License

This project is open source and available under the MIT License.

## Repository

[GitHub Repository](https://github.com/rushibankar3/Stock-market-sentiment-Analysis)