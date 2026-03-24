# Stock Market Sentiment Analysis

This project performs stock market prediction using sentiment analysis on Twitter data combined with historical stock prices from Yahoo Finance. It utilizes machine learning models including LSTM and GRU networks to forecast stock prices based on sentiment scores derived from tweets.

## Files

- `stock_prediction.ipynb`: Jupyter notebook containing the complete analysis and prediction pipeline
- `stock_tweets.csv`: Dataset containing Twitter posts related to stocks
- `stock_yfinance_data.csv`: Historical stock price data from Yahoo Finance

## Requirements

- Python 3.x
- Jupyter Notebook
- Required Python packages:
  - numpy
  - pandas
  - matplotlib
  - tensorflow
  - scikit-learn
  - nltk
  - plotly
  - tqdm
  - statsmodels

## Installation

1. Clone this repository
2. Install the required packages:
   ```bash
   pip install numpy pandas matplotlib tensorflow scikit-learn nltk plotly tqdm statsmodels
   ```
3. Download NLTK data:
   ```python
   import nltk
   nltk.download('vader_lexicon')
   ```

## Usage

1. Open `stock_prediction.ipynb` in Jupyter Notebook
2. Run the cells in order to:
   - Load and preprocess tweet data
   - Perform sentiment analysis using VADER
   - Load stock price data
   - Train machine learning models (LSTM/GRU)
   - Make predictions and visualize results

## Methodology

1. **Data Collection**: Twitter data and stock prices are collected
2. **Sentiment Analysis**: Tweets are analyzed for sentiment using NLTK's VADER
3. **Feature Engineering**: Sentiment scores are combined with technical indicators
4. **Model Training**: Deep learning models (LSTM/GRU) are trained on the combined features
5. **Prediction**: Models forecast future stock prices

## Results

The notebook includes visualizations and performance metrics for the prediction models.