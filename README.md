# Stock Market Sentiment Analysis

Stock Market Sentiment Analysis is an end-to-end project that combines Twitter sentiment signals with historical market data to analyze how public opinion around stocks aligns with price movement.

The repository includes:
- A full analysis and model development notebook (`stock_prediction.ipynb`)
- A Streamlit web app (`streamlit_app.py`) for interactive tweet sentiment prediction
- Supporting datasets and scalers used during experimentation

## Project Objectives

- Extract sentiment from stock-related tweets using VADER (NLTK)
- Explore sentiment patterns by stock and date
- Join sentiment with Yahoo Finance market data
- Engineer predictive features for downstream ML models
- Provide an interactive Streamlit interface for quick sentiment checks

## Tech Stack

- Python
- Pandas, NumPy
- NLTK (VADER)
- Scikit-learn
- TensorFlow / Keras
- Plotly, Matplotlib, Seaborn
- Streamlit

## Repository Structure

```text
Stock-market-sentiment-Analysis/
├── README.md
├── requirements.txt
├── streamlit_app.py
├── stock_prediction.ipynb
├── stock_tweets.csv
├── stock_yfinance_data.csv
├── X_scaler.pkl
├── y_scaler.pkl
└── models_gan/
```

## Datasets Used

### 1) `stock_tweets.csv`
Contains stock-related tweets and metadata (for example stock symbol/name, date, tweet text).

### 2) `stock_yfinance_data.csv`
Contains historical OHLCV market data fetched from Yahoo Finance.

## End-to-End Workflow

1. Load tweet and market datasets.
2. Clean and normalize tweet text.
3. Compute sentiment scores (`pos`, `neu`, `neg`, `compound`) using VADER.
4. Classify tweets into sentiment labels:
   - Positive (`compound >= 0.05`)
   - Negative (`compound <= -0.05`)
   - Neutral (otherwise)
5. Aggregate sentiment by stock/date and analyze correlations with market features.
6. Build and evaluate ML/DL models in the notebook.
7. Serve interactive inference and exploration through Streamlit.

## Notebook (`stock_prediction.ipynb`)

The notebook contains full experimentation and research flow:

- Data loading and profiling
- Tweet preprocessing
- Sentiment generation and aggregation
- Feature engineering with market data
- Correlation and visualization analysis
- Model training (classical ML + deep learning experiments)
- Model evaluation and result interpretation

## Streamlit App (`streamlit_app.py`)

The Streamlit app is designed for quick, interactive sentiment analysis.

### Main capabilities

- Predict sentiment of any input tweet text
- Show compound score as a metric
- Display component-level sentiment bar chart (`positive`, `neutral`, `negative`)
- Explore dataset samples by stock
- Visualize sample sentiment distribution with a pie chart

## Streamlit Screen Working (UI Walkthrough)

When you run the app, you will see these screens/sections:

### 1) Header and Tweet Input Panel

- Title: **Stock Tweet Sentiment Predictor**
- A text area with default sample tweet
- **Predict Sentiment** button to trigger inference

### 2) Prediction Output Panel

After clicking **Predict Sentiment**:

- Predicted label appears (`Positive` / `Neutral` / `Negative`)
- Compound score is shown as a metric
- Plotly bar chart displays `positive`, `neutral`, `negative` component scores

### 3) Rules and Tips Sidebar Panel

- Shows threshold rules used for label mapping
- Notes that this app performs sentiment scoring (not direct price prediction)

### 4) Dataset Explorer Section

- Stock filter dropdown (`All` + available stock names)
- Adjustable row preview slider
- Data table preview
- Pie chart of label distribution for sampled tweets

## How to Run Locally

### 1) Clone repository

```bash
git clone https://github.com/rushibankar3/Stock-market-sentiment-Analysis.git
cd Stock-market-sentiment-Analysis
```

### 2) Install dependencies

```bash
pip install -r requirements.txt
```

### 3) Download NLTK VADER lexicon (first run)

```python
import nltk
nltk.download('vader_lexicon')
```

### 4) Run Streamlit app

```bash
streamlit run streamlit_app.py
```

The app opens in your browser, usually at:
- `http://localhost:8501`

## Expected Inputs and Outputs

### Input
- A tweet-like text about a stock or company.

### Output
- Sentiment label (`Positive`, `Neutral`, `Negative`)
- Compound sentiment score
- Component breakdown chart
- Optional aggregate explorer visuals from dataset samples

## Important Notes

- The Streamlit app uses lexicon-based sentiment analysis (VADER).
- Sentiment output indicates tone, not guaranteed market direction.
- Model/scaler artifacts in the repository are for experiments in the notebook pipeline.

## Future Improvements

- Add transformer-based sentiment model for improved context understanding
- Add model selection toggle in Streamlit (VADER vs ML model)
- Add backtesting dashboard for sentiment-based signals
- Add Docker setup for one-command deployment

## Contributing

Contributions are welcome. You can open an issue or submit a pull request for improvements.

## License

This project is open-source under the MIT License.

## Author

Maintained by [@rushibankar3](https://github.com/rushibankar3)
