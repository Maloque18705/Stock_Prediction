# Stock Price Prediction with LSTM

This project implements a Long Short-Term Memory (LSTM) neural network to predict stock prices based on historical data and technical indicators. The system processes stock data, computes technical indicators, trains an LSTM model, and generates predictions, visualized through actual vs. predicted price plots.

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Usage](#usage)
- [Data](#data)
- [Technical Details](#technical-details)


## Overview
The project fetches stock data from a MinIO storage system, preprocesses it by computing technical indicators (e.g., SMA, EMA, RSI, MACD, Bollinger Bands, Stochastic Oscillator), and uses an LSTM model to predict future stock prices. 

## Features
- **Data Retrieval**: Fetches stock data from MinIO with configurable time filtering (e.g., last 3 years).
- **Preprocessing**: Computes technical indicators and creates time-series windows for LSTM input.
- **Model Training**: Uses an LSTM model with customizable architecture (hidden size, number of layers, dropout).
- **Visualization**: Plots actual vs. predicted stock prices and saves results as PNG files.

## Requirements
- Python 3.8 or higher
- PyTorch
- pandas
- NumPy
- scikit-learn
- Optuna
- Matplotlib
- TA-Lib (for technical indicators)
- fastparquet
- MinIO client (for data access)

## Installation
1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd stock-price-prediction
   ```

2. Create a virtual environment and activate it:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Install TA-Lib (may require additional setup depending on your system):
   - Follow instructions at [TA-Lib Installation](https://github.com/TA-Lib/ta-lib-python).
   - Example for Ubuntu:
     ```bash
     sudo apt-get install libta-lib0 libta-lib0-dev
     pip install TA-Lib
     ```

5. Configure MinIO credentials in a `.env` file or environment variables:
   ```plaintext
   MINIO_ACCESS_KEY=<your-access-key>
   MINIO_SECRET_KEY=<your-secret-key>
   MINIO_ENDPOINT=<your-endpoint-url>
   BUCKET=<your-bucket-name>
   ```

## Project Structure
```
stock-price-prediction/
├── main.py                # Main script to run the pipeline
├── process.py             # Data preprocessing and technical indicator computation
├── model.py               # LSTM model definition
├── optimizer.py           # Optimizer and DataLoader setup
├── train.py               # Training and evaluation logic
├── scalers/               # Directory to store scaler files
├── plots/                 # Directory to store prediction plots
├── models/                # Directory to store trained models
├── requirements.txt       # Dependencies
├── README.md              # Project documentation
```

## Usage
1. **Prepare Data**:
   - Ensure stock data is stored in MinIO under `s3a://{BUCKET}/RAW_STOCK_DATA/symbol=<symbol>`.
   - Data should be in Parquet format with columns: `trading_date`, `close`, `high`, `low`, `volume`.
   - Changing input inside "Main_model.py", in "__main__", under 'df' variable.

2. **Output**:
   - Scaler files saved in `./scalers/`.
   - Trained model checkpoints saved in `./models/`.
   - Prediction plots saved in `./plots/` (e.g., `FPT_price_prediction.png`).

## Hyperparameter 
- `batch_size`: [16]
- `learning_rate`: [1e-5] (log scale)
- `hidden_size`: [25]
- `num_layers`: [2]
- `dropout`: [0.2,


## Data
- **Source**: Stock data stored in MinIO.
- **Format**: Parquet files with columns `trading_date`, `close`, etc.
- **Time Range**: Default is the last 3 years, configurable in `get_data`.
- **Preprocessing**:
  - Technical indicators (SMA, EMA, RSI, MACD, Bollinger Bands, Stochastic Oscillator) computed with TA-Lib.
  - Data scaled using MinMaxScaler and saved for reuse.
  - Time-series windows created with sequence length of 3 days.

## Technical Details
### Optimizer Class (`optimizer.py`)
The `Optimizer` class configures the training setup:
- **Initialization**:
  - Validates input shapes: `X_train` (samples, seq_len, input_size), `y_train` (samples,).
  - Moves model and data to specified device (CPU or CUDA).
  - Uses Adam optimizer with configurable `learning_rate` (default: 1e-3).
  - Uses Mean Squared Error (MSE) as the loss function.
- **DataLoader**:
  - Creates batches with `batch_size` (default: 16) and shuffles data for training.

### Main Pipeline (`main.py`)
- Fetches data for a single stock symbol.
- Preprocesses data (cleaning, technical indicators, scaling, windowing).
- Optimizes hyperparameters using Optuna.
- Trains the LSTM model with early stopping (patience: 50 epochs).
- Generates predictions and visualizes results.

### Model Architecture
- **LSTM**:
  - Input size: 12 (number of features, e.g., close, volume, technical indicators).
  - Hidden size: Configurable (default: 32).
  - Number of layers: Configurable (default: 2).
  - Dropout: Configurable (default: 0.2).
- **Output**: Predicts the next day's closing price.

### Training
- **Epochs**: Up to 1000, with early stopping based on validation RMSE.
- **Evaluation**: Train and validation RMSE computed every 100 epochs.
- **Loss Function**: MSE.
- **Optimizer**: Adam.
