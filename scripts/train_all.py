"""
Train LightGBM -> TCN -> LSTM in order.
"""
from forecast.training.train_lightgbm import train_lightgbm
from forecast.training.train_tcn import train_tcn
from forecast.training.train_lstm import train_lstm

def main():
    print("Training LightGBM...")
    train_lightgbm()
    print("Training TCN...")
    train_tcn()
    print("Training LSTM...")
    train_lstm()
    print("All models trained.")

if __name__ == "__main__":
    main()
