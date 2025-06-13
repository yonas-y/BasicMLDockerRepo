# main.py
from data_loader import load_data
# from train import train_ocsvm

if __name__ == "__main__":
    df = load_data()
    print(df.info())
    print(df.head())
    # train_ocsvm(df)
