import numpy as np
import pandas as pd
import yfinance as yf
import datetime

def remove_symbols(x):
    x = str(x)
    x = remove_chars(x)
    return x.replace(",", "").replace("%", "")

def remove_chars(x):
    chars = [chr(i) for i in range(65, 65+26)]
    x = str(x)
    for c in x:
        if c in chars:
            x = x.replace(c, "")
    return x

def data_preprocess(raw_data: pd.DataFrame) -> pd.DataFrame:
    df = raw_data.copy()
    # Handle the columns based on yfinance output
    # yfinance output columns: ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
    # We need: Date, Open, High, Low
    
    if 'Date' not in df.columns:
        df['Date'] = df.index
        df.index = range(df.shape[0])
    
    # Ensure columns exist
    cols = ['Date', 'Open', 'High', 'Low', 'Close']
    df = df[cols]
    
    df["Open"] = df["Open"].apply(remove_symbols).astype("float")
    df["High"] = df["High"].apply(remove_symbols).astype("float")
    df["Low"] = df["Low"].apply(remove_symbols).astype("float")
    


    # Sort and drop NAs
    df = df.sort_values(by="Date")
    
    # Calculate target (Up if next day Open is >= today's Open)
    future_value = df["Close"].shift(-1)
    difference = future_value - df["Close"]
    df["Difference"] = difference

    df["Target"] = np.where(
        difference >= 0, "Up", "Down"
    )

    df = df.dropna()
    return df

def format_data(data: pd.DataFrame) -> list[str]:
    def row_to_string(row):
        # Format date as YYYY-MM-DD
        date_str = row['Date'].strftime('%Y-%m-%d') if hasattr(row['Date'], 'strftime') else str(row['Date'])
        # Rounding to 2 decimal places to reduce noise for the LLM
        open_val = round(float(row['Open']), 2)
        high_val = round(float(row['High']), 2)
        low_val = round(float(row['Low']), 2)
        return (f"Date: {date_str}, Open: {open_val}, High: {high_val}, "
                f"Low: {low_val}, Output: {row['Target']}")

    return data.apply(row_to_string, axis=1).tolist()

def write_data(data: list[str]) -> bool:
    with open("train.txt", "w") as f:
        for line in data:
            f.write(line + "\n")
    return True

if __name__ == "__main__":
    print("Downloading NIFTY 50 data...")
    # Fetch data from 2010 to current
    one_month_back = datetime.datetime.now() - datetime.timedelta(days=30)
    data = yf.download("^NSEI", start="2010-01-01", end=one_month_back.strftime("%Y-%m-%d"), multi_level_index=False)
    if data.empty:
        print("Failed to download data.")
    else:
        print("Preprocessing data...")
        df = data_preprocess(data)
        print("Formatting data...")
        formatted_data = format_data(df)
        print(f"Writing {len(formatted_data)} lines to train.txt...")
        write_data(formatted_data)
        print("Done.")
