import pandas as pd

def save_file(file: pd.DataFrame, path: str = 'temp/data.csv'):
    file.to_csv(path)