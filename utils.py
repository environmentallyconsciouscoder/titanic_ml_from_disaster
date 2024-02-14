import pandas as pd

def open_csv_file(fileName: str) -> pd.DataFrame:
    data = pd.read_csv(fileName)
    return data