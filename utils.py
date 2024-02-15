import pandas as pd
import numpy as np
import sweetviz as sv

def open_csv_file(fileName: str) -> pd.DataFrame:
    data = pd.read_csv(fileName)
    return data


def analyze_dataframe(df):
    # Get list of column names
    column_names = list(df.get_df().keys())

    # Initialize lists to store categorical and numerical column names
    categorical_columns = []
    numerical_columns = []

    # Iterate over columns and categorize them
    for col in column_names:
        # Check if the values in the column are strings (categorical) or numbers (numerical)
        if all(isinstance(val, str) for val in df.get_df()[col]):
            categorical_columns.append(col)
        elif all(isinstance(val, (int, float)) for val in df.get_df()[col]):
            numerical_columns.append(col)

    # Return the total number of categorical and numerical columns along with their names
    return {
        'total_categorical_columns': len(categorical_columns),
        'total_numerical_columns': len(numerical_columns),
        'categorical_column_names': categorical_columns,
        'numerical_column_names': numerical_columns
    }

def create_report(df):
    report = sv.analyze([df, 'sex_female'], target_feat='Survived')
    return report.show_html('Report.html')

