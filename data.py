import pandas as pd
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler, KBinsDiscretizer, LabelEncoder, OneHotEncoder

class DataFrame:
    def __init__(self, df):
        self.df = df

    def get_df(self):
        return self.df

    def check_missing_data(self):
        missing_values_count = self.df.isnull().sum()
        return missing_values_count

    def check_data_type(self):
        data_types = self.df.dtypes
        return data_types

    def remove_columns(self, column_names: list):
        self.df = self.df.drop(column_names, axis=1)

    def replace_missing_value_with_median(self, column_name: str):
        self.df[column_name] = self.df[column_name].fillna(self.df[column_name].median())

    def label_encoding(self, column_name: str):
        le = LabelEncoder()
        self.df[column_name + '_encoded'] = le.fit_transform(self.df[column_name])

    def hot_encoding(self, column_name: str):
        ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        ohe_transform = ohe.fit_transform(self.df[[column_name]])
        ohe_df = pd.DataFrame(ohe_transform, columns=ohe.get_feature_names_out([column_name]))
        self.df = pd.concat([self.df, ohe_df], axis=1)
        self.df.drop(columns=[column_name], inplace=True)

    def create_last_name_column(self, column_name: str):
        last_names = [name.split(",")[0] for name in self.df[column_name]]
        self.df['Last_name'] = last_names

    def remove_columns(self, columns: list):
        self.df.drop(columns=columns, inplace=True)

    def min_max_scaler(self, column: str):
        # Create an instance of MinMaxScaler
        scaler = MinMaxScaler(feature_range=(0, 1))
        # Reshape the data if it's a single feature
        data_to_scale = self.df[column].values.reshape(-1, 1)
        # Fit and transform the data
        scaled_data = scaler.fit_transform(data_to_scale)
        # Optionally, assign the scaled data back to the DataFrame
        self.df[column + '_scaled'] = scaled_data
        self.df.drop(columns=[column], inplace=True)

    def bucket_quantile(self, column: str):
        # Number of buckets
        number_buckets = 5
        # Initialize the instance of KBinsDiscretizer
        dis = KBinsDiscretizer(n_bins=number_buckets, encode='ordinal', strategy='quantile')
        ## fit and transform the data
        self.df[ column + '_bucket'] = dis.fit_transform(self.df[[column]])

    def get_bin_edges(self, column: str, strategy: str):
        # Number of buckets
        number_buckets = 5
        # Initialize the instance of KBinsDiscretizer
        dis = KBinsDiscretizer(n_bins=number_buckets, encode='ordinal', strategy=strategy)
        dis.fit_transform(self.df[[column]])
        # Get the bin edges
        bin_edges = dis.bin_edges_
        return bin_edges
