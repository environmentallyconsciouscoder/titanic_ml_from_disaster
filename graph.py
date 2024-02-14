import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from matplotlib import pyplot as plt

class Graph:
    def __init__(self, df):
        self.df = df

    def get_graph(self):
        return self.df

    def bar_graph(self, data):
        self.df.displot(data=data)

    def create_correlation_matrix(self, data):
        plt.figure(figsize=(20,20))
        self.df.heatmap(data, annot=True)