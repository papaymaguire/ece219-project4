import pathlib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
class WineDataset:
    def __init__(self, folder_path="../data/ECE219_wine_data/wine+quality/") -> None:
        path = pathlib.Path(folder_path)
        reds_file = "winequality-red.csv"
        whites_file = "winequality-white.csv"
        reds_df = pd.read_csv(path.joinpath(reds_file), delimiter=";")
        reds_df['type'] = 'red'
        whites_df = pd.read_csv(path.joinpath(whites_file), delimiter=";")
        whites_df['type'] = 'white'
        self.data = pd.concat([reds_df, whites_df]).reset_index(drop=True)
        cols = list(self.data.columns)
        cols = cols[-2:] + cols[:-2]
        self.data = self.data[cols]
        self.og_data = self.data.copy(deep=True)


def one_hot_encode_categorical(dataset: WineDataset):
    categorical_variables = ['type']
    for var in categorical_variables:
        data = dataset.data[var].to_frame().to_numpy()
        ohe = OneHotEncoder()
        features = ohe.fit_transform(data).toarray()
        df_features = pd.DataFrame(features, columns=ohe.categories_[0])
        dataset.data.drop([var], axis=1, inplace=True)
        dataset.data = dataset.data.join(df_features)