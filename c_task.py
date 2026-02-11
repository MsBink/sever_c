import numpy as np
import pandas as pd


class DataPreprocessor:
    def __init__(self, data):
        if not isinstance(data, pd.DataFrame):
            raise TypeError(f"init value need to be pd.dataframe, got {type(data)}")
        
        self.data = data.copy()
        self.transform_log = {
            "removed_cols": [],
            "filled_cols": {},
            "onehot_cols": [],
        }

    def remove_missing(self, threshold = 0.5):
        if not 0 <= threshold <= 1:
            raise ValueError(f"threshold need to be in [0, 1]")

        missing_part = self.data.isna().mean()
        cols_to_drop = list(missing_part[missing_part > threshold].index)
        self.data = self.data.drop(columns=cols_to_drop)
        self.transform_log["removed_cols"] = cols_to_drop

        for col in self.data.columns:
            if self.data[col].isna().sum() == 0:
                continue

            fill_value = self.data[col].mode().iloc[0]
            self.data[col] = self.data[col].fillna(fill_value)
            self.transform_log["filled_cols"][col] = fill_value
        
        return self

    def encode_categorical(self):
        cat_cols = list(self.data.select_dtypes(include=["object", "category", "string"]).columns)

        if not cat_cols:
            return self

        self.data = pd.get_dummies(self.data, columns=cat_cols, dtype=int)

        new_cols = [
            c for c in self.data.columns
            if any(c.startswith(f"{orig}_") for orig in cat_cols)
        ]
        self.transform_log["onehot_cols"] = new_cols

        return self
    
    def normalize_numeric(self, method = "minmax"):
        if method not in ("minmax", "std"):
            raise ValueError(
                f"Invalid method, method need to be 'minmax' or 'std'"
            )

        num_cols = self.data.select_dtypes(include=np.number).columns.tolist()

        for col in num_cols:
            if method == "minmax":
                col_min = float(self.data[col].min())
                col_max = float(self.data[col].max())
                denom = col_max - col_min
                if denom == 0:
                    self.data[col] = 0.0
                else:
                    self.data[col] = (self.data[col] - col_min) / denom
            else:
                col_mean = float(self.data[col].mean())
                col_std = float(self.data[col].std(ddof=0))
                if col_std == 0:
                    self.data[col] = 0.0
                else:
                    self.data[col] = (self.data[col] - col_mean) / col_std

        return self

    def fit_transform(self, threshold = 0.5, method = "minmax"):
        self.remove_missing(threshold=threshold)
        self.encode_categorical()
        self.normalize_numeric(method=method)
        return self.data

