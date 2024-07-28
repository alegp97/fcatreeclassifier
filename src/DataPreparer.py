import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import KBinsDiscretizer, Binarizer, QuantileTransformer
from sklearn.preprocessing import OneHotEncoder

class DataPreparer:
    def __init__(self):
        self.reset()

    def reset(self):
        self.X_original = None
        self.y_original = None
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.attributes = None
        self.targets = None
        self.ohe = OneHotEncoder(sparse_output=False)


    def get_train_test_split_data_binarized(self, debug=False, random_state=42, test_size=0.2):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=test_size, random_state=random_state)
        self.binarize(debug)
        return self.X_train, self.X_test, self.y_train, self.y_test

    def get_X_y(self):
        return self.X, self.y

    def transform_delimiters(self, file_path, new_delimiter=','):
        with open(file_path, 'r') as file:
            first_line = file.readline()
        
        if ',' in first_line:
            delimiter = ','
        elif ';' in first_line:
            delimiter = ';'
            print(f"(Delimitadores fueron transformados a '{new_delimiter}' en el archivo CSV) \n\n ")
        elif '\t' in first_line:
            delimiter = '\t'
            print(f"(Delimitadores fueron transformados a '{new_delimiter}' en el archivo CSV) \n\n ")
        else:
            delimiter = ','
            print(f"(Delimitadores fueron transformados a '{new_delimiter}' en el archivo CSV) \n\n ")
        
        data = pd.read_csv(file_path, sep=delimiter)
        data.to_csv(file_path, sep=new_delimiter, index=False)
        
        return data

    def binarize(self, debug=False):
        combined_data = pd.concat([self.X_train, self.X_test])
        if(debug):
            print("Combinando datos de entrenamiento y prueba para el OneHotEncoding...")

        self.ohe.fit(combined_data)
        self.X_train = self.ohe.transform(self.X_train)
        self.X_test = self.ohe.transform(self.X_test)
        
        cols = self.ohe.get_feature_names_out()
        if(debug):
            print(f"Columnas después de la transformación OneHot: {cols}")
        
        self.X_train = pd.DataFrame(self.X_train, columns=cols)
        self.X_test = pd.DataFrame(self.X_test, columns=cols)
        
        self.attributes = list(self.X_train.columns)

        if(debug):
            print("Atributos actualizados después de la binarización.")
            print("Tamaño del conjunto de entrenamiento:", self.X_train.shape)
            print("Tamaño del conjunto de prueba:", self.X_test.shape)
            print("Primeras 5 filas del conjunto de entrenamiento tras la binarización:\n", self.X_train.head())

    def discretize_with_custom_thresholds(self, df, thresholds):
        def remove_duplicates_from_thresholds(thresholds):
            return {k: sorted(set(v)) for k, v in thresholds.items()}

        thresholds = remove_duplicates_from_thresholds(thresholds)
        df_discretized = df.copy()
        for attr_index, cuts in thresholds.items():
            attr = df.columns[attr_index]
            bins = [-float('inf')] + sorted(cuts) + [float('inf')]
            labels = range(len(bins) - 1)
            df_discretized[attr] = pd.cut(df[attr], bins=bins, labels=labels, duplicates='drop')
        return df_discretized

    def convert_numeric_to_categorical(self, df, selected_discretizer, n_bins):
        discretizers = {
            'equal-width': KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='uniform'),
            'equal-frequency': KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='quantile'),
            'kmeans': KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='kmeans'),
            'binarizer': Binarizer(threshold=0.5),  # The threshold might need adjustment
            'quantile-transformer': QuantileTransformer(n_quantiles=10, output_distribution='normal', random_state=0)
        }
        
        discretizer = discretizers[selected_discretizer]
        
        print(f"\n\nDiscretization Report (Used method: {selected_discretizer} )")
        
        for col in df.columns:
            if df[col].dtype in ['int64', 'float64']:
                transformed_col = discretizer.fit_transform(df[[col]])
                
                if isinstance(discretizer, KBinsDiscretizer):
                    bin_edges = discretizer.bin_edges_[0]
                    bin_mapping = {}
                    for i in range(len(bin_edges) - 1):
                        bin_mapping[i] = f"[{bin_edges[i]:.2f}, {bin_edges[i+1]:.2f})"
                    print(f"\tColumn {col} - Bins: {bin_mapping}")
                elif isinstance(discretizer, Binarizer):
                    threshold = discretizer.threshold
                    print(f"\tColumn {col} - Binarized at threshold: {threshold}")
                elif isinstance(discretizer, QuantileTransformer):
                    quantiles = discretizer.quantiles_[:, 0]
                    quantile_mapping = {}
                    for i in range(len(quantiles) - 1):
                        quantile_mapping[i] = f"[{quantiles[i]:.2f}, {quantiles[i+1]:.2f})"
                    print(f"\tColumn {col} - Quantiles: {quantile_mapping}")
                else:
                    print(f"\tColumn {col} - Mapping not available for this discretizer.")
                
                df.loc[:, col] = transformed_col.ravel()

        print(f"///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////")
        print("\n\n")
        return df
    



    def prepare_csvfile_data(self, file_path=None, discretizer=None, target_colum=-1, n_bins=None, columns_to_remove=None, discretizing_thresholds=None):
        data = self.transform_delimiters(file_path)
        
        if(columns_to_remove):
            data = data.drop(columns=columns_to_remove)

        columns_with_nan = data.columns[data.isnull().any()].tolist()
        if columns_with_nan:
            data = data.drop(columns=columns_with_nan)

        duplicated_rows = data[data.duplicated()]
        if duplicated_rows.shape[0] > 0:
            data = data.drop_duplicates()

        if target_colum == -1:
            self.X = data.iloc[:, :-1]
            self.y = data.iloc[:, -1]
            self.y = self.y.rename('objective_target')
        else:
            self.X = data.drop(data.columns[target_colum], axis=1)
            self.y = data.iloc[:, target_colum]
            self.y = self.y.rename('objective_target')
            self.X = pd.concat([self.X, self.y], axis=1).reset_index(drop=True)
            self.y = self.X.iloc[:, -1]
            self.X = self.X.iloc[:, :-1]

        if discretizing_thresholds is not None:
            self.thresholds = discretizing_thresholds
            self.X = self.discretize_with_custom_thresholds(self.X, discretizing_thresholds)
        elif discretizer is not None:
            self.X = self.convert_numeric_to_categorical(self.X, discretizer, n_bins)

        self.attributes = list(self.X.columns)
        self.targets = list(self.y.unique())

