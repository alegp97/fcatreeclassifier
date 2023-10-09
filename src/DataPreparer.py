import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import KBinsDiscretizer, Binarizer, QuantileTransformer
from sklearn import datasets


""" 
class DataPreparer:
    def __init__(self):
        self.data = []
        self.training_data = []
        self.test_data = []

    def load_data_from_csv(self, file_path):
        df = pd.read_csv(file_path)  
        self.data.append(df)

    def extract_data_from_api(self, api_url):
        response = requests.get(api_url)
        json = response.json()
        df = pd.DataFrame(json)
        self.data.append(df)

    def is_local_path(url):
        parsed_url = urlparse(url)
        
        # Check if the scheme is empty or file://
        if not parsed_url.scheme or parsed_url.scheme == 'file':
            return True

        # Check if the netloc (domain) is empty
        if not parsed_url.netloc:
            return True
        
        # Check if the URL starts with a slash indicating a relative path
        if parsed_url.path.startswith('/'):
            return True

        return False 

    def load_data_from_multiple_sources(self, data_sources, test_size):
        ''' 
        @param data_sources: list of string with data route-sources
        '''

        for source in data_sources:
            try:
                if self.is_local_path(source) : #source is a local path
                    self.load_data_from_csv(source)
                    continue
            except ValueError:
                raise ValueError(f"Cannot determine the type of '{source}'")

            try:
                if os.path.isabs(source): #source is an API url
                    self.extract_data_from_api(source)
                    continue
            except OSError:
                raise ValueError(f"Cannot determine the type of '{source}'")
            
        merged_df = pd.concat(self.data)
        ''' HACER PREPROCESAMIENTO AQUI SI VIENEN NO ESTRUCTURADOS'''


    def unstructured_data_to_structured(self, unstructured_data):
        ''' HACER PREPROCESAMIENTO AQUI de no estructurado a estructurado'''

        structured_data = ""
        return structured_data


    def scale_attributes(self, attributes):
        scaler = StandardScaler()
        scaled_attributes = scaler.fit_transform(attributes)
        return scaled_attributes


    def create_train_test_data(self, test_size):

        data_array = np.array(self.data)
        X = data_array[:, :-1]  # Input features
        y = data_array[:, -1]   # Target variable

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

        return X_train, X_test, y_train, y_test

    

    def read_context_from_csv(context_string_file_path):

        csv_delimiter = ','

        with open( context_string_file_path ) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=csv_delimiter)
            objects = []
            attributes = []
            relations = []

            first_row = next(csv_reader)
            attributes = first_row # the set of attributes is the first row of the dataset

            for row in csv_reader:
                obj = row[0]
                objects.append(obj)
                for i in range(1, len(row)):
                    attr = row[i]
                    if(attr == '1'):
                        relations.append((obj, attributes[i]))
                
        return objects, attributes, relations
    
    def handle_categorical_attributes(self, attribute_column):
        distinct_values = set(row[attribute_column] for row in self.data)

        if len(distinct_values) > 10:
            for value in distinct_values:
                new_col = [1 if row[attribute_column] == value else 0 for row in self.data]
                for i, row in enumerate(self.data):
                    row.append(new_col[i])
            for row in self.data:
                del row[attribute_column]


    def get_prepared_data(self, csv_path, attribute_column=None):
        self.load_data_from_csv(csv_path)

        #  si se especifica una columna para manejar atributos categóricos, proceder con ello.
        if attribute_column:
            self.handle_categorical_attributes(attribute_column)

        #  añadir más funciones de preprocesamiento si es necesario
        return self.data """





class DataPreparer:
    def __init__(self):
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None


    def get_split_data(self):
        return self.X_train, self.X_test, self.y_train, self.y_test

    def get_data(self):
        return self.X, self.y


    def transform_delimiters(self, file_path, new_delimiter=','):
        # Intenta detectar automáticamente el delimitador usando las primeras líneas del archivo
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
            # Si no se puede detectar, asume coma por defecto
            delimiter = ','
            print(f"(Delimitadores fueron transformados a '{new_delimiter}' en el archivo CSV) \n\n ")
        
        # Lee el archivo con el delimitador detectado
        data = pd.read_csv(file_path, sep=delimiter)
        
        # Reescribe el archivo con el nuevo delimitador
        data.to_csv(file_path, sep=new_delimiter, index=False)
        
        return data






    

    def convert_numeric_to_categorical(self, df, selected_discretizer):
        """
        Convert numeric attributes to categorical.
        
        Parameters:
        - df: DataFrame to be transformed.
        - selected_discretizer: Discretization method to be used. Options are 'equal-width', 'equal-frequency', 'kmeans', 'binarizer', and 'quantile-transformer'.
        
        Returns:
        - DataFrame with numeric attributes converted to categorical.
        """
        
        # Define discretizers
        discretizers = {
            'equal-width': KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='uniform'),
            'equal-frequency': KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='quantile'),
            'kmeans': KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='kmeans'),
            'binarizer': Binarizer(threshold=0.5),  # The threshold might need adjustment
            'quantile-transformer': QuantileTransformer(n_quantiles=10, output_distribution='normal', random_state=0)
        }
        
        discretizer = discretizers[selected_discretizer]
        
        print(f"Discretization Report (Used method: {selected_discretizer} )")
        
        for col in df.columns:
            if df[col].dtype in ['int64', 'float64']:
                # Fit and transform the discretizer for each column individually
                transformed_col = discretizer.fit_transform(df[[col]])
                
                # If the discretizer is KBinsDiscretizer, print the bin edges
                if isinstance(discretizer, KBinsDiscretizer):
                    bin_edges = discretizer.bin_edges_[0]
                    bin_mapping = {}
                    for i in range(len(bin_edges) - 1):
                        bin_mapping[i] = f"[{bin_edges[i]:.2f}, {bin_edges[i+1]:.2f})"
                    print(f"\tColumn {col} - Bins: {bin_mapping}")
                # If the discretizer is Binarizer, print the threshold
                elif isinstance(discretizer, Binarizer):
                    threshold = discretizer.threshold
                    print(f"\tColumn {col} - Binarized at threshold: {threshold}")
                # For QuantileTransformer, print the quantiles
                elif isinstance(discretizer, QuantileTransformer):
                    quantiles = discretizer.quantiles_[:, 0]
                    quantile_mapping = {}
                    for i in range(len(quantiles) - 1):
                        quantile_mapping[i] = f"[{quantiles[i]:.2f}, {quantiles[i+1]:.2f})"
                    print(f"\tColumn {col} - Quantiles: {quantile_mapping}")
                # For other discretizers, we don't have a direct mapping, so we skip
                else:
                    print(f"\tColumn {col} - Mapping not available for this discretizer.")
                
                # Apply the discretization to the dataframe column
                df[col] = transformed_col.ravel()  # Use .ravel() to convert it to 1D array

        print(f"///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////")
        print("\n\n")
        return df





    def prepare_csvfile_data(self, file_path=None, test_size=None, random_state=None, discretizer=None , target_colum=-1):
        """
        Prepares the data by splitting it into training and test sets.
        
        Parameters:
        - data: DataFrame or sklearn dataset with the data.
        - test_size: Proportion of the dataset to include in the test split.
        - random_state: Seed used by the random number generator.
        - target_colum: Index of the target column. By default, it's the last column.
        - discretizer: Discretization method to be used.

        Returns:
        - attributes, concepts
        """
        data = self.transform_delimiters(file_path)
        
        # Check for columns with NaN values
        columns_with_nan = data.columns[data.isnull().any()].tolist()
        if columns_with_nan:
            # If there are columns with NaN values, drop them
            data = data.drop(columns=columns_with_nan)

        # Remove duplicate rows
        duplicated_rows = data[data.duplicated()]
        if duplicated_rows.shape[0] > 0:
            data = data.drop_duplicates()

        # Splitting the data into features and target based on the target_colum
        if target_colum == -1:
            self.X = data.iloc[:, :-1]
            self.y = data.iloc[:, -1]
        else:
            self.X = data.drop(data.columns[target_colum], axis=1)
            self.y = data.iloc[:, target_colum]

        # Convert numeric attributes to categorical
        self.X = self.convert_numeric_to_categorical(self.X, discretizer)

        attributes = list(self.X.columns)
        concepts = list(self.y.unique())

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=test_size, random_state=random_state)

        return attributes, concepts





    #######################################################################
    """ 
        elif dataset_name == "Linnerud":
            data = datasets.load_linnerud()
        elif dataset_name == "Olivetti Faces":
            data = datasets.fetch_olivetti_faces()
        elif dataset_name == "20 Newsgroups":
            data = datasets.fetch_20newsgroups()
        elif dataset_name == "Fetch Species Distributions":
            data = datasets.fetch_species_distributions()
        elif dataset_name == "RCV1":
            data = datasets.fetch_rcv1()
        elif dataset_name == "KDDCup99":
            data = datasets.fetch_kddcup99()
        elif dataset_name == "Optical Recognition of Handwritten Digits":
            data = datasets.load_digits()
        elif dataset_name == "Pen-Based Recognition of Handwritten Digits":
            data = datasets.load_digits() 
        elif dataset_name == "Fetch Labeled Faces in the Wild":
            data = datasets.fetch_lfw_people() """

    def load_sklearn_dataset(self, dataset_name):
        if dataset_name == "Iris":
            data = datasets.load_iris()
        elif dataset_name == "Digits":
            data = datasets.load_digits()
        elif dataset_name == "Wine":
            data = datasets.load_wine()
        elif dataset_name == "Breast Cancer":
            data = datasets.load_breast_cancer()

        else:
            raise ValueError(f"Dataset {dataset_name} no reconocido por la aplicacion.")

        attributes = data.feature_names if hasattr(data, 'feature_names') else None
        concepts = data.target_names if hasattr(data, 'target_names') else None

        return data, attributes, concepts


    def prepare_sklearn_data(self, selected_sklearn_dataset, test_size, randomState, discretizer_choice):

        dataset_name = selected_sklearn_dataset
        dataset, attributes, concepts = self.load_sklearn_dataset(dataset_name)

        df = pd.DataFrame(dataset.data, columns=dataset.feature_names)
        target = pd.Series(dataset.target)

        dataset_categorical, discretizer_report = self.convert_numeric_to_categorical(df, discretizer_choice)

        X_train, X_test, y_train, y_test = train_test_split(dataset_categorical, target, test_size=test_size, random_state=randomState)
        
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

        return attributes, concepts, discretizer_report

