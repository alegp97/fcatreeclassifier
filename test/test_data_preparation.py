import sys
sys.path.append('C:\\Users\\alegp\\OneDrive\\Escritorio\\TFG\\CODE\\fca-decision-tree-classifier')

from src.DataPreparer import DataPreparer

# Test the DataPreparer class
def test_data_preparation():
    # Create an instance of DataPreparer
    dataPreparer = DataPreparer()

    # Test loading data from CSV
    dataPreparer.load_data_from_csv("CODE\datasets\\real-datasets\\breastcancer.csv")
    print("Data loaded from CSV:")
    print(dataPreparer.data)

    """     # Test extracting data from API
    DataPreparer.extract_data_from_api('https://api.example.com/data')
    print("Data extracted from API:")
    print(DataPreparer.data)

    # Test structuring unstructured data
    unstructured_data = 'Some unstructured data'
    structured_data = DataPreparer.structure_unstructured_data(unstructured_data)
    print("Structured data:")
    print(structured_data) 
    """

    # Test creating train-test data
    X_train, X_test, y_train, y_test = dataPreparer.create_train_test_data(test_size=0.2)
    print("Training set (X_train):")
    print(X_train)
    print("Test set (X_test):")
    print(X_test)
    print("Training set (y_train):")
    print(y_train)
    print("Test set (y_test):")
    print(y_test)

    # Test scaling attributes
    attributes = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    scaled_attributes = DataPreparer.scale_attributes(attributes)
    print("Scaled attributes:")
    print(scaled_attributes)

    # Test additional method
    DataPreparer.additional_method()


# Run the test
test_data_preparation()
