

from src.DataPreparer import DataPreparer
from src.OtherModels import *
from src.FCADecisionTree import FCADecisionTree

from sklearn.preprocessing import OneHotEncoder
import pandas as pd


data_preparer = DataPreparer()
attributes, concepts = data_preparer.prepare_csvfile_data(
    file_path=r'fca-decision-tree-classifier\datasets\\user\\iris.csv',
    test_size=0.2,
    random_state=42,
    discretizer='equal-width',
    target_colum=-1
)

print(f"Atributos: {attributes}")
print(f"Conceptos: {concepts}")



X_train, X_test, y_train, y_test = data_preparer.get_split_data()
print(f"Tipo de X_train: {type(X_train)}")
print(f"Tipo de X_test: {type(X_test)}")
print(f"Tipo de y_train: {type(y_train)}")
print(f"Tipo de y_test: {type(y_test)}")
print(f"Forma de X_train: {X_train.shape if hasattr(X_train, 'shape') else 'No disponible'}")
print(f"Forma de X_test: {X_test.shape if hasattr(X_test, 'shape') else 'No disponible'}")
print(f"Forma de y_train: {y_train.shape if hasattr(y_train, 'shape') else 'No disponible'}")
print(f"Forma de y_test: {y_test.shape if hasattr(y_test, 'shape') else 'No disponible'}")

# Imprimir las primeras filas de X_train
print("X_train:")
print(X_train.head() if hasattr(X_train, 'head') else X_train[:5])

# Imprimir las primeras filas de X_test
print("\nX_test:")
print(X_test.head() if hasattr(X_test, 'head') else X_test[:5])

# Imprimir las primeras filas de y_train
print("\ny_train:")
print(y_train.head() if hasattr(y_train, 'head') else y_train[:5])

# Imprimir las primeras filas de y_test
print("\ny_test:")
print(y_test.head() if hasattr(y_test, 'head') else y_test[:5])








# Inicializar el codificador
encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')

# Ajustar el codificador con los datos de entrenamiento y transformar
X_train_bin = encoder.fit_transform(X_train)

# Transformar los datos de prueba con el mismo codificador
X_test_bin = encoder.transform(X_test)

# Convertir las matrices numpy a DataFrames para una mejor visualización
X_train_bin_df = pd.DataFrame(X_train_bin, columns=encoder.get_feature_names_out(X_train.columns))
X_test_bin_df = pd.DataFrame(X_test_bin, columns=encoder.get_feature_names_out(X_test.columns))

# Imprimir las primeras filas de los datos binarizados
print("X_train binarizado:")
print(X_train_bin_df.head())

print("\nX_test binarizado:")
print(X_test_bin_df.head())


# Usar los datos binarizados para entrenar el árbol de decisión basado en FCA
FCAtree = FCADecisionTree(attribute_names=X_train_bin_df.columns.tolist(), concept_names=concepts)
FCAtree.train_tree(data=X_train_bin_df.values.tolist(), target=y_train.values.tolist())

# Realizar predicciones con el árbol entrenado
predictions_tree = FCAtree.classify_instances(X_test_bin_df.values.tolist())

# Evaluar las métricas del árbol
FCAtree.evaluation_metrics(y_test, predictions_tree, average='macro')



# Obtener el tamaño del retículo de conceptos
print(f"Size of the lattice: {FCAtree.lattice_size()}")

# Obtener la distribución de clases en el retículo
print(f"Class distribution in the lattice: {FCAtree.lattice_class_distribution()}")

# Obtener la profundidad máxima del retículo
print(f"Maximum depth of the lattice: {FCAtree.lattice_max_depth()}")


print(" \n\n\n   TREE NODE STRUCTURE: \n\n   ")
FCAtree.print_tree_structure()
print(" \n\n##########################################################################################")
print(" \n\n\n   TREE LOGIC DECISIONS : \n\n   ")
FCAtree.print_tree_logic()