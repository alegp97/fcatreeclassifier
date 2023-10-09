""" import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split



from fcapy.lattice import ConceptLattice
from fcapy.context import FormalContext


import pandas as pd

file_path = 'fca-decision-tree-classifier\\datasets\\real-datasets\\sponge.csv'

def adapt_csv_for_fcapy(file_path):
    # Leemos el CSV original
    df = pd.read_csv(file_path)
    # Reemplazamos los valores "1" y "0" por "True" y "False"
    df = df.replace({1: "True", 0: "False"})
    # Guardamos el CSV modificado en el mismo lugar
    df.to_csv(file_path, index=False)

adapt_csv_for_fcapy(file_path)

K = FormalContext.read_csv('fca-decision-tree-classifier\\datasets\\real-datasets\\sponge.csv') 
print('# objects', K.n_objects, '; # attributes', K.n_attributes)

L = ConceptLattice.from_context(K, algo='CbO')
print("# concepts:", len(L))


 """



from fcapy.context import FormalContext
from fcapy.lattice import ConceptLattice
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score, matthews_corrcoef

from sklearn.datasets import load_breast_cancer, load_iris, load_wine
import numpy as np

""" 
Preprocesamiento y Reducción de Características: Usar FCA para identificar y eliminar características redundantes. 
Las características que aparecen en los mismos conjuntos de conceptos tienen relaciones similares con los objetos y podrían ser redundantes.
 """
def reduce_features_using_fca(X):
    X_bin = np.where(X > X.mean(), True, False)
    K = FormalContext(data=X_bin)
    L = ConceptLattice.from_context(K)
    attrs = set(range(X.shape[1]))
    for concept in L:
        attrs -= set(concept.intent)
    return X[:, list(attrs)]


""" 2. Uso del retículo de conceptos para guiar la construcción del árbol:

Esta tarea es compleja y requiere una integración profunda con el proceso de construcción del árbol de decisión. 
En lugar de una implementación completa, aquí hay una idea 
simplificada donde utilizamos el tamaño del intento (conjunto de características) de un concepto para determinar la profundidad del árbol: """

def get_tree_depth_using_fca(X):
    X_bin = np.where(X > X.mean(), True, False)
    K = FormalContext(data=X_bin)
    L = ConceptLattice.from_context(K)
    return max([len(c.intent) for c in L])


""" Condensación de Datos: Utilizar el retículo para identificar grupos de objetos que siempre aparecen juntos en los mismos conceptos. 
Estos objetos podrían tratarse como una única instancia durante el entrenamiento, reduciendo así el tamaño del conjunto de datos. """

def condense_data_using_fca(X, y):
    X_bin = np.where(X > X.mean(), True, False)
    unique_rows, indices = np.unique(X_bin, axis=0, return_inverse=True)
    y_condensed = [np.bincount(y[indices == i]).argmax() for i in range(unique_rows.shape[0])]
    return unique_rows, y_condensed



def process_dataset(load_dataset_func):
    data = load_dataset_func()
    X = data.data
    y = data.target

    X_reduced = reduce_features_using_fca(X)
    max_depth = get_tree_depth_using_fca(X_reduced)
    X_condensed, y_condensed = condense_data_using_fca(X_reduced, y)

    X_train, X_test, y_train, y_test = train_test_split(X_condensed, y_condensed, test_size=0.2, random_state=42)
    clf = DecisionTreeClassifier(max_depth=max_depth)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    print(f'\nDataset: {load_dataset_func.__name__}')
    print(f'Precisión del árbol de decisión: {accuracy_score(y_test, y_pred)}')
    print(f'Matriz de Confusión:\n{confusion_matrix(y_test, y_pred)}')
    print(f'Precisión: {precision_score(y_test, y_pred, zero_division=0)}')
    print(f'Recall: {recall_score(y_test, y_pred, zero_division=0)}')
    print(f'F1-Score: {f1_score(y_test, y_pred, zero_division=0)}')

    if len(np.unique(y_test)) > 1:
        print(f'ROC AUC: {roc_auc_score(y_test, y_pred)}')
    else:
        print("ROC AUC no puede ser calculado porque solo hay una clase presente.")

    print(f'Índice de Matthews: {matthews_corrcoef(y_test, y_pred)}')



# Lista de funciones de carga de datasets
datasets = [load_breast_cancer, load_iris, load_wine]

# Procesar cada dataset
for dataset in datasets:
    process_dataset(dataset)


