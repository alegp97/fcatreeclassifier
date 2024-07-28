from flask import render_template, request, redirect, url_for
from web_app import app


from src.FCADecisionTree import FCADecisionTree
from src.DecisionTreeC45 import DecisionTreeC45
from src.DataPreparer import DataPreparer
from src.OtherModels import *

import os


""" class Color:
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    MAGENTA = '\033[95m'
    CYAN = '\033[96m'
    END = '\033[0m'

print(f"{Color.RED}Texto en rojo{Color.END}")
print(f"{Color.GREEN}Texto en verde{Color.END}")
print(f"{Color.YELLOW}Texto en amarillo{Color.END}")
print(f"{Color.BLUE}Texto en azul{Color.END}")
print(f"{Color.MAGENTA}Texto en magenta{Color.END}")
print(f"{Color.CYAN}Texto en cian{Color.END}") """


import sys

class CapturedOutput:
    def __init__(self):
        self._stdout = sys.stdout
        self._captured_output = []

    def start_capture(self):
        sys.stdout = self

    def stop_capture(self):
        sys.stdout = self._stdout

    def write(self, text):
        self._captured_output.append(text)

    def get_captured_output(self):
        return ''.join(self._captured_output)
    



USER_DATASETS_DIRECTORY_PATH = 'fca-decision-tree-classifier\\datasets\\user'


@app.route('/fca_tree_builder')
def fca_tree_builder():
    files = [f for f in os.listdir(USER_DATASETS_DIRECTORY_PATH) if os.path.isfile(os.path.join(USER_DATASETS_DIRECTORY_PATH, f))]
    return render_template('fca_tree_builder.html', files=files)







selected_average = 'weighted'


captured_output_metrics = None
captured_output_tree_logic_structure = None

@app.route('/build-Tree', methods=['POST'])
def build_Tree():

    global captured_output_metrics, captured_output_tree_logic_structure


    selected_file = request.form['file']
    discretizer_choice  = request.form['discretizer']


    random_state = request.form.get('random_state', '42')  # Establece 42 como valor predeterminado si no se proporciona nada en 'random_state'
    random_state = int(random_state)

    CV = request.form.get('CV', '1')  
    CV = int(CV)

    test_size = request.form.get('test_size', '0.2')  
    test_size = float(test_size)


    target_colum = request.form['target_colum']
    if not target_colum:  # usa la última columna por defecto
        target_colum = -1





    # Crear una instancia de CapturedOutput
    captured_output_metrics = CapturedOutput()
    # Iniciar la captura de la salida
    captured_output_metrics.start_capture()

    data_preparer = DataPreparer()

    attributes, concepts = data_preparer.prepare_csvfile_data(
        file_path=os.path.join(USER_DATASETS_DIRECTORY_PATH, selected_file),
        test_size=test_size,
        random_state=random_state,
        discretizer=discretizer_choice,
        target_colum=target_colum
    )

    print(f"Atributos: {attributes}")
    print(f"Conceptos: {concepts}")

    
    X_train, X_test, y_train, y_test = data_preparer.get_split_data()





    # Entrenar y evaluar un árbol de decisión C4.5

    # Crear una instancia de DecisionTreeC45
    decision_treec45 = DecisionTreeC45(attribute_names=attributes, concept_names=concepts, max_depth=4)

    # Entrenar el árbol de decisión con los datos de entrenamiento
    decision_treec45.train_tree(data_preparer.X_train.values.tolist(), data_preparer.y_train.tolist())

    # Calcular la precisión del árbol en el conjunto de prueba
    predictions_treec45 = decision_treec45.classify_instances(X_test.values.tolist())

    # Métricas de evaluación del árbol
    decision_treec45.evaluation_metrics(y_test, predictions_treec45, average='macro') 
    


    FCAtree = FCADecisionTree(attribute_names=attributes, concept_names=concepts)
    FCAtree.train_tree(data=X_train.values.tolist(), target=y_train.values.tolist())
    predictions_tree = FCAtree.classify_instances(X_test.values.tolist())


    # Métricas de evaluación del árbol
    FCAtree.evaluation_metrics(y_test, predictions_tree, average='macro')


    # Usar los otros modelos para compararlo
    otherModelsResults = tryOtherModels(X_train, X_test, y_train, y_test, CV, selected_average)


    captured_output_metrics.stop_capture()
    captured_output_metrics = captured_output_metrics.get_captured_output()



    captured_output_tree_logic_structure = CapturedOutput()
    # Imprimir la estructura del árbol de decisión
    captured_output_tree_logic_structure.start_capture()
    print(" \n\n\n   TREE NODE STRUCTURE: \n\n   ")
    decision_treec45.print_tree_structure()
    print(" \n\n##########################################################################################")
    print(" \n\n\n   TREE LOGIC DECISIONS : \n\n   ")
    decision_treec45.print_tree_logic()
    captured_output_tree_logic_structure.stop_capture()
    captured_output_tree_logic_structure = captured_output_tree_logic_structure.get_captured_output()

    return render_template('metrics_report.html', captured_output=captured_output_metrics)


@app.route('/tree_logic_structure_report')
def show_tree_logic_structure_report():
    global captured_output_tree_logic_structure 
    return render_template('tree_logic_structure_report.html', captured_output=captured_output_tree_logic_structure)


