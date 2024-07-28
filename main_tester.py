

# Luego, puedes usar el siguiente código para generar y mostrar los conceptos:
from src.DataPreparer import DataPreparer
from src.FormalContext import FormalContext
from src import FCADecisionTree as FCADtree


import pandas as pd

data_preparer = DataPreparer()
data_preparer.prepare_csvfile_data(
    file_path= 'C:\\Users\\Focus\\Desktop\\TFG\\TFG\\fca-decision-tree-classifier\\datasets\\real_datasets\\car_acceptability.csv',
    test_size=0.2,
    random_state=42,
    discretizer='equal-width',
    n_bins = 5,
    target_colum=-1
)

data_preparer.binarize()   


X_train, X_test, y_train, y_test = data_preparer.get_split_data()
labeled_train_data = pd.concat([X_train, y_train.reset_index(drop=True)], axis=1)



objects = list(labeled_train_data.index)
properties = [prop for prop in labeled_train_data.columns if prop != 'objective_target']
num_objects = len(objects)
num_properties = len(properties)
max_context_size = num_objects * num_properties



print(f"Number of Objects: {num_objects}")
print(f"Number of Properties-Attributes: {num_properties}")
print(f"Max context Size: {max_context_size}")   



formal_context = FormalContext(labeled_data=labeled_train_data)
formal_context.build_lattice(debug=True, compute_parents_childs=True)



print("\nFinalizando Algoritmo:")
print("Número de objetos:", len(objects))
print("Número de atributos:", len(properties))
print("Número de conceptos formales:", len(formal_context.get_all_concepts_lattice()))

""" 
formal_context.print_lattice()
formal_context.print_contingency_table()
formal_context.display_concept_parents()
print("\n")
formal_context.display_concept_children() 

 """

model = FCADtree.FCADecisionTree(formal_context=formal_context, max_depth=10)

#model.print_tree_complete()
model.print_tree_logic() 

complexity_info = model.display_tree_complexity()
print(complexity_info)


print(X_test)

print(y_test)
""" 
print(X_test.info())

print(y_test.info())
 """


""" model.create_tree_diagram(output_filename="FCAdecision_tree_iris_5bins")
formal_context.create_hasse_diagram(output_filename="hasse_diagram_iris_5bins") """


results = model.evaluate(X_test, y_test, debug=False, plot_results=True)
print(results) 


