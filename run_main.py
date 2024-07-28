'''
To implement the methods exposed in the scientific publication, we need to understand the approach described in the paper. The paper presents a method of decision tree induction based on formal concept analysis (FCA). The decision tree is derived using a concept lattice, which is a hierarchy of clusters provided by FCA. The idea is to view the concept lattice as a collection of overlapping trees and select one of these trees as the decision tree.

Based on the information provided in the paper, we can outline the steps to implement the method:

Data Transformation: Convert the input data with categorical attributes to logical attributes suitable for FCA. This involves scaling the attributes appropriately.

Building the Concept Lattice: Use the transformed data to build the concept lattice. Each formal concept in the lattice represents a collection of objects with common attributes.

Selection of Decision Tree: Select one of the overlapping trees from the concept lattice as the decision tree. The selection criteria are not explicitly mentioned in the abstract, but you can choose a suitable method based on your understanding of the paper.
'''



### LOS FICHEROS EN SNAKE CASE Y LAS CLASES Y FUNCIONES EN CAMELCASE
""" from src.data_preparer import DataPreparer
from src.decision_tree import DecisionTree """



from web_app import app

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)





    

        @classmethod
    def test_CV(cls, discretizer=None, test_size=None, DATA_CSV_FILE_PATH=None, random_state=None, thresholds=None, k=None, max_depth=None, debug=False):
        # Preparar datos para validación cruzada
        data_preparer = DataPreparer()
        X, y = data_preparer.prepare_csvfile_data(DATA_CSV_FILE_PATH, thresholds=thresholds, discretizer=discretizer, CV=True)
        X = np.array(X)
        y = np.array(y)
        # Configurar KFold
        kf = KFold(n_splits=k, shuffle=True, random_state=random_state)
        accuracies = []
        precisions = []
        recalls = []
        f1s = []
        execution_times = []
        
        # Iterar sobre cada pliegue definido por KFold
        for train_index, test_index in kf.split(X):
            # Obtener los subconjuntos de datos para este pliegue
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            
            # Convertir de vuelta a DataFrames de pandas
            X_train_df = pd.DataFrame(X_train, columns=data_preparer.attributes)
            y_train_df = pd.DataFrame(y_train, columns=['objective_target'])
            X_test_df = pd.DataFrame(X_test, columns=data_preparer.attributes)


            # Crear y preparar el contexto formal
            labeled_train_data = pd.concat([X_train_df, y_train_df], axis=1)
            print("AAAAAAAAAAAA: ", labeled_train_data)
            formal_context = FormalContext(labeled_train_data)
            formal_context.build_lattice(debug=debug, compute_parents_childs=True)

            # Inicializar y evaluar el modelo FCA
            start_time = time.time()
            model = cls(formal_context=formal_context, max_depth=max_depth)
            results = model.evaluate(X_test_df, y_test, debug=debug, plot_results=False)
            end_time = time.time()

            # Recopilar métricas de cada pliegue
            accuracies.append(results["Accuracy"])
            precisions.append(results["Precision"])
            recalls.append(results["Recall"])
            f1s.append(results["F1-Score"])
            execution_times.append(end_time - start_time)

        # Imprimir resultados agregados
        print(f"Modelo: FCADecisionTree")
        print(f"Exactitud media: {np.mean(accuracies):.4f}")
        print(f"Precisión media: {np.mean(precisions):.4f}")
        print(f"Recall media: {np.mean(recalls):.4f}")
        print(f"F1-Score media: {np.mean(f1s):.4f}")
        print(f"Tiempo de ejecución promedio: {np.mean(execution_times):.4f} segundos")
        print(f"Desviación estándar de la Exactitud: {np.std(accuracies):.4f}")
