from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import pickle
import graphviz
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from sklearn.model_selection import KFold
import numpy as np
import time

from src.FormalContext import Concept, FormalContext
from src.DataPreparer import DataPreparer

class FCANode(Concept):
    def __init__(self, formal_concept):
        super().__init__(
            context=formal_concept.context,
            extent=formal_concept.extent,
            intent=formal_concept.intent,
            inmediate_subconcepts=formal_concept.inmediate_subconcepts
        )
        self.parent = formal_concept.parent
        self.label = formal_concept.label
        self.id = formal_concept.id

class FCADecisionTree:
    def __init__(self, formal_context, max_depth, label_method='max_extent', build_method=None, classes_threshold=None):
        
        if(len(formal_context.get_all_concepts_lattice()) == 0):
            raise ValueError("El retículo está vacío")
        
        self.build_method = build_method
        self.classes_threshold = classes_threshold
        
        self.formal_context = formal_context
        self.max_depth = max_depth

        self.sort_lattice_for_tree(label_method)

        self.set_root()
        self.build_tree(self.root, depth=0) # comienza a construir el árbol
        
    
    def set_root(self):
        # Seleccionar el concepto/supraconcepto/más informativo con el extent más grande como raíz
        # deja como padre al concepto com mayor extensión
        largest_extent_concept = self.sorted_lattice[0]
        self.root = FCANode(formal_concept=largest_extent_concept)
        # manejar el caso especial en el que el padre no tiene conceptos inmediatos, busca el siguiente más general
        if len(self.root.inmediate_subconcepts) == 0:
            for concept in self.sorted_lattice:
                if len(concept.inmediate_subconcepts) > 1:
                    self.root = FCANode(formal_concept=concept)
        self.root.label = "ROOT"

    def sort_lattice_for_tree(self, label_method):
        # Establece las etiquetas de los nodos
        self.formal_context.set_concept_lattice_labels_ids(method=label_method)
        # Ordenar convenientemente el retículo por el tamaño de la extensión cada concepto
        self.sorted_lattice = sorted(self.formal_context.get_all_concepts_lattice(), key=lambda concept: len(concept.extent), reverse=True) 


    def build_tree(self, fcanode, depth):
        """
        Construye el árbol de decisión recursivamente.
        """
        # Mientras no se llegue a un límite de profundidad o no haya hijos (es hoja)
        if depth >= self.max_depth or self.is_leaf(fcanode):
            if len(fcanode.extent)==0: return None #caso particular con ínfimo del retículo
            else: 
                return # No se puede seguir construyendo, dejar el nodo como hoja
        
        # Comprobar los hijos del nodo actual
        children = []
        for child in fcanode.inmediate_subconcepts:
            child_node = FCANode(formal_concept=child)     
            children.append(child_node)
            
        # Asigna la nueva lista de nodos hijos al nodo actual
        fcanode.inmediate_subconcepts = children
        

        # MÉTODO DE simplificación-poda
        if(self.classes_threshold == None):
            # Verificar si los hijos tienen la misma etiqueta de clase para la poda
            if children:
                if all(child.label == fcanode.label for child in children):
                    fcanode.inmediate_subconcepts = []  # Eliminar los hijos del nodo actual
                    return
        else:
            if children:
                # Verificar si el porcentaje de hijos con la misma etiqueta que el padre supera el umbral
                threshold_count = int(len(children) * self.classes_threshold)
                matching_children = sum(1 for child in children if child.label == fcanode.label)
                
                if matching_children >= threshold_count:
                    fcanode.inmediate_subconcepts = []  # Eliminar los hijos del nodo actual
                    return
            
        
        # Si no hay stop, continuar construyendo recursivamente
        for child_node in children:
            self.build_tree(child_node, depth + 1)

    def is_leaf(self, fcanode):
        return len(fcanode.inmediate_subconcepts) == 0
    
    def fit_with_new_lattice(self, new_lattice, label_method='max_extent'):
        """
        Actualiza el árbol de decisión con un nuevo retículo.
        """
        self.formal_context.add_concepts(new_lattice)

        self.sort_lattice_for_tree(label_method)
        
        # Seleccionar el concepto con la extensión más grande como raíz
        largest_extent_concept = self.formal_context.get.lattice[0]
        self.root = FCANode(formal_concept=largest_extent_concept)
        self.set_root()
        # Reconstruir el árbol
        self.build_tree(self.root, 0)
        self.root.label = "ROOT"

    def classify(self, attributes_instance, debug=False):
        """
        Clasifica una nueva instancia usando el árbol de decisión.
        
        Parameters:
        - attributes_instance: El vector de características o diccionario a clasificar.
        - debug: Booleano para habilitar la impresión del flujo de clasificación.
        
        Returns:
        - La etiqueta de clase.
        """
        if(debug):
            print("\nVector a clasificar: ", attributes_instance)

        node = self.root
        level = 0
        while node.inmediate_subconcepts:
            best_match = None
            best_similarity = 0
            for child in node.inmediate_subconcepts:
                similarity = self.compute_similarity(child, attributes_instance)
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match = child
            
            if debug:
                prefix = '│   ' * level
                print(f"{prefix}└──Nivel {level}: Nodo actual (Intent: {node.intent}, Label: {node.label}), Mejor coincidencia (Intent: {best_match.intent if best_match else None})")

            if best_match is None:
                break
            node = best_match
            level += 1

        if debug:
            print(f"Resultado final: Nodo (Intent: {node.intent}, Label: {node.label})")

        return node.label
    
    def classify_instances(self, instances):
        fca_predicted_labels = []

        for i in range(len(instances)):
            test_vector = instances.iloc[i]
            if isinstance(test_vector, pd.Series):
                test_vector = test_vector[test_vector == 1].index.tolist()
            prediction = self.classify(test_vector, debug=False)
            fca_predicted_labels.append(prediction)

        return fca_predicted_labels

    def compute_similarity(self, fcanode, target_attributes):
        """
        Calcula la similitud entre la intención del nodo y los atributos proporcionados.
        """
        node_intent_attributes = {self.formal_context.attributes[i] for i in fcanode.intent}
        
        if all(isinstance(attr, (int, bool)) for attr in target_attributes):
            input_attributes = {attr for attr, val in zip(self.formal_context.attributes, target_attributes) if val}
        else:
            input_attributes = set(target_attributes)
        
        common_attributes = node_intent_attributes.intersection(input_attributes)
        return len(common_attributes)

    def evaluate(self, X_test, y_test, plot_results=False, debug=False):
        if len(X_test) != len(y_test):
            raise ValueError("X_test y y_test deben tener el mismo tamaño")

        predicted_labels = []

        y_test_labels = y_test.tolist()

        for i in range(len(X_test)):
            test_vector = X_test.iloc[i]
            if isinstance(test_vector, pd.Series):
                test_vector = test_vector[test_vector == 1].index.tolist()
            prediction = self.classify(test_vector, debug=debug)
            predicted_labels.append(prediction)

        accuracy = accuracy_score(y_test_labels, predicted_labels)

        # Calcular métricas por clase
        precision_per_class = precision_score(y_test_labels, predicted_labels, average=None, labels=list(set(y_test_labels + predicted_labels)), zero_division=1)
        recall_per_class = recall_score(y_test_labels, predicted_labels, average=None, labels=list(set(y_test_labels + predicted_labels)), zero_division=1)
        f1_per_class = f1_score(y_test_labels, predicted_labels, average=None, labels=list(set(y_test_labels + predicted_labels)), zero_division=1)

        # Calcular matriz de confusión
        conf_matrix = confusion_matrix(y_test_labels, predicted_labels, labels=list(set(y_test_labels + predicted_labels)))

        metrics = {
            "Accuracy": accuracy,
            "Precision": dict(zip(list(set(y_test_labels + predicted_labels)), precision_per_class)),
            "Recall": dict(zip(list(set(y_test_labels + predicted_labels)), recall_per_class)),
            "F1-Score": dict(zip(list(set(y_test_labels + predicted_labels)), f1_per_class)),
            "Confusion Matrix": conf_matrix
        }

        if plot_results:
            self.plot_metrics(metrics)
            self.plot_confusion_matrix(conf_matrix, list(set(y_test_labels + predicted_labels)))

        return metrics


    def save_instance(self, file_path):
        """
        Guarda la instancia actual en un archivo utilizando pickle.

        :param file_path: Ruta del archivo donde se guardará la instancia.
        """
        with open(file_path, 'wb') as f:
            pickle.dump(self, f)








    ############################## FUNCIONES DE VISUALIZACIÓN ##############################
    def print_tree_complete(self, node=None, depth=0):
        if node is None:
            node = self.root

        # Imprimir el nodo actual con la indentación correspondiente a la profundidad
        indent = "    " * depth
        print(f"{indent}Node: Intent: {node.intent}, Extent: {node.extent}, Label: {node.label}")

        if(isinstance(node, FCANode)):
            # Recorrer recursivamente los hijos
            for child in node.inmediate_subconcepts:
                self.print_tree_complete(child, depth + 1)
        else:
            node = FCANode(formal_concept=node)
            for child in node.inmediate_subconcepts:
                self.print_tree_complete(child, depth + 1)
                
    def print_tree_logic(self, node=None, depth=0):
        if node is None:
            node = self.root

        # Definir el prefijo para la indentación
        prefix = '│   ' * depth

        # Imprimir el nodo actual
        print(f"{prefix}└── [{node.id} - class-label: {node.label}]")

        # Recorrer recursivamente los hijos
        for child in node.inmediate_subconcepts:
            self.print_tree_logic(child, depth + 1)

    def print_tree_structure(self, node=None, depth=0):
        """Imprime la estructura del árbol de decisión."""
        if node is None:
            node = self.root

        prefix = '│   ' * depth

        if node.label:
            print(f"{prefix}└── [Decision leaf node: {node.label})]")
            return

        print(f"{prefix}└── [Branch Node with attribute: {node.attribute}]")
        for child_value, child_node in node.inmediate_subconcepts.items():
            print(f"{prefix}├── Value: {child_value}")
            self.print_tree_structure(child_node, depth + 1)


    def get_tree_complexity(self, node=None, depth=0):
        if node is None:
            node = self.root

        total_nodes = 1
        leaf_nodes = 0 if node.inmediate_subconcepts else 1
        max_depth = depth
        
        for child in node.inmediate_subconcepts:
            child_total_nodes, child_leaf_nodes, child_max_depth = self.get_tree_complexity(child, depth + 1)
            total_nodes += child_total_nodes
            leaf_nodes += child_leaf_nodes
            max_depth = max(max_depth, child_max_depth)

        return total_nodes, leaf_nodes, max_depth

    def display_tree_complexity(self):
        total_nodes, leaf_nodes, max_depth = self.get_tree_complexity()
        avg_inmediate_subconcepts = (total_nodes - leaf_nodes) / (total_nodes - leaf_nodes + 1)  # +1 to avoid division by zero

        complexity_info = {
            "Tree Depth": max_depth,
            "Total Nodes": total_nodes,
            "Leaf Nodes": leaf_nodes,
            "Average inmediate_subconcepts per Node": avg_inmediate_subconcepts
        }

        return complexity_info
    
    def plot_metrics(self, metrics):
        labels = list(metrics["Precision"].keys())
        precision = list(metrics["Precision"].values())
        recall = list(metrics["Recall"].values())
        f1 = list(metrics["F1-Score"].values())

        x = range(len(labels))

        plt.figure(figsize=(12, 6))
        plt.bar(x, precision, width=0.2, label='Precision', align='center')
        plt.bar([p + 0.2 for p in x], recall, width=0.2, label='Recall', align='center')
        plt.bar([p + 0.4 for p in x], f1, width=0.2, label='F1-Score', align='center')

        plt.xlabel('Classes')
        plt.ylabel('Scores')
        plt.title('Precision, Recall and F1-Score per class')
        plt.xticks([p + 0.2 for p in x], labels)
        plt.legend()
        plt.show()

    def plot_confusion_matrix(self, conf_matrix, labels):
        plt.figure(figsize=(10, 7))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix')
        plt.show()

    def create_tree_diagram(self, output_filename):
        dot = graphviz.Digraph(comment='Decision Tree Diagram')

        def add_node(dot, node):
            label = f'ID: {node.id}\nLabel: {node.label if node.label else "None"}'
            dot.node(str(node.id), label)
            for child in node.inmediate_subconcepts:
                edge_label = f'Intent: {child.intent if child.intent else "None"}'
                dot.edge(str(node.id), str(child.id), label=edge_label)
                add_node(dot, child)

        add_node(dot, self.root)
        dot.render(output_filename, format='png')






