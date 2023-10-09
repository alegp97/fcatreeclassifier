
import math
from collections import Counter

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report



class Node:
    def __init__(self, attribute=None, category=None, children=None, decision=None):
        self.attribute = attribute
        self.category = category
        self.children = {} if children is None else children
        self.decision = decision


class Concept:
    def __init__(self, extent, intent):
        self.extent = extent
        self.intent = intent
        self.children = []

class FCADecisionTree:
    def __init__(self, attribute_names=None, concept_names=None, balance=False):
        self.root = None
        self.context = None  # Atributo para almacenar el contexto formal
        self.attribute_names = attribute_names
        self.concept_names = concept_names
        self.default_decision = None
        self.balance = balance
        self.concepts = []
  


    def _in_close(self, context, c, y, attribute_names):
        new_intent = set(attribute_names)
        for g in c.extent:
            new_intent &= set([attribute_names[i] for i, val in enumerate(context[g]) if val])

        if new_intent == c.intent:
            for j in range(y + 1, len(attribute_names)):
                if all(context[g][j] for g in c.extent):
                    new_extent = [g for g in c.extent if context[g][j]]
                    new_intent = c.intent | {attribute_names[j]}
                    new_concept = Concept(new_extent, new_intent)

                    if new_concept not in self.concepts:
                        print(f"Adding new concept with extent {new_extent} and intent {new_intent}")  # Debugging statement
                        c.children.append(new_concept)
                        self.concepts.append(new_concept)
                        self._in_close(context, new_concept, j, attribute_names)



    def construct_lattice(self, context, attribute_names):
        self.concepts = []
        top_concept = Concept(list(range(len(context))), set())
        self.concepts.append(top_concept)
        self._in_close(context, top_concept, -1, attribute_names)
        return top_concept
    
    def _select_concept_tree(self, concept):
        if not concept.children:
            return concept

        class_distributions = [Counter([self.concept_names[i] for i in child.extent if 0 <= i < len(self.concept_names)]) for child in concept.children]

        if self.balance:
            most_balanced_child = max(concept.children, key=lambda c: -abs(class_distributions[concept.children.index(c)]['yes'] - class_distributions[concept.children.index(c)]['no']))
            return most_balanced_child
        else:
            most_pure_child = max(concept.children, key=lambda c: max(class_distributions[concept.children.index(c)].values()))
            return most_pure_child



    def _transform_to_decision_tree(self, concept, attribute_names):
        if not concept.children:
            valid_indices = [i for i in concept.extent if 0 <= i < len(self.concept_names)]
            if not valid_indices:
                return Node(decision=self.default_decision)  # Use the default decision if there are no valid instances

            return Node(decision=Counter([self.concept_names[i] for i in valid_indices]).most_common(1)[0][0])

        node = Node(attribute=attribute_names[concept.intent.pop()])  # Use pop() to get an attribute from the intent
        for child in concept.children:
            node.children[self.attribute_names[child.intent.pop()]] = self._transform_to_decision_tree(child, attribute_names)
        return node

    def train_tree(self, data, target):
        self.context = data
        top_concept = self.construct_lattice(self.context, self.attribute_names)
        concept_tree_root = self._select_concept_tree(top_concept)
        self.root = self._transform_to_decision_tree(concept_tree_root, self.attribute_names)
        self.default_decision = Counter(target).most_common(1)[0][0]

        


    def lattice_size(self):
        """Devuelve el número de conceptos en el retículo."""
        return len(self.concepts)

    def lattice_class_distribution(self):
        """Devuelve la distribución de clases en el retículo."""
        distributions = [Counter([self.concept_names[i] for i in concept.extent if 0 <= i < len(self.concept_names)]) for concept in self.concepts]
        overall_distribution = Counter()
        for dist in distributions:
            overall_distribution += dist
        return overall_distribution

    def lattice_max_depth(self, concept=None, current_depth=0):
        """Devuelve la profundidad máxima del retículo."""
        if concept is None:
            concept = self.construct_lattice(self.context, self.attribute_names)
        if not concept.children:
            return current_depth
        return max(self.lattice_max_depth(child, current_depth + 1) for child in concept.children)



















    def classify(self, instance, node=None):
        if node is None:
            node = self.root
        if node.decision is not None:
            return node.decision
        if instance[node.attribute] in node.children:
            return self.classify(instance, node.children[instance[node.attribute]])
        else:
            # Si la instancia tiene una categoría no vista en el entrenamiento, 
            # devolvemos la decisión predeterminada
            return self.default_decision

    def classify_instances(self, instances):
        return [self.classify(instance) for instance in instances]

    def score(self, data, target):
        predictions = self.classify_instances(data)
        return sum(1 for pred, true in zip(predictions, target) if pred == true) / len(target)
    

    def evaluation_metrics(self, y_test, predictions_tree, average):
        accuracy_tree = accuracy_score(y_test, predictions_tree)
        
        # Agregar el parámetro zero_division a precision_score y f1_score
        precision_tree = precision_score(y_test, predictions_tree, average=average, zero_division=0)
        recall_tree = recall_score(y_test, predictions_tree, average=average)
        f1_tree = f1_score(y_test, predictions_tree, average=average, zero_division=0)
        
        conf_matrix_tree = confusion_matrix(y_test, predictions_tree)
        class_report = classification_report(y_test, predictions_tree, target_names=None, zero_division=0)  # También aquí

        # Imprimir las métricas
        print("\n\n////////////////////////////////////   FCA TREE EVALUATION METRICS   /////////////////////////////////////////////////////")
        print("Accuracy:", accuracy_tree)
        print("Precision:", precision_tree)
        print("Recall:", recall_tree)
        print("F1 Score:", f1_tree)
        print("Confusion Matrix:\n", conf_matrix_tree)
        print("Classification Report:\n", class_report)
        print("\n//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////")
        print("\n") 
 



    def print_tree_logic(self, node=None, depth=0, condition=''):

        """Imprime la lógica del árbol de decisión."""
        if node is None:
            node = self.root

        # Si el nodo es una hoja (nodo de decisión)
        if node.decision is not None:
            decision_name = self.concept_names[node.decision] if isinstance(node.decision, int) else node.decision
            print(f"{condition} => Decision: {decision_name}")
            return

        # Si el nodo es un nodo interno
        for category, child_node in node.children.items():
            prefix = ' ' * depth * 4  # Indentación para una mejor visualización
            new_condition = f"{prefix}IF {self.attribute_names[node.attribute]} = {category}"
            if condition:
                new_condition = f"{condition} AND\n{new_condition}"
            self.print_tree_logic(child_node, depth + 1, new_condition)



    
    def print_tree_structure(self, node=None, depth=0):
        """Imprime la estructura del árbol de decisión en forma de estructura de directorio."""

        if node is None:
            node = self.root

        prefix = '│   ' * depth  # Indentación para una mejor visualización

        # Si el nodo es una hoja (nodo de decisión)
        if node.decision is not None:
            if self.concept_names:
                if isinstance(node.decision, int) and 0 <= node.decision < len(self.concept_names):
                    decision_name = self.concept_names[node.decision]
                else:
                    decision_name = node.decision
            else:
                decision_name = node.decision
            print(f"{prefix}└── [Decision leaf node: {decision_name}]")
            return

        # Si el nodo es un nodo interno
        attribute_name = self.attribute_names[node.attribute] if self.attribute_names else node.attribute
        print(f"{prefix}└── [Attribute: {attribute_name}]")
        for idx, (category, child_node) in enumerate(node.children.items()):
            if idx < len(node.children) - 1:
                print(f"{prefix}├── Category: {category}")
                self.print_tree_structure(child_node, depth + 1)
            else:
                print(f"{prefix}└── Category: {category}")
                self.print_tree_structure(child_node, depth + 1)




    def explain_classification(self, instance, node=None, logic=None):
        """Explica la clasificación de una instancia dada mostrando la lógica de decisión."""
        if node is None:
            node = self.root
            logic = []

        # Si el nodo es una hoja (nodo de decisión)
        if node.decision is not None:
            decision_name = self.concept_names[node.decision] if (self.concept_names is not None and len(self.concept_names) > 0) else node.decision
            logic.append(f"Decision: {decision_name}")
            return logic

        # Si el nodo es un nodo interno
        attribute_name = self.attribute_names[node.attribute] if self.attribute_names else node.attribute
        if instance[node.attribute] in node.children:
            logic.append(f"IF {attribute_name} = {instance[node.attribute]}")
            return self.explain_classification(instance, node.children[instance[node.attribute]], logic)
        else:
            # Si la instancia tiene una categoría no vista en el entrenamiento, devolvemos la lógica hasta ahora y la clase más común de los hijos
            logic.append(f"IF {attribute_name} = {instance[node.attribute]} (unseen category)")
            decision = Counter(child.decision for child in node.children.values()).most_common(1)[0][0]
            decision_name = self.concept_names[decision] if (self.concept_names is not None and len(self.concept_names) > 0) else decision
            logic.append(f"Decision (based on most common child): {decision_name}")
            return logic











