import math
import numpy as np
from collections import Counter
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

class Node:
    def __init__(self, attribute=None, category=None, children=None, decision=None):
        self.attribute = attribute
        self.category = category
        self.children = {} if children is None else children
        self.decision = decision

class DecisionTreeC45:
    def __init__(self, attribute_names=None, concept_names=None, max_depth=None):
        self.root = None
        self.attribute_names = attribute_names
        self.concept_names = concept_names
        self.default_decision = None
        self.max_depth = max_depth

    def _induce_tree(self, data, target, attributes, depth=0):
        if len(set(target)) == 1:
            return Node(decision=target[0])

        if not data or not attributes or (self.max_depth is not None and depth >= self.max_depth):
            return Node(decision=Counter(target).most_common(1)[0][0])

        best_gain = -float('inf')
        best_attr, best_category = None, None
        for attr in attributes:
            categories = set(row[attr] for row in data)
            for category in categories:
                subset = [i for i, row in enumerate(data) if row[attr] == category]
                gain = self._information_gain(target, subset)
                if gain > best_gain:
                    best_gain, best_attr, best_category = gain, attr, category

        if best_gain <= 0:
            return Node(decision=Counter(target).most_common(1)[0][0])

        child_nodes = {}
        for category in set(row[best_attr] for row in data):
            child_data = [row for row in data if row[best_attr] == category]
            child_target = [target[i] for i, row in enumerate(data) if row[best_attr] == category]
            child_nodes[category] = self._induce_tree(child_data, child_target, attributes - {best_attr}, depth + 1)

        return Node(attribute=best_attr, children=child_nodes)

    def _information_gain(self, target, subset):
        def entropy(indices):
            counter = Counter([target[i] for i in indices])
            probs = [count / len(indices) for count in counter.values()]
            return -sum(p * math.log2(p) for p in probs if p > 0)  # Avoid log(0)

        total_entropy = entropy(range(len(target)))
        subset_entropy = entropy(subset)
        return total_entropy - (len(subset) / len(target)) * subset_entropy

    def train_tree(self, data, target):
        attributes = set(range(len(data[0])))
        self.root = self._induce_tree(data, target, attributes)
        self.default_decision = Counter(target).most_common(1)[0][0]

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

    def evaluation_metrics(self, y_test, predictions_tree, average):
        accuracy_tree = accuracy_score(y_test, predictions_tree)
        precision_tree = precision_score(y_test, predictions_tree, average=average)
        recall_tree = recall_score(y_test, predictions_tree, average=average)
        f1_tree = f1_score(y_test, predictions_tree, average=average)
        conf_matrix_tree = confusion_matrix(y_test, predictions_tree)
        class_report = classification_report(y_test, predictions_tree, target_names=None)

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