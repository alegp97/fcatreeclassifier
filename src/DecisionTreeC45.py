import math
from collections import Counter
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.model_selection import KFold
import numpy as np


class Node:
    def __init__(self, attribute=None, threshold=None, category=None, children=None, decision=None):
        self.attribute = attribute
        self.threshold = threshold
        self.category = category
        self.children = {} if children is None else children
        self.decision = decision

class DecisionTreeC45:
    def __init__(self, attribute_names=None, max_depth=None):
        self.root = None
        self.attribute_names = attribute_names
        self.default_decision = None
        self.max_depth = max_depth
        self.thresholds = {}  # Almacenar los puntos de corte

    def construct_tree(self, data, target, attributes, depth=0):
        if len(set(target)) == 1:
            return Node(decision=target[0])

        if not data or not attributes or (self.max_depth is not None and depth >= self.max_depth):
            return Node(decision=Counter(target).most_common(1)[0][0])

        best_gain_ratio = -float('inf')
        best_attr, best_threshold = None, None

        for attr in attributes:
            thresholds = self._find_thresholds(data, attr)
            for threshold in thresholds:
                gain_ratio = self._gain_ratio(data, target, attr, threshold)
                if gain_ratio > best_gain_ratio:
                    best_gain_ratio, best_attr, best_threshold = gain_ratio, attr, threshold

        if best_gain_ratio <= 0:
            return Node(decision=Counter(target).most_common(1)[0][0])

        if best_threshold is not None:
            if best_attr not in self.thresholds:
                self.thresholds[best_attr] = []
            self.thresholds[best_attr].append(best_threshold)

            left_data = [row for row in data if row[best_attr] <= best_threshold]
            left_target = [target[i] for i, row in enumerate(data) if row[best_attr] <= best_threshold]
            right_data = [row for row in data if row[best_attr] > best_threshold]
            right_target = [target[i] for i, row in enumerate(data) if row[best_attr] > best_threshold]
            children = {
                'left': self.construct_tree(left_data, left_target, attributes, depth + 1),
                'right': self.construct_tree(right_data, right_target, attributes, depth + 1)
            }
            return Node(attribute=best_attr, threshold=best_threshold, children=children)
        else:
            child_nodes = {}
            for category in set(row[best_attr] for row in data):
                child_data = [row for row in data if row[best_attr] == category]
                child_target = [target[i] for i, row in enumerate(data) if row[best_attr] == category]
                child_nodes[category] = self.construct_tree(child_data, child_target, attributes - {best_attr}, depth + 1)
            return Node(attribute=best_attr, children=child_nodes)

    def _find_thresholds(self, data, attr):
        values = sorted(set(row[attr] for row in data))
        return [(values[i] + values[i + 1]) / 2 for i in range(len(values) - 1)]

    def _gain_ratio(self, data, target, attr, threshold):
        def entropy(subset):
            counter = Counter(subset)
            probs = [count / len(subset) for count in counter.values()]
            return -sum(p * math.log2(p) for p in probs if p > 0)

        total_entropy = entropy(target)
        left_target = [target[i] for i, row in enumerate(data) if row[attr] <= threshold]
        right_target = [target[i] for i, row in enumerate(data) if row[attr] > threshold]

        subset_entropy = (len(left_target) / len(target)) * entropy(left_target) + (len(right_target) / len(target)) * entropy(right_target)
        information_gain = total_entropy - subset_entropy

        split_info = 0
        for subset in [left_target, right_target]:
            prob = len(subset) / len(target)
            if prob > 0:
                split_info -= prob * math.log2(prob)

        if split_info == 0:
            return 0  # Avoid division by zero
        return information_gain / split_info

    def train_tree(self, data, target):
        attributes = set(range(len(data[0])))
        self.root = self.construct_tree(data, target, attributes)
        self.default_decision = Counter(target).most_common(1)[0][0]

    def classify(self, instance, node=None):
        if node is None:
            node = self.root
        if node.decision is not None:
            return node.decision
        if node.threshold is not None:
            if instance[node.attribute] <= node.threshold:
                return self.classify(instance, node.children['left'])
            else:
                return self.classify(instance, node.children['right'])
        if instance[node.attribute] in node.children:
            return self.classify(instance, node.children[instance[node.attribute]])
        return self.default_decision

    def classify_instances(self, instances):
        return [self.classify(instance) for instance in instances]
    

    @classmethod
    def test_CV(cls, X, y, random_state, attribute_names, k=5, max_depth=None):
        kf = KFold(n_splits=k, shuffle=True, random_state=random_state)
        accuracies = []
        precisions = []
        recalls = []
        f1s = []

        # Asegurarse de que X y y son arrays de numpy
        X = np.array(X)
        y = np.array(y)

        for train_index, test_index in kf.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            # Convertir los datos a listas
            X_train_list = X_train.tolist()
            y_train_list = y_train.tolist()
            X_test_list = X_test.tolist()
            y_test_list = y_test.tolist()

            # Inicializar el modelo
            model = cls(attribute_names=attribute_names, max_depth=max_depth)

            # Entrenar el modelo
            model.train_tree(X_train_list, y_train_list)

            # Clasificar las instancias de prueba
            y_pred = model.classify_instances(X_test_list)

            # Evaluar las métricas
            accuracies.append(accuracy_score(y_test_list, y_pred))
            precisions.append(precision_score(y_test_list, y_pred, average='macro'))
            recalls.append(recall_score(y_test_list, y_pred, average='macro'))
            f1s.append(f1_score(y_test_list, y_pred, average='macro'))

        print(f"Exactitud media: {np.mean(accuracies):.4f}")
        print(f"Precisión media: {np.mean(precisions):.4f}")
        print(f"Recall media: {np.mean(recalls):.4f}")
        print(f"F1-Score media: {np.mean(f1s):.4f}")
        print(f"Desviación estándar de la Exactitud: {np.std(accuracies):.4f}")

    def evaluation_metrics(self, true_labels, predicted_labels, average='macro'):
        accuracy = accuracy_score(true_labels, predicted_labels)
        precision = precision_score(true_labels, predicted_labels, average=average)
        recall = recall_score(true_labels, predicted_labels, average=average)
        f1 = f1_score(true_labels, predicted_labels, average=average)
        conf_matrix = confusion_matrix(true_labels, predicted_labels)
        class_report = classification_report(true_labels, predicted_labels)

        print(f"Accuracy: {accuracy}")
        print(f"Precision: {precision}")
        print(f"Recall: {recall}")
        print(f"F1 Score: {f1}")
        print(f"Confusion Matrix:\n{conf_matrix}")
        print(f"Classification Report:\n{class_report}")

    def print_tree_structure(self, node=None, indent=""):
        if node is None:
            node = self.root
        if node.decision is not None:
            print(indent + "Decision: " + str(node.decision))
        else:
            if node.threshold is not None:
                print(indent + f"Attribute: {self.attribute_names[node.attribute]} <= {node.threshold}")
                self.print_tree_structure(node.children['left'], indent + "  ")
                self.print_tree_structure(node.children['right'], indent + "  ")
            else:
                print(indent + "Attribute: " + str(self.attribute_names[node.attribute]))
                for category, child in node.children.items():
                    print(indent + "Category: " + str(category))
                    self.print_tree_structure(child, indent + "  ")

    def get_thresholds(self):
        return self.thresholds
