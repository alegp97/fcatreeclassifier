from typing import List, Tuple, Union
import pandas as pd
""" class Node:
    def __init__(self, concept: formalConcept, index: int):
        self.index = index
        self.concept = concept
        self.children = []
        self.class_label = None

    def is_leaf(self):
        return len(self.children) == 0

    def add_child(self, node):
        self.children.append(node)

    def get_child(self, index):
        return self.children[index]

class Lattice:
    def __init__(self, data: pd.DataFrame = None):
        self.count = 0
        self.concepts = []
        if data is not None:
            self.initialize_from_data(data)

    def initialize_from_data(self, data: pd.DataFrame):
        for row in data.iterrows():
            concept = formalConcept(row[0], row[1])
            self.add(concept)

    def get_concepts(self) -> List[Node]:
        return self.concepts

    def get_size(self) -> int:
        return self.count

    def get_formal_concepts(self) -> List[formalConcept]:
        return [c.concept for c in self.concepts]

    def get_concept(self, i: int) -> Node:
        return self.concepts[i]

    def add(self, concept: formalConcept):
        self.concepts.append(Node(concept, self.count))
        self.count += 1

    def replace(self, f: formalConcept, newf: formalConcept):
        for i, c in enumerate(self.concepts):
            if c.concept == f:
                self.concepts[i].concept = newf

    def get_index(self, concp: formalConcept) -> int:
        for i, c in enumerate(self.concepts):
            if c.concept == concp:
                return i
        return -1

    def find(self, concp: formalConcept) -> bool:
        return self.get_index(concp) != -1

    def get_parents(self, c: formalConcept) -> List[formalConcept]:
        parents = []
        for node in self.concepts:
            if c[0] in node.concept[1] and node.concept != c:
                parents.append(node.concept)
        return parents

    def get_concepts_above(self, c: formalConcept) -> List[formalConcept]:
        above = []
        for node in self.concepts:
            if set(c[1]).issubset(set(node.concept[1])) and node.concept != c:
                above.append(node.concept)
        return above

    def next(self, c: formalConcept) -> Union[formalConcept, None]:
        for i, node in enumerate(self.concepts):
            if node.concept == c:
                if i + 1 < len(self.concepts):
                    return self.concepts[i + 1].concept
                else:
                    return None
        return None

 """



