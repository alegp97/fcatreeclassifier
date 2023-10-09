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





    

    