import time

from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb

from src.DataPreparer import DataPreparer

classifiers_for_comparing = {
    'Logistic Regression': LogisticRegression(solver='lbfgs', multi_class='ovr', max_iter=10000),
    'SVM': SVC(),
    'Decision Tree': DecisionTreeClassifier(),
    'K-Nearest Neighbors': KNeighborsClassifier(),
    'Random Forest': RandomForestClassifier(),
    'XGBoost': xgb.XGBClassifier()
}

def evaluate_classifier(classifier, X_train, y_train, X_test, y_test, selected_average):
    start_time = time.time()
    
    # Entrenar el clasificador
    classifier.fit(X_train, y_train)
    
    # Realizar predicciones
    y_pred = classifier.predict(X_test)
    
    # Imprimir métricas
    print(f"Classifier: {type(classifier).__name__}")
    print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
    print(f"Precision: {precision_score(y_test, y_pred, average='weighted')}")
    print(f"Recall: {recall_score(y_test, y_pred, average='weighted')}")
    print(f"F1 Score: {f1_score(y_test, y_pred, average='weighted')}")
    print(f"Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}")
    
    end_time = time.time()
    
    # Imprimir el tiempo de ejecución
    print(f"Execution Time: {end_time - start_time} seconds")
    print("--------------------------------------------------")

def tryOtherModels(X_train, X_test, y_train, y_test, CV, selected_average):

    # Codificación One-Hot para las características categóricas en X_train y X_test.
    encoder = OneHotEncoder(drop='first', sparse=False, handle_unknown='ignore')
    X_train_encoded = encoder.fit_transform(X_train)
    X_test_encoded = encoder.transform(X_test)

    # Codificación de las etiquetas de clase (y_train y y_test) en valores numéricos.
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_test_encoded = label_encoder.transform(y_test)

    for name, classifier in classifiers_for_comparing.items():
        print(f"Evaluating model: {name}")

        # Usamos las versiones codificadas de X_train, X_test, y_train y y_test para todos los modelos.
        X_train_model = X_train_encoded
        X_test_model = X_test_encoded
        y_train_model = y_train_encoded
        y_test_model = y_test_encoded

        # Si se ha especificado la validación cruzada (CV), se realiza para cada modelo.
        if CV and CV > 1:
            scores = cross_val_score(classifier, X_train_model, y_train_model, cv=CV, scoring='accuracy')
            mean_accuracy = scores.mean()
            std_accuracy = scores.std()
            print(f'Cross-Validation Mean Accuracy: {mean_accuracy}')
            print(f'Cross-Validation Std Accuracy: {std_accuracy}')
        
        evaluate_classifier(classifier, X_train_model, y_train_model, X_test_model, y_test_model, selected_average)
