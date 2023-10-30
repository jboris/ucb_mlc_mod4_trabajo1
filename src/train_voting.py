from sklearnex import patch_sklearn
patch_sklearn()

import argparse
from pathlib import Path
from uuid import uuid4
from datetime import datetime
import os
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix

import mlflow
import mlflow.sklearn

def main():
    # obtener parámetros:
    parser = argparse.ArgumentParser("train_voting")
    parser.add_argument("--dataset_path", type=str, help="File path to training data")
    parser.add_argument("--n_estimators", type=int, help="Estimators")
    parser.add_argument("--voting", type=str, help="Voting: soft or hard")

    args = parser.parse_args()

    mlflow.start_run()
    mlflow.sklearn.autolog()

    lines = [
        f"Data file: {args.dataset_path}",
        f"Estimators: {args.n_estimators}",
        f"Voting: {args.voting}",
    ]

    print("Parametros: ...")

    # imprimir parámetros:
    for line in lines:
        print(line)

    # log en mlflow
    mlflow.log_param('Data file', str(args.dataset_path))
    mlflow.log_param('Estimators', str(args.n_estimators))
    mlflow.log_param('Voting', str(args.voting))

    # leer dataset
    data = pd.read_csv(args.dataset_path)

    # separar el ds
    X = data.iloc[:, 1:-1] # evitar la columna de indices y la columna target
    y = data.iloc[:, -1] # target: Potability

    # separar el ds en train/tesst
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # entrenar modelo
    #dt = DecisionTreeClassifier(criterion=args.criterion, min_samples_split=args.min_samples_split)
    clf1 = LogisticRegression(multi_class='multinomial', random_state=1)
    clf2 = RandomForestClassifier(n_estimators=args.n_estimators, random_state=1)
    clf3 = GaussianNB()
    dt = VotingClassifier(
        estimators=[('lr', clf1), ('rf', clf2), ('gnb', clf3)], 
        voting=args.voting)

    dt.fit(X_train,y_train)

    # evaluar el modelo
    y_pred = dt.predict(X_test)
    f1 = f1_score(y_test, y_pred, average='weighted')
    print(confusion_matrix(y_test, y_pred))

    # imprimir metrica en mlflow
    mlflow.log_metric('F1 Score', float(f1))

    registered_model_name="sklearn-ensemble-voting-classifier"

    print("Registrar el modelo via MLFlow")
    mlflow.sklearn.log_model(
        sk_model=dt,
        registered_model_name=registered_model_name,
        artifact_path=registered_model_name
    )

    print("Guardar el modelo via MLFlow")
    mlflow.sklearn.save_model(
        sk_model=dt,
        path=os.path.join(registered_model_name, "trained_model"),
    )

    mlflow.end_run()

if __name__ == '__main__':
    main()