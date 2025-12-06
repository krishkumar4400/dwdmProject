import os
import sys
import numpy as np
import pandas as pd
import dill
from src.exception import CustomException
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV


def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)


def evaluate_models(X_train, y_train, X_test, y_test, models, params):
    try:
        report = {}
        best_model = None
        best_model_name = None
        best_score = -999

        for model_name, model in models.items():

            # Hyperparameter tuning
            param_grid = params[model_name]
            gs = GridSearchCV(model, param_grid, cv=3)
            gs.fit(X_train, y_train)

            # best estimator
            best_estimator = gs.best_estimator_
            best_estimator.fit(X_train, y_train)

            # prediction
            y_pred = best_estimator.predict(X_test)
            score = r2_score(y_test, y_pred)

            report[model_name] = score

            # choose best model
            if score > best_score:
                best_score = score
                best_model = best_estimator
                best_model_name = model_name

        return best_model_name, best_model, report

    except Exception as e:
        raise CustomException(e, sys)
   
def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return dill.load(file_obj)
    except Exception as e:
        raise CustomException(e, sys)
        
