import os
import joblib
import numpy as np
import warnings
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV, cross_val_score

warnings.filterwarnings("ignore", message=".*Inferred schema contains integer column.*")

def train_and_save_models(X_train, y_train, model_dir="models"):
    """
    Trains multiple models with optional hyperparameter tuning,
    evaluates using cross-validation, saves all models as .pkl,
    and saves the best model as best_model.pkl.

    Args:
        X_train: Features for training
        y_train: Target for training
        model_dir: Directory to save .pkl files
    """
    os.makedirs(model_dir, exist_ok=True)

    models = {
        "linear_regression": LinearRegression(),
        "ridge": Ridge(),
        "random_forest": RandomForestRegressor(),
        "gradient_boosting": GradientBoostingRegressor(),
        "decision_tree": DecisionTreeRegressor(),
        "svm": SVR(),
        "knn": KNeighborsRegressor(),
        "xgboost": XGBRegressor(verbosity=0)
    }

    param_grids = {
        "ridge": {"alpha": [0.1, 1.0, 10.0]},
        "random_forest": {"n_estimators": [50, 100], "max_depth": [None, 10]},
        "gradient_boosting": {"n_estimators": [50, 100], "learning_rate": [0.05, 0.1]},
        "decision_tree": {"max_depth": [None, 10, 20]},
        "svm": {"C": [1, 10], "kernel": ["rbf", "linear"]},
        "knn": {"n_neighbors": [3, 5, 7]},
        "xgboost": {"n_estimators": [50, 100], "learning_rate": [0.05, 0.1]}
    }

    best_model = None
    best_model_name = None
    best_score = -np.inf

    for name, model in models.items():
        print(f"🚀 Training: {name}")

        # Hyperparameter tuning if grid is defined
        if name in param_grids:
            grid = GridSearchCV(model, param_grids[name], cv=3, scoring="r2", n_jobs=-1)
            grid.fit(X_train, y_train)
            best_estimator = grid.best_estimator_
        else:
            model.fit(X_train, y_train)
            best_estimator = model

        # Evaluate with cross-validation
        scores = cross_val_score(best_estimator, X_train, y_train, cv=3, scoring="r2")
        avg_score = np.mean(scores)
        print(f"📊 {name} - R² CV Score: {avg_score:.4f}")

        # Save model
        model_path = os.path.join(model_dir, f"{name}.pkl")
        joblib.dump(best_estimator, model_path)
        print(f"✅ Saved: {model_path}")

        # Track best model
        if avg_score > best_score:
            best_score = avg_score
            best_model = best_estimator
            best_model_name = name

    # Save best model separately
    if best_model:
        best_model_path = os.path.join(model_dir, "best_model.pkl")
        joblib.dump(best_model, best_model_path)
        print(f"\n🏆 Best Model: {best_model_name} | R² = {best_score:.4f}")
        print(f"📦 Saved best model as → {best_model_path}")
