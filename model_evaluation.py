import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings("ignore")

def evaluate_models_without_mlflow(X_test, y_test, model_dir="models"):
    metrics_list = []
    best_r2 = -np.inf
    best_model_name = None
    best_model = None

    for model_file in os.listdir(model_dir):
        if not model_file.endswith(".pkl") or "scaler" in model_file.lower():
            continue

        model_name = model_file.replace(".pkl", "")
        model_path = os.path.join(model_dir, model_file)
        model = joblib.load(model_path)

        y_pred = model.predict(X_test)

        # Evaluation metrics
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)

        # Save scatter plot
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.scatter(y_test, y_pred, alpha=0.6)
        ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
        ax.set_xlabel("Actual Charges")
        ax.set_ylabel("Predicted Charges")
        ax.set_title(f"Actual vs Predicted ({model_name})")
        plot_path = f"{model_dir}/{model_name}_pred_plot.png"
        fig.savefig(plot_path, dpi=300)
        plt.close(fig)

        # Track metrics
        metrics_list.append({
            "model": model_name,
            "R2": r2,
            "MAE": mae,
            "RMSE": rmse
        })

        if r2 > best_r2:
            best_r2 = r2
            best_model_name = model_name
            best_model = model

    # Save model comparison to CSV
    df_metrics = pd.DataFrame(metrics_list)
    df_metrics.sort_values(by="R2", ascending=False, inplace=True)
    df_metrics.to_csv(f"{model_dir}/model_comparison_metrics.csv", index=False)
    print("📄 Saved model comparison to models/model_comparison_metrics.csv")

    # Save the best model as 'best_model.pkl'
    if best_model:
        joblib.dump(best_model, os.path.join(model_dir, "best_model.pkl"))
        print(f"🏆 Best model: {best_model_name} (R²: {best_r2:.4f}) saved as 'best_model.pkl'")