from data_loader import load_data
from data_preprocessing import preprocess_data
from model_training import train_and_save_models
from model_evaluation import evaluate_models_without_mlflow  # updated

def main(): 
    print("📥 Loading dataset...")
    df = load_data()

    print("🧼 Preprocessing data...")
    X_train, X_test, y_train, y_test, scaler = preprocess_data(df)

    print("🤖 Training and saving models...")
    train_and_save_models(X_train, y_train, model_dir="models")
    print("🎯 Model training completed and saved.")

    print("📊 Evaluating models locally...")
    evaluate_models_without_mlflow(X_test, y_test, model_dir="models")
    print("📈 Evaluation completed and best model selected.")

if __name__ == "__main__":
    main()
