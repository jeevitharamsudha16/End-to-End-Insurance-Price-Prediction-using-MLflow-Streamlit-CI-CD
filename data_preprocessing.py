import pandas as pd
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

def preprocess_data(df):
    """
    Preprocesses the insurance dataset:
    - Drops missing values and duplicates
    - Encodes categorical variables
    - Splits into train/test
    - Scales continuous features ('age', 'bmi') using MinMaxScaler
    - Saves the scaler to models/scalers/minmax_scaler.pkl

    Returns:
    - X_train, X_test, y_train, y_test, scaler
    """
    df = df.copy()

    # 1️⃣ Drop missing and duplicate entries
    df.dropna(inplace=True)
    df.drop_duplicates(inplace=True)

    # 2️⃣ Encode 'smoker' as 0/1
    le = LabelEncoder()
    df['smoker'] = le.fit_transform(df['smoker'])

    # 3️⃣ One-hot encode 'sex' (drop_first to avoid multicollinearity)
    df = pd.get_dummies(df, columns=['sex'], drop_first=True, dtype=int)

    # 4️⃣ Encode 'region' with a manual map
    region_map = {'southwest': 0, 'northwest': 1, 'northeast': 2, 'southeast': 3}
    df['region'] = df['region'].map(region_map)

    # 5️⃣ Split features and target
    X = df.drop('charges', axis=1)
    y = df['charges']

    # 6️⃣ Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 7️⃣ Scale only continuous numerical columns: 'age', 'bmi'
    scaler = MinMaxScaler()
    X_train[['age', 'bmi']] = scaler.fit_transform(X_train[['age', 'bmi']]).astype(float)
    X_test[['age', 'bmi']] = scaler.transform(X_test[['age', 'bmi']]).astype(float)

    # 8️⃣ Save the scaler inside a dedicated scalers folder
    scaler_dir = os.path.join("scalers")
    os.makedirs(scaler_dir, exist_ok=True)
    scaler_path = os.path.join(scaler_dir, "minmax_scaler.pkl")
    joblib.dump(scaler, scaler_path)
    print(f"📎 Scaler saved to {scaler_path}")

    print("✅ Data preprocessing complete.")
    return X_train, X_test, y_train, y_test, scaler