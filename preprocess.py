import pandas as pd
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder


def load_data(input_path):
    try:
        df = pd.read_csv(input_path)

        if 'label' not in df.columns:
            raise Exception("No label column")

        print("Custom dataset loaded")
        return df 

    except:
        print("Loading NSL-KDD format dataset...")

        columns = [
            "duration","protocol_type","service","flag","src_bytes","dst_bytes","land",
            "wrong_fragment","urgent","hot","num_failed_logins","logged_in","num_compromised",
            "root_shell","su_attempted","num_root","num_file_creations","num_shells",
            "num_access_files","num_outbound_cmds","is_host_login","is_guest_login",
            "count","srv_count","serror_rate","srv_serror_rate","rerror_rate",
            "srv_rerror_rate","same_srv_rate","diff_srv_rate","srv_diff_host_rate",
            "dst_host_count","dst_host_srv_count","dst_host_same_srv_rate",
            "dst_host_diff_srv_rate","dst_host_same_src_port_rate",
            "dst_host_srv_diff_host_rate","dst_host_serror_rate",
            "dst_host_srv_serror_rate","dst_host_rerror_rate",
            "dst_host_srv_rerror_rate","label"
        ]

        df = pd.read_csv(input_path, names=columns)

        
        if df['label'].dtype == 'object':
            df['label'] = df['label'].apply(
                lambda x: 0 if str(x).strip().startswith('normal') else 1
            )
        else:
            print("Labels already numeric")

        return df 


def preprocess_data(input_path, output_dir="data/processed"):
    df = load_data(input_path)

    # Drop missing values
    df = df.dropna()

    # Encode categorical columns
    for col in df.select_dtypes(include=['object']).columns:
        if col != 'label':
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])

    # Features & target
    X = df.drop('label', axis=1)
    y = df['label']

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )

    # Save outputs
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs("models", exist_ok=True)

    pd.DataFrame(X_train).to_csv(f"{output_dir}/X_train.csv", index=False)
    pd.DataFrame(X_test).to_csv(f"{output_dir}/X_test.csv", index=False)
    pd.DataFrame(y_train).to_csv(f"{output_dir}/y_train.csv", index=False)
    pd.DataFrame(y_test).to_csv(f"{output_dir}/y_test.csv", index=False)

    joblib.dump(scaler, "models/scaler.pkl")

    print("Preprocessing complete!")
    print(f"Train shape: {X_train.shape}")
    print(f"Test shape: {X_test.shape}")


if __name__ == "__main__":
    preprocess_data("data/raw/dataset.csv.txt")