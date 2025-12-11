# model_training.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
import joblib
import seaborn as sns
import matplotlib.pyplot as plt

# 1️⃣ Load dataset
print("📂 Loading creditcard.csv ...")
data = pd.read_csv("creditcard1.csv")
print("✅ Dataset loaded:", data.shape)

# 2️⃣ Drop Time column (not useful)
if "Time" in data.columns:
    data = data.drop(columns=["Time"])

# 3️⃣ Split features and labels
X = data.drop("Class", axis=1)
y = data["Class"]

# 4️⃣ Scale numeric features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Save the scaler
joblib.dump(scaler, "scaler.pkl")
print("💾 Saved scaler.pkl")

# 5️⃣ Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)
print("✅ Split done:", X_train.shape, X_test.shape)

# 6️⃣ Apply SMOTE to balance the classes
print("⚖️ Applying SMOTE for class balancing...")
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
print("✅ After SMOTE:", np.bincount(y_train_resampled))

# 7️⃣ Train XGBoost model
model = XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="logloss",
    random_state=42,
    n_jobs=-1
)

print("🚀 Training model on balanced data...")
model.fit(X_train_resampled, y_train_resampled)
print("✅ Model training complete")

# 8️⃣ Evaluate model
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

print("\n📊 Classification Report:")
print(classification_report(y_test, y_pred))

roc = roc_auc_score(y_test, y_prob)
print(f"🏆 ROC-AUC Score: {roc:.4f}")

# 9️⃣ Confusion Matrix Visualization
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# 🔟 Save the trained model
joblib.dump(model, "fraud_detection_model.pkl")
print("💾 Saved fraud_detection_model.pkl")

print("\n🎯 Training complete — model and scaler saved successfully.")
