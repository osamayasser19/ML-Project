import numpy as np
import joblib
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

X = np.load(r'I:\4th year\first term\ML\Project\ML-Project\X_features.npy')
y = np.load(r'I:\4th year\first term\ML\Project\ML-Project\y_labels.npy')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

svm_model = SVC(kernel='rbf', C=10, gamma='auto', probability=True, random_state=42)
print("Starting svm training...")
svm_model.fit(X_train_scaled, y_train)

joblib.dump(svm_model, 'svm_waste_model.pkl')
joblib.dump(scaler, 'scaler.pkl')

print(f"SVM Accuracy: {accuracy_score(y_test, svm_model.predict(X_test_scaled))*100:.2f}%")
print(classification_report(y_test, svm_model.predict(X_test_scaled)))