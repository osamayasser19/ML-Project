# import numpy as np
# import joblib
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.preprocessing import StandardScaler
# from sklearn.model_selection import train_test_split
#
# X = np.load('../features/X_features.npy')
# y = np.load('../features/y_labels.npy')
#
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)
# scaler = StandardScaler()
# X_train_scaled = scaler.fit_transform(X_train)
#
# knn = KNeighborsClassifier(n_neighbors=9)
# knn.fit(X_train_scaled, y_train)
#
# joblib.dump(knn, 'knn_waste_model.pkl')
# print("k-NN Model Saved.")
import numpy as np
import joblib
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

X = np.load(r'I:\4th year\first term\ML\Project\ML-Project\X_features.npy')
y = np.load(r'I:\4th year\first term\ML\Project\ML-Project\y_labels.npy')

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


knn_model = KNeighborsClassifier(n_neighbors=9, weights='distance')
print("Starting k-NN training...")
knn_model.fit(X_train_scaled, y_train)
print("k-NN training complete.")

y_pred = knn_model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)

print(f"\n--- k-NN Model Results ---")
print(f"Accuracy: {accuracy * 100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

joblib.dump(knn_model, 'knn_waste_model.pkl')
print("\nk-NN Model saved successfully as 'knn_waste_model.pkl'.")