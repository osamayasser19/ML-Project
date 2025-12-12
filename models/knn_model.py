import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import joblib


data = np.load('../dataset.npz')

X = data['features']
y = data['labels']

print("Total samples:", X.shape[0])
print("Feature vector length:", X.shape[1])
print("Classes:", np.unique(y))


class_counts = {cls: np.sum(y == cls) for cls in np.unique(y)}
print("Samples per class:", class_counts)


X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Training samples:", X_train.shape[0])
print("Testing samples:", X_test.shape[0])


k_values = [1, 3, 5, 7, 9]
weight_options = ['uniform', 'distance']

best_acc = 0
optimal_k = None
best_weight = None

print("\nTuning Results:")
for k in k_values:
    for w in weight_options:
        temp_model = KNeighborsClassifier(
            n_neighbors=k,
            weights=w,
            algorithm='brute'
        )
        temp_model.fit(X_train, y_train)
        temp_pred = temp_model.predict(X_test)
        temp_acc = accuracy_score(y_test, temp_pred)

        print(f"k={k}, weights={w}, accuracy={temp_acc}")

        if temp_acc > best_acc:
            best_acc = temp_acc
            optimal_k = k
            best_weight = w

print("\nBest configuration found:")
print("Best k:", optimal_k)
print("Best weights:", best_weight)
print("Best accuracy:", best_acc)


final_knn = KNeighborsClassifier(
    n_neighbors=optimal_k,
    weights=best_weight,
    algorithm='brute'
)

final_knn.fit(X_train, y_train)

joblib.dump(final_knn, 'knn_best_model.sav')
print("\nOptimized KNN model saved as knn_best_model.sav")


def predict_with_unknown(model, sample, threshold):
    """
    Rejects prediction if average distance to neighbors is too large.
    """
    distances, indices = model.kneighbors(sample)
    avg_distance = np.mean(distances)

    if avg_distance > threshold:
        return "unknown"   
    else:
        return model.predict(sample)[0]


unknown_threshold = 0.9


test_sample = X_test[0].reshape(1, -1)
normal_pred = final_knn.predict(test_sample)[0]
reject_pred = predict_with_unknown(final_knn, test_sample, unknown_threshold)

print("\nNormal prediction:", normal_pred)
print("Prediction with unknown handling:", reject_pred)
print("Actual label:", y_test[0])
