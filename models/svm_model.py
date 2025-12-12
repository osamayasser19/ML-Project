import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
from sklearn.impute import SimpleImputer

print("Loading dataset...")
data = np.load('../features/dataset.npz')
X = data['features']
y = data['labels']


imputer = SimpleImputer(strategy='mean')
X = imputer.fit_transform(X)

joblib.dump(imputer, 'imputer.save')


encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)
joblib.dump(encoder, 'label_encoder.save')

X_train, X_val, y_train, y_val = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

print("Training SVM classifier...")
svm_clf = SVC(
    kernel='rbf',
    C=10,
    gamma='scale',
    probability=True,
    random_state=42
)

svm_clf.fit(X_train, y_train)
joblib.dump(svm_clf, 'svm_model.save')

y_pred = svm_clf.predict(X_val)
acc = accuracy_score(y_val, y_pred)

print("\nValidation Accuracy:", acc)
print("\nClassification Report:\n")
print(classification_report(y_val, y_pred, target_names=encoder.classes_))

def predict_with_rejection(model, encoder, image_features, threshold=0.6):
    probs = model.predict_proba([image_features])[0]
    max_prob = np.max(probs)
    if max_prob < threshold:
        return "Unknown"
    else:
        return encoder.inverse_transform([np.argmax(probs)])[0]

example_feat = X_val[0]
predicted_class = predict_with_rejection(svm_clf, encoder, example_feat)
print("\nExample prediction with rejection:", predicted_class)
