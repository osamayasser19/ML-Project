import cv2
import numpy as np
import joblib
from skimage.feature import hog, local_binary_pattern

# -------------------------------
# Load Models and Preprocessors
# -------------------------------
scaler = joblib.load('scaler.save')
imputer = joblib.load('imputer.save')
encoder = joblib.load('label_encoder.save')
svm_clf = joblib.load('svm_model.save')

# -------------------------------
# Feature Extraction Functions
# -------------------------------
def extract_hog_features(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hog_features = hog(gray_image, orientations=9, pixels_per_cell=(8, 8),
                       cells_per_block=(2, 2), block_norm='L2-Hys', transform_sqrt=True)
    return hog_features

def color_histogram(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, [8, 8, 8], [0, 180, 0, 256, 0, 256])
    hist = cv2.normalize(hist, hist).flatten()
    return hist

def lbp_features(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    lbp = local_binary_pattern(gray_image, P=8, R=1, method='uniform')
    hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, 10), range=(0, 9))
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-7)
    return hist

def extract_features(image):
    hog_feat = extract_hog_features(image)
    color_feat = color_histogram(image)
    lbp_feat = lbp_features(image)
    combined = np.hstack([hog_feat, color_feat, lbp_feat])
    combined = imputer.transform([combined])
    combined = scaler.transform(combined)
    return combined[0]

# -------------------------------
# Prediction with Rejection
# -------------------------------
def predict_with_rejection(model, encoder, features, threshold=0.6):
    probs = model.predict_proba([features])[0]
    max_prob = np.max(probs)
    if max_prob < threshold:
        return "Unknown"
    else:
        return encoder.inverse_transform([np.argmax(probs)])[0]

# -------------------------------
# Main Deployment - Webcam
# -------------------------------
cap = cv2.VideoCapture(0)  # 0 = default camera

if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

print("Press 'q' to exit...")

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    # Resize to model input
    resized_frame = cv2.resize(frame, (128, 128))

    # Extract features
    features = extract_features(resized_frame)

    # Predict class
    pred_class = predict_with_rejection(svm_clf, encoder, features)

    # Display result
    cv2.putText(frame, f"Prediction: {pred_class}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Real-time Classification", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
