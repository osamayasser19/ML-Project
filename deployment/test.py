import os
import cv2
import joblib
import numpy as np
from skimage.feature import hog, local_binary_pattern
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing import image

def predict(dataFilePath, bestModelPath=None):


    # ---------- Load model and scaler ----------
    if bestModelPath is None:
        bestModelPath = 'svm_waste_model.pkl'

    SVM_model = joblib.load(bestModelPath)
    scaler = joblib.load('scaler.pkl')  
    
    dataset_dir = r"I:\\4th year\\first term\\ML\\Project\\ML-Project\\dataset"
    class_names = sorted([d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d))])

    cnn_extractor = ResNet50(weights='imagenet', include_top=False, pooling='avg')

    # ---------- Feature extraction functions ----------
    def get_cnn_features(frame):
        img = cv2.resize(frame, (224, 224))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        return cnn_extractor.predict(x, verbose=0)[0]

    def extract_hog_features(frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return hog(gray, orientations=9, pixels_per_cell=(8,8),
                   cells_per_block=(2,2), block_norm='L2-Hys', transform_sqrt=True)

    def lbp_features(frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        lbp = local_binary_pattern(gray, P=8, R=1, method='uniform')
        hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0,10), range=(0,9))
        hist = hist.astype("float")
        hist /= (hist.sum() + 1e-7)
        return hist

    def color_histogram(frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv],[0,1,2],None,[8,8,8],[0,180,0,256,0,256])
        hist = cv2.normalize(hist,hist).flatten()
        return hist

    def get_single_frame_features(frame):
        img_small = cv2.resize(frame, (128,128))
        hog_feat = extract_hog_features(img_small)
        lbp_feat = lbp_features(img_small)
        color_feat = color_histogram(img_small)
        # handcrafted = np.hstack([hog_feat, lbp_feat, color_feat])  # optional

        cnn_feat = get_cnn_features(frame)
        combined = np.hstack([cnn_feat]) 
        scaled = scaler.transform(combined.reshape(1,-1))
        return scaled

    def predict_with_unknown_svm(model, sample, threshold=0.3):
        probs = model.predict_proba(sample)[0]
        max_prob = np.max(probs)
        if max_prob < threshold:
            return "Unknown"
        else:
            return np.argmax(probs)

    # ---------- Process images ----------
    predictions = []
    for file_name in os.listdir(dataFilePath):
        file_path = os.path.join(dataFilePath, file_name)
        if not file_name.lower().endswith(('.jpg','.jpeg','.png')):
            continue
        
        frame = cv2.imread(file_path)
        if frame is None:
            print(f"Skipping invalid image: {file_name}")
            predictions.append("Invalid")
            continue

        features = get_single_frame_features(frame)
        pred_idx = predict_with_unknown_svm(SVM_model, features)
        if pred_idx != "Unknown":
            pred_class = class_names[int(pred_idx)]
        else:
            pred_class = pred_idx

        print(f"{file_name} → Predicted: {pred_class}")

        predictions.append(pred_class)

    return predictions

if __name__ == "__main__":
    test_folder_path = r'I:\4th year\first term\ML\Project\ML-Project\test_images'
    predict(test_folder_path)  
