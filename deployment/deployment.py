import cv2
import joblib
import numpy as np
import os
from skimage.feature import hog, local_binary_pattern
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing import image



SVM_model = joblib.load(r'I:\4th year\first term\ML\Project\ML-Project\svm_waste_model.pkl')
scaler = joblib.load(r'I:\4th year\first term\ML\Project\ML-Project\scaler.pkl')

dataset_dir = r'I:\4th year\first term\ML\Project\ML-Project\dataset'
class_folders = sorted([d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d))])
class_names = class_folders

cnn_extractor = ResNet50(weights='imagenet', include_top=False, pooling='avg')

def get_cnn_features(frame):
    img = cv2.resize(frame, (224, 224))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return cnn_extractor.predict(x, verbose=0)[0]

def extract_hog_features(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hog_features = hog(gray_image, orientations=9, pixels_per_cell=(8, 8),
                       cells_per_block=(2, 2),block_norm='L2-Hys',transform_sqrt=True )
    return hog_features

def color_histogram(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0,1,2], None, [8,8,8], [0,180,0,256,0,256])
    hist =cv2.normalize(hist, hist).flatten()
    return hist

def lbp_features(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    lbp = local_binary_pattern(gray_image, P=8, R=1, method='uniform')
    (hist, _) = np.histogram(lbp.ravel(),
                             bins=np.arange(0, 10),
                             range=(0, 9))
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-7)
    return hist

def get_single_frame_features(frame):

    image = cv2.resize(frame, (128, 128))
    
    hog_feat = extract_hog_features(image)
    color_hist_feat = color_histogram(image)
    lbp_feat = lbp_features(image)
    #handcrafted_feat = np.hstack([hog_feat, color_feat, lbp_feat])

    cnn_feat = get_cnn_features(frame)
    combined_features = np.hstack([cnn_feat])
    
    combined_features=np.array(combined_features)
    
    scaled_features = scaler.transform(combined_features.reshape(1, -1))
    
    return scaled_features

def predict_with_unknown_svm(model, sample, threshold=0.6):
    probs = model.predict_proba(sample)[0]
    max_prob = np.max(probs)

    if max_prob < threshold:
        return "Unknown"
    else:
        return np.argmax(probs)


cap = cv2.VideoCapture(0)

print("Starting Real-Time MSI System... Press 'q' to exit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    features = get_single_frame_features(frame)
    
    features = features.reshape(1, -1)

    prediction = predict_with_unknown_svm(SVM_model, features)
    
    try:
        label = class_names[int(prediction)]
    except Exception:
        label = prediction

    display_text = f"{label}"
    cv2.putText(frame,display_text , (20, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    cv2.imshow('Material Stream Identification System', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()