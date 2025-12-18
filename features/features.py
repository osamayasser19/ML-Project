import os
import glob
import cv2
import numpy as np
import joblib
from skimage.feature import hog, local_binary_pattern
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing import image


cnn_extractor = ResNet50(weights='imagenet', include_top=False, pooling='avg')

def get_cnn_features(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return cnn_extractor.predict(x, verbose=0)[0]


def extract_hog_features(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return hog(gray, orientations=9, pixels_per_cell=(8,8),
               cells_per_block=(2,2), block_norm='L2-Hys', transform_sqrt=True)

def color_histogram(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0,1,2], None, [8,8,8], [0,180,0,256,0,256])
    hist = cv2.normalize(hist, hist).flatten()
    return hist

def lbp_features(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    lbp = local_binary_pattern(gray, P=8, R=1, method='uniform')
    hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, 10), range=(0, 9))
    hist = hist.astype("float") / (hist.sum() + 1e-7)
    return hist

def process_all_images(data_dir):
    features, labels = [], []
    class_folders = sorted(os.listdir(data_dir))
    
    for class_label, class_name in enumerate(class_folders):
        folder_path = os.path.join(data_dir, class_name)
        if not os.path.isdir(folder_path):
            continue
        print(f"Processing class: {class_name}")
        
        for img_file in glob.glob(os.path.join(folder_path, '*.jpg')):
            img = cv2.imread(img_file)
            if img is None:
                continue
            img_resized = cv2.resize(img, (128,128))
            
            hog_feat = extract_hog_features(img_resized)
            color_feat = color_histogram(img_resized)
            lbp_feat = lbp_features(img_resized)
            
            cnn_feat = get_cnn_features(img_file)
            
            combined_feat = np.hstack([cnn_feat])
            features.append(combined_feat)
            labels.append(class_label)
    
    X = np.array(features)
    y = np.array(labels)
    
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    joblib.dump(scaler, 'scaler.save')
    
    np.save('X_features.npy', X)
    np.save('y_labels.npy', y)
    print("Features saved as X_features.npy and y_labels.npy.")


if __name__ == "__main__":
    process_all_images(r'I:\4th year\first term\ML\Project\ML-Project\dataset')
