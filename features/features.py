import cv2
import os
import numpy as np
from skimage.feature import hog, local_binary_pattern
from sklearn.preprocessing import StandardScaler
import joblib

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


featurs =[]
class_labels = []
class_folders = os.listdir('./dataset')
print("Extracting features from images...",class_folders)


for class_label in class_folders:
    class_folder_path = os.path.join('./dataset', class_label)
    print(f"Processing class: {class_label}")
    for idx, image_name in enumerate(os.listdir(class_folder_path)):
        image_path = os.path.join(class_folder_path, image_name)
        image = cv2.imread(image_path)
        if image is not None:
            image = cv2.resize(image, (128, 128))
            hog_feat = extract_hog_features(image)
            color_hist_feat = color_histogram(image)
            lbp_feat = lbp_features(image)
            combined_features = np.hstack([hog_feat, color_hist_feat, lbp_feat])
            featurs.append(combined_features)
            class_labels.append(class_label)

featurs = np.array(featurs)
class_labels = np.array(class_labels)
scaler = StandardScaler()
featurs = scaler.fit_transform(featurs)
joblib.dump(scaler, 'scaler.save')
np.savez_compressed('dataset.npz', features=featurs, labels=class_labels)
print("Feature extraction completed. Features and labels saved.")

