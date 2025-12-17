import cv2
import joblib
import numpy as np
from skimage.feature import hog, local_binary_pattern


knn_model = joblib.load('./knn_best_model.sav')
scaler = joblib.load('./scaler.save')


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

    combined_features = np.hstack([hog_feat, color_hist_feat, lbp_feat])
    
    
    scaled_features = scaler.transform(combined_features.reshape(1, -1))
    
    return scaled_features
  


class_map = {
    0: "Glass", 1: "Paper", 2: "Cardboard", 
    3: "Plastic", 4: "Metal", 5: "Trash", 6: "Unknown"
}


cap = cv2.VideoCapture(0)

print("Starting Real-Time MSI System... Press 'q' to exit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    features = get_single_frame_features(frame)
    
    features = features.reshape(1, -1)

    prediction = knn_model.predict(features)[0]
    
    print(prediction)
    label = class_map.get(prediction,'Unknown')

      

    display_text = f"{label}"
    cv2.putText(frame,display_text , (20, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    cv2.imshow('Material Stream Identification System', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()