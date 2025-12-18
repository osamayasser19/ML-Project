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
    
    combined_features=np.array(combined_features)
    
    scaled_features = scaler.transform(combined_features.reshape(1, -1))
    
    return scaled_features

def predict_with_unknown(model, sample, threshold=0.9):
   
    distances, indices = model.kneighbors(sample)
    distances_min = np.min(distances)
    distances_max = np.max(distances)
    denominator = distances_max - distances_min

    if denominator == 0:
        scaled_distances = np.zeros_like(distances)
    else:
        scaled_distances = (distances - distances_min) / denominator
        
    avg_distance = np.mean(scaled_distances)
    
    if avg_distance > threshold:
        return "Unknown"   
    else:
        return model.predict(sample)[0]


cap = cv2.VideoCapture(0)

print("Starting Real-Time MSI System... Press 'q' to exit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    features = get_single_frame_features(frame)
    
    features = features.reshape(1, -1)

    prediction = predict_with_unknown(knn_model,features)
    
    label = prediction

    display_text = f"{label}"
    cv2.putText(frame,display_text , (20, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    cv2.imshow('Material Stream Identification System', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()