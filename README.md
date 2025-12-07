1️⃣ Member 1 — Data & Augmentation

Organizes the dataset and cleans images.

Applies and documents data augmentation (≥30%).

Balances classes and prepares final train/validation splits.

2️⃣ Member 2 — Feature Extraction

Chooses and implements the image-to-vector feature extraction method.

Tests multiple descriptors (e.g., HOG, LBP, color histograms).

Produces final feature vector dataset for both models.

3️⃣ Member 3 — SVM Model

Implements and trains the SVM classifier.

Tunes kernels/hyperparameters.

Adds rejection mechanism for Unknown class.

Evaluates performance.

4️⃣ Member 4 — k-NN Model

Implements and trains k-NN with optimized k and weighting.

Adds Unknown class rejection.

Evaluates performance and compares with SVM.

5️⃣ Member 5 — Deployment

Integrates the best-performing model into a real-time camera application.

Handles frame capture, preprocessing, feature extraction, classification, and display.

Optimizes speed and prepares the final demo.