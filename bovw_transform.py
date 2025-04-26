import cv2
import numpy as np
import os
from sklearn.cluster import MiniBatchKMeans
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import LinearSVC
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from glob import glob
import pickle
from pathlib import Path
from PIL import Image, ImageEnhance

# ==========================
# Transformation Helpers
# ==========================
def apply_transformation(image_path, transform_type='rotate'):
    image = Image.open(image_path).convert('RGB')
    if transform_type == 'rotate':
        image = image.rotate(30)
    elif transform_type == 'hflip':
        image = image.transpose(Image.FLIP_LEFT_RIGHT)
    elif transform_type == 'brightness':
        image = ImageEnhance.Brightness(image).enhance(1.5)
    elif transform_type == 'gaussian_noise':
        img_np = np.array(image).astype(np.float32)
        noise = np.random.normal(0, 25, img_np.shape).astype(np.float32)
        noisy = np.clip(img_np + noise, 0, 255).astype(np.uint8)
        image = Image.fromarray(noisy)
    return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)

# ==========================
# BoVW + SIFT Pipeline
# ==========================
def extract_sift_descriptors(image_path, sift=None, transformed=False, transform_type=None):
    if sift is None:
        sift = cv2.SIFT_create()
    if transformed and transform_type:
        img = apply_transformation(image_path, transform_type)
    else:
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    keypoints, descriptors = sift.detectAndCompute(img, None)
    return descriptors

def build_vocabulary(descriptor_list, vocab_size=500, max_descriptors=100000):
    all_descriptors = np.vstack(descriptor_list)
    if len(all_descriptors) > max_descriptors:
        np.random.shuffle(all_descriptors)
        all_descriptors = all_descriptors[:max_descriptors]
    kmeans = MiniBatchKMeans(n_clusters=vocab_size, batch_size=10000)
    kmeans.fit(all_descriptors)
    return kmeans

def compute_bovw_histogram(descriptors, kmeans):
    if descriptors is None or len(descriptors) == 0:
        return np.zeros(kmeans.n_clusters)
    words = kmeans.predict(descriptors)
    histogram, _ = np.histogram(words, bins=np.arange(kmeans.n_clusters + 1))
    return histogram.astype("float32") / (histogram.sum() + 1e-6)

def save_pickle(obj, path):
    with open(path, 'wb') as f:
        pickle.dump(obj, f)

def load_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

# ==========================
# Main Script
# ==========================
DATASET_DIR = 'caltech-101'
VOCAB_SIZE = 500
SELECTED_CLASSES = []

if __name__ == "__main__":
    all_image_paths = glob(os.path.join(DATASET_DIR, '*/*.jpg'))
    if len(SELECTED_CLASSES) > 0:
        image_paths = [p for p in all_image_paths if Path(p).parent.name in SELECTED_CLASSES]
    else:
        image_paths = all_image_paths

    descriptor_list = []
    valid_image_paths = []
    sift = cv2.SIFT_create()

    print("📸 Extracting SIFT features...")
    for img_path in image_paths:
        desc = extract_sift_descriptors(img_path, sift)
        if desc is not None:
            descriptor_list.append(desc)
            valid_image_paths.append(img_path)
        else:
            print(f"⚠️ No descriptors for: {img_path}")

    print("🔨 Building visual vocabulary...")
    kmeans = build_vocabulary(descriptor_list, vocab_size=VOCAB_SIZE)
    save_pickle(kmeans, 'kmeans_vocab.pkl')

    image_histograms = []
    for img_path in valid_image_paths:
        desc = extract_sift_descriptors(img_path, sift)
        hist = compute_bovw_histogram(desc, kmeans)
        image_histograms.append(hist)

    save_pickle((valid_image_paths, image_histograms), 'db_histograms.pkl')

    labels = [os.path.basename(os.path.dirname(p)) for p in valid_image_paths]
    le = LabelEncoder()
    y = le.fit_transform(labels)
    X = np.array(image_histograms)

    X_train, X_test, y_train, y_test, path_train, path_test = train_test_split(
        X, y, valid_image_paths, test_size=0.2, random_state=42
    )

    print("\n🧠 Training SVM with GridSearchCV...")
    svc = GridSearchCV(
        LinearSVC(max_iter=20000, class_weight='balanced'),
        {'C': [0.01, 0.1, 1, 10]},
        cv=5,
        n_jobs=-1
    )
    svc.fit(X_train, y_train)
    print("✅ Best C:", svc.best_params_)

    print("\n🧪 Evaluating on original test images...")
    y_pred = svc.predict(X_test)
    print(classification_report(y_test, y_pred, target_names=le.classes_))
    print("🎯 Accuracy (Original):", accuracy_score(y_test, y_pred))

    # ==========================
    # Evaluate on Transformed Images
    # ==========================
    transform_types = ['rotate', 'hflip', 'brightness', 'gaussian_noise']
    print("\n🧪 Evaluating on transformed test images...")

    for t_type in transform_types:
        print(f"\n🔄 Transformation: {t_type}")
        transformed_histograms = []
        for path in path_test:
            desc = extract_sift_descriptors(path, sift, transformed=True, transform_type=t_type)
            hist = compute_bovw_histogram(desc, kmeans)
            transformed_histograms.append(hist)
        X_transformed = np.array(transformed_histograms)
        y_pred_transformed = svc.predict(X_transformed)
        print(classification_report(y_test, y_pred_transformed, target_names=le.classes_))
        print(f"🎯 Accuracy ({t_type}):", accuracy_score(y_test, y_pred_transformed))

    # Save model
    save_pickle({'kmeans': kmeans, 'svm': svc, 'label_encoder': le}, 'bovw_model.pkl')

# 🎯 Accuracy (gaussian_noise): 0.37780207763805357
# 🎯 Accuracy (brightness): 0.3466375068343357
# 🎯 Accuracy (hflip): 0.3870967741935484
# 🎯 Accuracy (rotate): 0.33406232914160744
# 🎯 Accuracy (Original): 0.3969382176052488