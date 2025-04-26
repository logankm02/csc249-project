import torch
import torchvision.transforms as transforms
from torchvision.models import resnet50, ResNet50_Weights
from PIL import Image, ImageEnhance
import numpy as np
import os
from glob import glob
from tqdm import tqdm
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import pickle
from pathlib import Path

# ==========================
# Transformation Helpers
# ==========================
def get_transformed_image(image, transform_type='rotate'):
    if transform_type == 'rotate':
        return image.rotate(30)  # Rotate by 30 degrees
    elif transform_type == 'hflip':
        return image.transpose(Image.FLIP_LEFT_RIGHT)
    elif transform_type == 'brightness':
        return ImageEnhance.Brightness(image).enhance(1.5)
    elif transform_type == 'gaussian_noise':
        img_np = np.array(image).astype(np.float32)
        noise = np.random.normal(0, 25, img_np.shape).astype(np.float32)
        noisy = np.clip(img_np + noise, 0, 255).astype(np.uint8)
        return Image.fromarray(noisy)
    else:
        return image

# ==========================
# Feature Extractor
# ==========================
class CNNFeatureExtractor:
    def __init__(self):
        self.model = resnet50(weights=ResNet50_Weights.DEFAULT)
        self.model.eval()
        self.model = torch.nn.Sequential(*list(self.model.children())[:-1])
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def extract(self, image_path, transform_type=None):
        image = Image.open(image_path).convert('RGB')
        if transform_type:
            image = get_transformed_image(image, transform_type)
        input_tensor = self.transform(image).unsqueeze(0)
        with torch.no_grad():
            features = self.model(input_tensor).squeeze().numpy()
        return features / np.linalg.norm(features)

# ==========================
# Save & Load Helpers
# ==========================
def save_pickle(obj, path):
    with open(path, 'wb') as f:
        pickle.dump(obj, f)

def load_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

# ==========================
# Main Logic
# ==========================
if __name__ == "__main__":
    extractor = CNNFeatureExtractor()
    DATASET_DIR = 'caltech-101'

    image_paths = glob(os.path.join(DATASET_DIR, '*/*.jpg'))
    feature_list = []
    valid_image_paths = []

    print("📸 Extracting CNN features...")
    for img_path in tqdm(image_paths, desc="Extracting CNN features"):
        try:
            features = extractor.extract(img_path)
            feature_list.append(features)
            valid_image_paths.append(img_path)
        except Exception as e:
            print(f"❌ Error with {img_path}: {e}")

    save_pickle((valid_image_paths, feature_list), 'cnn_features.pkl')

    # Step 2: Labels
    labels = [Path(p).parent.name for p in valid_image_paths]
    le = LabelEncoder()
    y = le.fit_transform(labels)
    X = np.array(feature_list)

    # Step 3: Train/test split
    X_train, X_test, y_train, y_test, path_train, path_test = train_test_split(
        X, y, valid_image_paths, test_size=0.2, random_state=42
    )

    # Step 4: Train SVM
    print("\n🧠 Training SVM with GridSearchCV...")
    svc = GridSearchCV(
        LinearSVC(max_iter=20000, class_weight='balanced'),
        {'C': [0.01, 0.1, 1, 10]},
        cv=5,
        n_jobs=-1
    )
    svc.fit(X_train, y_train)
    print("✅ Best C:", svc.best_params_)

    # Step 5: Evaluate on original test images
    print("\n🧪 Evaluating on original test images...")
    y_pred = svc.predict(X_test)
    print(classification_report(y_test, y_pred, target_names=le.classes_))
    print("🎯 Accuracy (Original):", accuracy_score(y_test, y_pred))

    # Step 6: Evaluate on multiple transformations
    transform_types = ['rotate', 'hflip', 'brightness', 'gaussian_noise']
    print("\n🧪 Evaluating on transformed test images...")

    for t_type in transform_types:
        print(f"\n🔄 Testing with transformation: {t_type}")
        X_test_transformed = []
        for path in tqdm(path_test, desc=f"Transforming ({t_type})"):
            try:
                feat = extractor.extract(path, transform_type=t_type)
                X_test_transformed.append(feat)
            except Exception as e:
                print(f"Error transforming {path}: {e}")

        X_test_transformed = np.array(X_test_transformed)
        y_pred_transformed = svc.predict(X_test_transformed)
        print(classification_report(y_test, y_pred_transformed, target_names=le.classes_))
        print(f"🎯 Accuracy ({t_type}):", accuracy_score(y_test, y_pred_transformed))

    # Step 7: Save model
    save_pickle({'svm': svc, 'label_encoder': le}, 'resnet_svm_model.pkl')

# 🎯 Accuracy (gaussian_noise): 0.8906506287588847
# 🎯 Accuracy (brightness): 0.9338436303991252
# 🎯 Accuracy (hflip): 0.9409513395297977
# 🎯 Accuracy (rotate): 0.878622197922362
# 🎯 Accuracy (Original): 0.9414980863860033


