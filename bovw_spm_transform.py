import cv2
import numpy as np
import os
from glob import glob
import pickle
import time
from tqdm import tqdm
from PIL import Image, ImageEnhance
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import normalize
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, accuracy_score


def save_pickle(obj, path):
    with open(path, 'wb') as f:
        pickle.dump(obj, f)


def load_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def rootsift(descriptors):
    eps = 1e-7
    desc_l1 = descriptors / (np.linalg.norm(descriptors, ord=1, axis=1, keepdims=True) + eps)
    return np.sqrt(desc_l1)


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


class SpatialPyramidBOW:
    def __init__(self, vocab_size=1000, levels=(0, 1, 2), use_rootsift=True, batch_size=10000):
        self.vocab_size = vocab_size
        self.levels = levels
        self.use_rootsift = use_rootsift
        self.sift = cv2.SIFT_create()
        self.kmeans = MiniBatchKMeans(n_clusters=vocab_size, batch_size=batch_size)
        self.idf = None

    def fit(self, image_paths):
        descriptor_list = []
        for path in tqdm(image_paths, desc="Extracting descriptors"):
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            kps, desc = self.sift.detectAndCompute(img, None)
            if desc is not None:
                if self.use_rootsift:
                    desc = rootsift(desc)
                descriptor_list.append(desc)
        all_desc = np.vstack(descriptor_list).astype(np.float64)
        self.kmeans.fit(all_desc)

    def _compute_spm_hist(self, img, keypoints, descriptors):
        h, w = img.shape
        pyramid_hist = []
        for level in self.levels:
            num_cells = 2 ** level
            cell_w = w / num_cells
            cell_h = h / num_cells
            cell_hists = [np.zeros(self.vocab_size, dtype=np.float32)
                          for _ in range(num_cells * num_cells)]
            for kp, d in zip(keypoints, descriptors):
                x, y = kp.pt
                i = int(x // cell_w)
                j = int(y // cell_h)
                idx = j * num_cells + i
                word = self.kmeans.predict(d.reshape(1, -1))[0]
                cell_hists[idx][word] += 1
            for hist in cell_hists:
                if hist.sum() > 0:
                    hist /= hist.sum()
                pyramid_hist.append(hist)
        return np.concatenate(pyramid_hist)

    def transform(self, image_paths, transform_type=None):
        feats = []
        total_dim = self.vocab_size * sum((2 ** l) ** 2 for l in self.levels)
        for path in tqdm(image_paths, desc="Transforming images"):
            if transform_type:
                img = apply_transformation(path, transform_type)
            else:
                img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            kps, desc = self.sift.detectAndCompute(img, None)
            if desc is None or len(desc) == 0:
                feats.append(np.zeros(total_dim, dtype=np.float32))
            else:
                if self.use_rootsift:
                    desc = rootsift(desc)
                desc = desc.astype(np.float64)
                feats.append(self._compute_spm_hist(img, kps, desc))
        return np.array(feats)

    def tfidf_weight(self, histograms):
        df = np.sum(histograms > 0, axis=0)
        n_docs = histograms.shape[0]
        self.idf = np.log((n_docs + 1) / (df + 1)) + 1
        tfidf = histograms * self.idf
        return normalize(tfidf, norm='l2')

    def apply_idf(self, histograms):
        tfidf = histograms * self.idf
        return normalize(tfidf, norm='l2')


if __name__ == "__main__":
    DATASET_DIR = 'caltech-101'
    VOCAB_SIZE = 1000
    LEVELS = (0, 1, 2)
    TEST_SIZE = 0.2
    RANDOM_STATE = 42

    categories = sorted([d for d in os.listdir(DATASET_DIR) if os.path.isdir(os.path.join(DATASET_DIR, d))])
    image_paths, labels = [], []
    for cat in categories:
        paths = glob(os.path.join(DATASET_DIR, cat, '*.jpg'))
        image_paths.extend(paths)
        labels.extend([cat] * len(paths))

    X_train, X_test, y_train, y_test = train_test_split(
        image_paths, labels, test_size=TEST_SIZE,
        stratify=labels, random_state=RANDOM_STATE
    )

    spm = SpatialPyramidBOW(vocab_size=VOCAB_SIZE, levels=LEVELS, use_rootsift=True)

    print("Fitting vocabulary...")
    spm.fit(X_train)

    print("Transforming training images to histograms...")
    X_train_hist = spm.transform(X_train)

    print("Transforming test images to histograms...")
    X_test_hist = spm.transform(X_test)

    print("Applying TF-IDF...")
    X_train_tfidf = spm.tfidf_weight(X_train_hist)
    X_test_tfidf = spm.apply_idf(X_test_hist)

    print("Training SVM with cross-validation...")
    param_grid = {'C': [0.01, 0.1, 1, 10]}
    svc = GridSearchCV(LinearSVC(max_iter=10000), param_grid, cv=5, n_jobs=-1)
    svc.fit(X_train_tfidf, y_train)
    print("Best parameters:", svc.best_params_)

    y_pred = svc.predict(X_test_tfidf)
    print("\nOriginal Test Accuracy:")
    print(classification_report(y_test, y_pred))
    print("Accuracy:", accuracy_score(y_test, y_pred))

    # Evaluate on Transformed Images
    transform_types = ['rotate', 'hflip', 'brightness', 'gaussian_noise']
    for t_type in transform_types:
        print(f"\nEvaluating with transformation: {t_type}")
        X_test_trans = spm.transform(X_test, transform_type=t_type)
        X_test_trans_tfidf = spm.apply_idf(X_test_trans)
        y_pred_trans = svc.predict(X_test_trans_tfidf)
        print(classification_report(y_test, y_pred_trans))
        print(f"Accuracy ({t_type}):", accuracy_score(y_test, y_pred_trans))

    # Save model
    save_pickle({'kmeans': spm.kmeans, 'idf': spm.idf, 'svm': svc}, 'bow_spm_svm_model.pkl')

# OG Accuracy: 0.4417714598141061
# Accuracy (rotate): 0.2498633132859486
# Accuracy (hflip): 0.37561509021323125
# Accuracy (brightness): 0.4264625478403499
# Accuracy (gaussian_noise): 0.4324767632586113