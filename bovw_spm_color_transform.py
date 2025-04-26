import cv2
import numpy as np
import os
import pickle
from glob import glob
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

def compute_color_hist(image_bgr, bins=(8,8,8)):
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0,1,2], None, bins, [0,180,0,256,0,256])
    hist = hist.flatten().astype(np.float32)
    return hist / (hist.sum() + 1e-7)

def apply_transformation(image_path, transform_type):
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
    return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

class SoftSPMBOW:
    def __init__(self, vocab_size=2000, levels=(0,1,2), use_rootsift=True, soft_k=3, batch_size=10000):
        self.vocab_size = vocab_size
        self.levels = levels
        self.use_rootsift = use_rootsift
        self.soft_k = soft_k
        self.sift = cv2.SIFT_create()
        self.kmeans = MiniBatchKMeans(n_clusters=vocab_size, batch_size=batch_size, verbose=1)
        self.idf = None

    def fit(self, image_paths):
        descriptors = []
        for p in tqdm(image_paths, desc="Fitting vocabulary"):
            img_gray = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
            _, desc = self.sift.detectAndCompute(img_gray, None)
            if desc is not None:
                if self.use_rootsift:
                    desc = rootsift(desc)
                descriptors.append(desc)
        all_desc = np.vstack(descriptors).astype(np.float64)
        self.kmeans.fit(all_desc)

    def _compute_spm(self, img_bgr):
        img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        h, w = img_gray.shape
        kps, desc = self.sift.detectAndCompute(img_gray, None)
        if desc is None or len(desc) == 0:
            return np.zeros(self.vocab_size * sum((2**l)**2 for l in self.levels), dtype=np.float32)
        if self.use_rootsift:
            desc = rootsift(desc)
        desc = desc.astype(np.float64)
        centers = self.kmeans.cluster_centers_
        cell_hists = [np.zeros(self.vocab_size, dtype=np.float32)
                      for _ in range(sum((2**l)**2 for l in self.levels))]

        for kp, d in zip(kps, desc):
            x, y = kp.pt
            for lvl_idx, level in enumerate(self.levels):
                cells = 2**level
                cell_w, cell_h = w / cells, h / cells
                i, j = int(x // cell_w), int(y // cell_h)
                cell_idx = sum((2**l)**2 for l in self.levels[:lvl_idx]) + j * cells + i
                dists = np.linalg.norm(centers - d, axis=1)
                nearest = np.argsort(dists)[:self.soft_k]
                weights = 1.0 / (dists[nearest] + 1e-7)
                weights /= weights.sum()
                for word, wgt in zip(nearest, weights):
                    cell_hists[cell_idx][word] += wgt

        for hist in cell_hists:
            if hist.sum() > 0:
                hist /= hist.sum()
        return np.concatenate(cell_hists)

    def transform(self, image_paths, transform_type=None):
        feats = []
        for p in tqdm(image_paths, desc=f"Extracting features{' (' + transform_type + ')' if transform_type else ''}"):
            if transform_type:
                img_bgr = apply_transformation(p, transform_type)
            else:
                img_bgr = cv2.imread(p)
            spm = self._compute_spm(img_bgr)
            color = compute_color_hist(img_bgr)
            feats.append(np.concatenate([spm, color]))
        return np.array(feats)

    def tfidf(self, X):
        df = np.sum(X > 0, axis=0)
        n = X.shape[0]
        self.idf = np.log((n + 1) / (df + 1)) + 1
        return normalize(X * self.idf, norm='l2')

    def apply_idf(self, X):
        return normalize(X * self.idf, norm='l2')

if __name__ == '__main__':
    DATA_DIR = 'caltech-101'
    VOCAB_SIZE = 2000
    LEVELS = (0, 1, 2)
    TEST_SIZE = 0.2
    RS = 42

    cats = sorted(d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d)))
    paths, labels = [], []
    for c in cats:
        ps = glob(os.path.join(DATA_DIR, c, '*.jpg'))
        paths.extend(ps)
        labels.extend([c] * len(ps))

    X_tr, X_te, y_tr, y_te = train_test_split(paths, labels, test_size=TEST_SIZE, stratify=labels, random_state=RS)

    model = SoftSPMBOW(vocab_size=VOCAB_SIZE, levels=LEVELS)
    model.fit(X_tr)

    X_tr_raw = model.transform(X_tr)
    X_te_raw = model.transform(X_te)

    X_tr_tfidf = model.tfidf(X_tr_raw)
    X_te_tfidf = model.apply_idf(X_te_raw)

    svc = GridSearchCV(LinearSVC(max_iter=20000, class_weight='balanced'),
                       {'C': [0.01, 0.1, 1, 10]}, cv=5, n_jobs=-1)
    svc.fit(X_tr_tfidf, y_tr)
    print("Best C:", svc.best_params_)

    y_pr = svc.predict(X_te_tfidf)
    print("\n🧪 Original Evaluation:")
    print(classification_report(y_te, y_pr))
    print("Accuracy:", accuracy_score(y_te, y_pr))

    # 🔁 Evaluate on Transformed Images
    transform_types = ['rotate', 'hflip', 'brightness', 'gaussian_noise']
    for t in transform_types:
        X_te_trans = model.transform(X_te, transform_type=t)
        X_te_trans_tfidf = model.apply_idf(X_te_trans)
        y_pred_trans = svc.predict(X_te_trans_tfidf)
        print(f"\n🔄 Evaluation with transformation: {t}")
        print(classification_report(y_te, y_pred_trans))
        print(f"Accuracy ({t}):", accuracy_score(y_te, y_pred_trans))

    save_pickle({'kmeans': model.kmeans, 'idf': model.idf, 'svm': svc}, 'model_soft_color.pkl')

# OG Accuracy: 0.5390924002186988
# Accuracy (rotate): 0.2990705303444505
# Accuracy (hflip): 0.45434663750683435
# Accuracy (brightness): 0.4980863860032805
# Accuracy (gaussian_noise): 0.49480590486604703
