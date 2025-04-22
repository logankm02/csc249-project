import cv2
import numpy as np
import os
import pickle
from glob import glob
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import normalize
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, accuracy_score
from tqdm import tqdm


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
    if hist.sum() > 0:
        hist /= (hist.sum() + 1e-7)
    return hist


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
        for p in tqdm(image_paths, desc="Gathering descriptors for KMeans fit", unit="img"):
            img_gray = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
            _, desc = self.sift.detectAndCompute(img_gray, None)
            if desc is not None:
                if self.use_rootsift:
                    desc = rootsift(desc)
                descriptors.append(desc)
        all_desc = np.vstack(descriptors).astype(np.float64)
        self.kmeans.fit(all_desc)

    def _compute_spm(self, image_bgr):
        img_gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        h, w = img_gray.shape
        kps, desc = self.sift.detectAndCompute(img_gray, None)
        if desc is None or len(desc)==0:
            spm_dim = self.vocab_size * sum((2**l)**2 for l in self.levels)
            return np.zeros(spm_dim, dtype=np.float32)
        if self.use_rootsift:
            desc = rootsift(desc)
        desc = desc.astype(np.float64)
        cell_hists = []
        for level in self.levels:
            cells = 2**level
            for _ in range(cells*cells):
                cell_hists.append(np.zeros(self.vocab_size, dtype=np.float32))
        centers = self.kmeans.cluster_centers_
        for kp, d in zip(kps, desc):
            x, y = kp.pt
            for lvl_idx, level in enumerate(self.levels):
                cells = 2**level
                cell_w, cell_h = w/cells, h/cells
                i = int(x//cell_w)
                j = int(y//cell_h)
                cell_idx = sum((2**l)**2 for l in self.levels[:lvl_idx]) + j*cells + i
                dists = np.linalg.norm(centers - d, axis=1)
                nearest = np.argsort(dists)[:self.soft_k]
                weights = 1.0 / (dists[nearest] + 1e-7)
                weights /= weights.sum()
                for word, wgt in zip(nearest, weights):
                    cell_hists[cell_idx][word] += wgt
        for hist in cell_hists:
            s = hist.sum()
            if s>0:
                hist /= s
        return np.concatenate(cell_hists)

    def transform(self, image_paths):
        feats = []
        for p in tqdm(image_paths, desc="Extracting SPM+Color features", unit="img"):
            img_bgr = cv2.imread(p)
            spm_hist = self._compute_spm(img_bgr)
            color_hist = compute_color_hist(img_bgr)
            feats.append(np.concatenate([spm_hist, color_hist]))
        return np.array(feats)

    def tfidf(self, X):
        df = np.sum(X>0, axis=0)
        n = X.shape[0]
        self.idf = np.log((n+1)/(df+1)) + 1
        X_tfidf = X * self.idf
        return normalize(X_tfidf, norm='l2')

    def apply_idf(self, X):
        return normalize(X*self.idf, norm='l2')


if __name__ == '__main__':
    DATA_DIR = 'caltech-101'
    VOCAB_SIZE = 2000
    LEVELS = (0,1,2)
    TEST_SIZE = 0.2
    RS = 42

    cats = sorted(d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d)))
    paths, labels = [], []
    for c in cats:
        ps = glob(os.path.join(DATA_DIR, c, '*.jpg'))
        paths.extend(ps); labels.extend([c]*len(ps))

    X_tr, X_te, y_tr, y_te = train_test_split(paths, labels, test_size=TEST_SIZE,
                                          stratify=labels, random_state=RS)

    model = SoftSPMBOW(vocab_size=VOCAB_SIZE, levels=LEVELS)
    print('Fitting vocabulary...')
    model.fit(X_tr)

    print('Extracting features...')
    X_tr_raw = model.transform(X_tr)
    X_te_raw = model.transform(X_te)

    print('Applying TF-IDF...')
    X_tr_tfidf = model.tfidf(X_tr_raw)
    X_te_tfidf = model.apply_idf(X_te_raw)

    print('Training SVM...')
    svc = GridSearchCV(LinearSVC(max_iter=20000, class_weight='balanced'),
                      {'C':[0.01,0.1,1,10]}, cv=5, n_jobs=-1)
    svc.fit(X_tr_tfidf, y_tr)
    print('Best C:', svc.best_params_)

    print('Evaluating...')
    y_pr = svc.predict(X_te_tfidf)
    print(classification_report(y_te, y_pr))
    print('Accuracy:', accuracy_score(y_te, y_pr))

    save_pickle({'kmeans': model.kmeans, 'idf': model.idf, 'svm': svc}, 'model_soft_color.pkl')

#     output:
#                        precision    recall  f1-score   support

# BACKGROUND_Google       0.54      0.32      0.40        93
#             Faces       0.69      0.70      0.69        87
#        Faces_easy       0.72      0.89      0.79        87
#          Leopards       0.57      1.00      0.73        40
#        Motorbikes       0.85      0.97      0.90       160
#         accordion       0.57      0.73      0.64        11
#         airplanes       0.88      0.98      0.93       160
#            anchor       0.43      0.38      0.40         8
#               ant       0.00      0.00      0.00         8
#            barrel       0.00      0.00      0.00         9
#              bass       0.00      0.00      0.00        11
#            beaver       0.00      0.00      0.00         9
#         binocular       0.18      0.29      0.22         7
#            bonsai       0.47      0.69      0.56        26
#             brain       0.47      0.75      0.58        20
#      brontosaurus       0.20      0.11      0.14         9
#            buddha       0.38      0.35      0.36        17
#         butterfly       0.25      0.11      0.15        18
#            camera       0.71      0.50      0.59        10
#            cannon       0.00      0.00      0.00         9
#          car_side       0.58      1.00      0.74        25
#       ceiling_fan       0.50      0.44      0.47         9
#         cellphone       0.25      0.17      0.20        12
#             chair       0.43      0.25      0.32        12
#        chandelier       0.29      0.29      0.29        21
#       cougar_body       0.14      0.11      0.12         9
#       cougar_face       0.39      0.50      0.44        14
#              crab       0.23      0.20      0.21        15
#          crayfish       0.11      0.07      0.09        14
#         crocodile       0.00      0.00      0.00        10
#    crocodile_head       0.00      0.00      0.00        10
#               cup       0.14      0.09      0.11        11
#         dalmatian       0.42      0.38      0.40        13
#       dollar_bill       0.67      0.60      0.63        10
#           dolphin       0.29      0.31      0.30        13
#         dragonfly       0.18      0.14      0.16        14
#   electric_guitar       0.27      0.20      0.23        15
#          elephant       0.25      0.15      0.19        13
#               emu       0.15      0.18      0.17        11
#         euphonium       0.56      0.69      0.62        13
#              ewer       0.33      0.29      0.31        17
#             ferry       0.11      0.08      0.09        13
#          flamingo       0.17      0.08      0.11        13
#     flamingo_head       0.22      0.22      0.22         9
#          garfield       0.75      0.43      0.55         7
#           gerenuk       0.11      0.14      0.12         7
#        gramophone       0.25      0.20      0.22        10
#       grand_piano       0.55      0.90      0.68        20
#         hawksbill       0.37      0.55      0.44        20
#         headphone       0.43      0.38      0.40         8
#          hedgehog       0.60      0.55      0.57        11
#        helicopter       0.07      0.06      0.06        18
#              ibis       0.24      0.25      0.24        16
#      inline_skate       0.80      0.67      0.73         6
#       joshua_tree       0.54      0.54      0.54        13
#          kangaroo       0.19      0.24      0.21        17
#             ketch       0.47      0.61      0.53        23
#              lamp       0.20      0.08      0.12        12
#            laptop       0.64      0.56      0.60        16
#             llama       0.40      0.38      0.39        16
#           lobster       0.00      0.00      0.00         8
#             lotus       0.17      0.15      0.16        13
#          mandolin       0.33      0.11      0.17         9
#            mayfly       0.00      0.00      0.00         8
#           menorah       0.50      0.41      0.45        17
#         metronome       1.00      0.50      0.67         6
#           minaret       0.50      0.80      0.62        15
#          nautilus       0.50      0.27      0.35        11
#           octopus       0.33      0.14      0.20         7
#             okapi       0.44      0.50      0.47         8
#            pagoda       0.64      0.78      0.70         9
#             panda       0.71      0.62      0.67         8
#            pigeon       0.38      0.33      0.35         9
#             pizza       0.41      0.64      0.50        11
#          platypus       0.00      0.00      0.00         7
#           pyramid       0.11      0.09      0.10        11
#          revolver       0.57      0.50      0.53        16
#             rhino       0.46      0.50      0.48        12
#           rooster       0.67      0.40      0.50        10
#         saxophone       0.20      0.12      0.15         8
#          schooner       0.33      0.08      0.12        13
#          scissors       0.43      0.38      0.40         8
#          scorpion       0.27      0.18      0.21        17
#         sea_horse       0.27      0.36      0.31        11
#            snoopy       0.38      0.43      0.40         7
#       soccer_ball       0.71      0.77      0.74        13
#           stapler       0.00      0.00      0.00         9
#          starfish       0.57      0.24      0.33        17
#       stegosaurus       0.40      0.50      0.44        12
#         stop_sign       0.65      0.85      0.73        13
#        strawberry       0.38      0.43      0.40         7
#         sunflower       0.76      0.94      0.84        17
#              tick       0.30      0.30      0.30        10
#         trilobite       0.48      0.82      0.61        17
#          umbrella       0.40      0.40      0.40        15
#             watch       0.60      0.58      0.59        48
#       water_lilly       0.00      0.00      0.00         7
#        wheelchair       0.47      0.58      0.52        12
#          wild_cat       0.00      0.00      0.00         7
#     windsor_chair       0.69      0.82      0.75        11
#            wrench       0.50      0.50      0.50         8
#          yin_yang       0.56      0.75      0.64        12

#          accuracy                           0.54      1829
#         macro avg       0.38      0.38      0.36      1829
#      weighted avg       0.50      0.54      0.51      1829

# Accuracy: 0.5401858939311099