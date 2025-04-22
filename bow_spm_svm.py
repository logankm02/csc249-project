import cv2
import numpy as np
import os
from glob import glob
import pickle
import time
from tqdm import tqdm
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
    """
    RootSIFT normalization: L1-normalize then square-root each element.
    """
    eps = 1e-7
    desc_l1 = descriptors / (np.linalg.norm(descriptors, ord=1, axis=1, keepdims=True) + eps)
    return np.sqrt(desc_l1)


class SpatialPyramidBOW:
    def __init__(self, vocab_size=1000, levels=(0,1,2), use_rootsift=True, batch_size=10000):
        self.vocab_size = vocab_size
        self.levels = levels
        self.use_rootsift = use_rootsift
        self.sift = cv2.SIFT_create()
        self.kmeans = MiniBatchKMeans(n_clusters=vocab_size, batch_size=batch_size)
        self.idf = None

    def fit(self, image_paths):
        descriptor_list = []
        for path in tqdm(image_paths, desc="Extracting descriptors", unit="img"):
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

    def transform(self, image_paths):
        feats = []
        total_dim = self.vocab_size * sum((2**l)**2 for l in self.levels)
        for path in tqdm(image_paths, desc="Transforming images", unit="img"):
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

    # Split train/test
    X_train, X_test, y_train, y_test = train_test_split(
        image_paths, labels, test_size=TEST_SIZE,
        stratify=labels, random_state=RANDOM_STATE
    )

    spm = SpatialPyramidBOW(vocab_size=VOCAB_SIZE, levels=LEVELS, use_rootsift=True)
    print("Fitting vocabulary...")
    start_fit = time.perf_counter()
    spm.fit(X_train)
    end_fit = time.perf_counter()
    print(f"Vocabulary fit time: {end_fit - start_fit:.2f}s")

    # Training images
    print("Transforming training images to histograms...")
    start_train = time.perf_counter()
    X_train_hist = spm.transform(X_train)
    end_train = time.perf_counter()
    train_time = end_train - start_train
    print(f"Transformed {len(X_train)} images in {train_time:.2f}s ({train_time/len(X_train):.3f}s per image)")

    # Test images
    print("Transforming test images to histograms...")
    start_test = time.perf_counter()
    X_test_hist = spm.transform(X_test)
    end_test = time.perf_counter()
    test_time = end_test - start_test
    print(f"Transformed {len(X_test)} images in {test_time:.2f}s ({test_time/len(X_test):.3f}s per image)")

    # TF-IDF
    print("Applying TF-IDF...")
    X_train_tfidf = spm.tfidf_weight(X_train_hist)
    X_test_tfidf = spm.apply_idf(X_test_hist)

    print("Training SVM with cross-validation...")
    param_grid = {'C': [0.01, 0.1, 1, 10]}
    svc = GridSearchCV(LinearSVC(max_iter=10000), param_grid, cv=5, n_jobs=-1)
    svc.fit(X_train_tfidf, y_train)
    print("Best parameters:", svc.best_params_)

    y_pred = svc.predict(X_test_tfidf)
    print(classification_report(y_test, y_pred))
    print("Test accuracy:", accuracy_score(y_test, y_pred))

    save_pickle({'kmeans': spm.kmeans, 'idf': spm.idf, 'svm': svc}, 'bow_spm_svm_model.pkl')

#     output:
#                        precision    recall  f1-score   support

# BACKGROUND_Google       0.24      0.44      0.31        93
#             Faces       0.43      0.64      0.52        87
#        Faces_easy       0.57      0.79      0.66        87
#          Leopards       0.53      0.95      0.68        40
#        Motorbikes       0.66      0.98      0.79       160
#         accordion       0.55      0.55      0.55        11
#         airplanes       0.60      0.97      0.74       160
#            anchor       0.40      0.25      0.31         8
#               ant       0.00      0.00      0.00         8
#            barrel       0.00      0.00      0.00         9
#              bass       0.00      0.00      0.00        11
#            beaver       0.00      0.00      0.00         9
#         binocular       0.00      0.00      0.00         7
#            bonsai       0.21      0.31      0.25        26
#             brain       0.44      0.70      0.54        20
#      brontosaurus       0.00      0.00      0.00         9
#            buddha       0.20      0.12      0.15        17
#         butterfly       0.06      0.06      0.06        18
#            camera       0.29      0.20      0.24        10
#            cannon       0.00      0.00      0.00         9
#          car_side       0.61      0.76      0.68        25
#       ceiling_fan       0.50      0.11      0.18         9
#         cellphone       0.00      0.00      0.00        12
#             chair       0.00      0.00      0.00        12
#        chandelier       0.18      0.14      0.16        21
#       cougar_body       1.00      0.11      0.20         9
#       cougar_face       0.27      0.21      0.24        14
#              crab       0.00      0.00      0.00        15
#          crayfish       0.00      0.00      0.00        14
#         crocodile       0.00      0.00      0.00        10
#    crocodile_head       0.00      0.00      0.00        10
#               cup       1.00      0.09      0.17        11
#         dalmatian       0.50      0.31      0.38        13
#       dollar_bill       0.62      0.50      0.56        10
#           dolphin       0.00      0.00      0.00        13
#         dragonfly       0.22      0.14      0.17        14
#   electric_guitar       0.20      0.13      0.16        15
#          elephant       0.50      0.15      0.24        13
#               emu       0.25      0.09      0.13        11
#         euphonium       0.58      0.54      0.56        13
#              ewer       0.25      0.18      0.21        17
#             ferry       0.00      0.00      0.00        13
#          flamingo       0.25      0.08      0.12        13
#     flamingo_head       0.00      0.00      0.00         9
#          garfield       1.00      0.43      0.60         7
#           gerenuk       0.14      0.14      0.14         7
#        gramophone       0.11      0.10      0.11        10
#       grand_piano       0.48      0.65      0.55        20
#         hawksbill       0.14      0.15      0.15        20
#         headphone       0.40      0.25      0.31         8
#          hedgehog       0.78      0.64      0.70        11
#        helicopter       0.11      0.06      0.07        18
#              ibis       0.20      0.19      0.19        16
#      inline_skate       1.00      0.33      0.50         6
#       joshua_tree       0.33      0.08      0.12        13
#          kangaroo       0.20      0.18      0.19        17
#             ketch       0.42      0.35      0.38        23
#              lamp       0.33      0.08      0.13        12
#            laptop       0.38      0.31      0.34        16
#             llama       0.38      0.19      0.25        16
#           lobster       0.00      0.00      0.00         8
#             lotus       0.00      0.00      0.00        13
#          mandolin       0.00      0.00      0.00         9
#            mayfly       0.00      0.00      0.00         8
#           menorah       0.33      0.24      0.28        17
#         metronome       0.67      0.33      0.44         6
#           minaret       0.27      0.27      0.27        15
#          nautilus       0.60      0.27      0.38        11
#           octopus       0.50      0.14      0.22         7
#             okapi       0.50      0.25      0.33         8
#            pagoda       0.75      0.67      0.71         9
#             panda       0.57      0.50      0.53         8
#            pigeon       0.20      0.11      0.14         9
#             pizza       0.25      0.09      0.13        11
#          platypus       0.33      0.14      0.20         7
#           pyramid       0.00      0.00      0.00        11
#          revolver       0.75      0.38      0.50        16
#             rhino       0.33      0.17      0.22        12
#           rooster       0.25      0.10      0.14        10
#         saxophone       1.00      0.12      0.22         8
#          schooner       0.14      0.08      0.10        13
#          scissors       0.20      0.12      0.15         8
#          scorpion       0.09      0.06      0.07        17
#         sea_horse       0.12      0.09      0.11        11
#            snoopy       1.00      0.14      0.25         7
#       soccer_ball       0.50      0.46      0.48        13
#           stapler       0.00      0.00      0.00         9
#          starfish       0.33      0.24      0.28        17
#       stegosaurus       0.60      0.25      0.35        12
#         stop_sign       0.67      0.62      0.64        13
#        strawberry       0.50      0.29      0.36         7
#         sunflower       0.67      0.82      0.74        17
#              tick       0.20      0.10      0.13        10
#         trilobite       0.59      0.59      0.59        17
#          umbrella       0.27      0.27      0.27        15
#             watch       0.43      0.56      0.49        48
#       water_lilly       0.00      0.00      0.00         7
#        wheelchair       0.20      0.08      0.12        12
#          wild_cat       0.00      0.00      0.00         7
#     windsor_chair       0.88      0.64      0.74        11
#            wrench       0.50      0.25      0.33         8
#          yin_yang       0.44      0.67      0.53        12

#          accuracy                           0.44      1829
#         macro avg       0.33      0.24      0.26      1829
#      weighted avg       0.39      0.44      0.39      1829

# Accuracy: 0.44231820667031163