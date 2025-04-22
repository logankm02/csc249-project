import cv2
import numpy as np
import os
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics.pairwise import cosine_similarity
from glob import glob
import pickle

def extract_sift_descriptors(image_path, sift=None):
    if sift is None:
        sift = cv2.SIFT_create()
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    keypoints, descriptors = sift.detectAndCompute(img, None)
    return descriptors

def build_vocabulary(descriptor_list, vocab_size=500):
    kmeans = MiniBatchKMeans(n_clusters=vocab_size, batch_size=10000)
    all_descriptors = np.vstack(descriptor_list)
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

DATASET_DIR = 'caltech-101'
VOCAB_SIZE = 500

if __name__ == "__main__":
    # Step 1: Collect and extract SIFT descriptors
    image_paths = glob(os.path.join(DATASET_DIR, '*/*.jpg'))
    descriptor_list = []
    image_histograms = []
    sift = cv2.SIFT_create()

    for img_path in image_paths:
        print(img_path)
        desc = extract_sift_descriptors(img_path, sift)
        if desc is not None:
            descriptor_list.append(desc)
        else:
            print(f"Warning: No descriptors found for {img_path}")

    # Step 2: Build visual vocabulary
    kmeans = build_vocabulary(descriptor_list, vocab_size=VOCAB_SIZE)
    save_pickle(kmeans, 'kmeans_vocab.pkl')

    # Step 3: Compute and store histograms
    for img_path in image_paths:
        desc = extract_sift_descriptors(img_path, sift)
        hist = compute_bovw_histogram(desc, kmeans)
        image_histograms.append(hist)

    save_pickle((image_paths, image_histograms), 'db_histograms.pkl')