import cv2
import numpy as np
import os
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics.pairwise import cosine_similarity
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
