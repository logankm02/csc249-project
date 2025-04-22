import os
from glob import glob
import numpy as np
import cv2
from bow import extract_sift_descriptors, build_vocabulary, compute_bovw_histogram, save_pickle, load_pickle
from sklearn.metrics.pairwise import cosine_similarity

# Load trained kmeans model
kmeans = load_pickle('kmeans_vocab.pkl')

# Initialize SIFT extractor
sift = cv2.SIFT_create()

query_img = 'query2.jpg'
query_desc = extract_sift_descriptors(query_img, sift)
query_hist = compute_bovw_histogram(query_desc, kmeans)

# Load DB histograms
image_paths, image_histograms = load_pickle('db_histograms.pkl')
sift_sims = cosine_similarity([query_hist], image_histograms).flatten()
sift_top_k = np.argsort(sift_sims)[::-1][:5]

print("\n🔍 Top Matches:")
for idx in sift_top_k:
    print(f"{image_paths[idx]} (score: {sift_sims[idx]:.4f})")

print("\n🔍 Running CNN Feature Search (pre-extracted)...")

# Load CNN features
cnn_paths, cnn_features = load_pickle('cnn_features.pkl')

# Load query feature
query_cnn_path = 'query_cnn_feature.pkl'
query_cnn_vector = load_pickle(query_cnn_path)

# Similarity search (CNN)
cnn_sims = cosine_similarity([query_cnn_vector], cnn_features).flatten()
cnn_top_k = np.argsort(cnn_sims)[::-1][:5]

print("\n🔍 Top Matches (CNN Features):")
for idx in cnn_top_k:
    print(f"{cnn_paths[idx]} (score: {cnn_sims[idx]:.4f})")