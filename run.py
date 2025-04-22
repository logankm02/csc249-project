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

query_img = 'Random_Turtle.jpg'
query_desc = extract_sift_descriptors(query_img, sift)
query_hist = compute_bovw_histogram(query_desc, kmeans)

# Load DB histograms
image_paths, image_histograms = load_pickle('db_histograms.pkl')
sims = cosine_similarity([query_hist], image_histograms).flatten()
top_k = np.argsort(sims)[::-1][:5]

print("\n🔍 Top Matches:")
for idx in top_k:
    print(f"{image_paths[idx]} (score: {sims[idx]:.4f})")
