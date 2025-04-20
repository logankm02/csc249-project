import os
from glob import glob
import numpy as np
import cv2
from bow import extract_sift_descriptors, build_vocabulary, compute_bovw_histogram, save_pickle, load_pickle
from sklearn.metrics.pairwise import cosine_similarity

DATASET_DIR = 'images'
VOCAB_SIZE = 500

# Step 1: Collect and extract SIFT descriptors
image_paths = glob(os.path.join(DATASET_DIR, '*/*.jpg'))
descriptor_list = []
image_histograms = []
sift = cv2.SIFT_create()

for img_path in image_paths:
    desc = extract_sift_descriptors(img_path, sift)
    if desc is not None:
        descriptor_list.append(desc)

# Step 2: Build visual vocabulary
kmeans = build_vocabulary(descriptor_list, vocab_size=VOCAB_SIZE)
save_pickle(kmeans, 'kmeans_vocab.pkl')

# Step 3: Compute and store histograms
for img_path in image_paths:
    desc = extract_sift_descriptors(img_path, sift)
    hist = compute_bovw_histogram(desc, kmeans)
    image_histograms.append(hist)

save_pickle((image_paths, image_histograms), 'db_histograms.pkl')

# Step 4: Retrieval example
query_img = 'query.jpg'
query_desc = extract_sift_descriptors(query_img, sift)
query_hist = compute_bovw_histogram(query_desc, kmeans)

# Load DB histograms
image_paths, image_histograms = load_pickle('db_histograms.pkl')
sims = cosine_similarity([query_hist], image_histograms).flatten()
top_k = np.argsort(sims)[::-1][:5]

print("\n🔍 Top Matches:")
for idx in top_k:
    print(f"{image_paths[idx]} (score: {sims[idx]:.4f})")
