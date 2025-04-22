import torch
import torchvision.transforms as transforms
from torchvision.models import resnet50, ResNet50_Weights
from PIL import Image
import numpy as np
import os
from glob import glob
import pickle
from tqdm import tqdm  # Progress bar

# CNN Feature Extractor using pretrained ResNet-50
class CNNFeatureExtractor:
    def __init__(self):
        self.model = resnet50(weights=ResNet50_Weights.DEFAULT)
        self.model.eval()
        self.model = torch.nn.Sequential(*list(self.model.children())[:-1])  # Remove FC layer
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def extract(self, image_path):
        image = Image.open(image_path).convert('RGB')
        input_tensor = self.transform(image).unsqueeze(0)
        with torch.no_grad():
            features = self.model(input_tensor).squeeze().numpy()
        return features / np.linalg.norm(features)

def save_pickle(obj, path):
    with open(path, 'wb') as f:
        pickle.dump(obj, f)

if __name__ == "__main__":
    extractor = CNNFeatureExtractor()
    DATASET_DIR = 'caltech-101'
    image_paths = glob(os.path.join(DATASET_DIR, '*/*.jpg'))
    feature_list = []

    for img_path in tqdm(image_paths, desc="Extracting CNN features"):
        features = extractor.extract(img_path)
        feature_list.append(features)

    save_pickle((image_paths, feature_list), 'cnn_features.pkl')

    # Save the CNN feature for the query image
    query_img = 'query2.jpg'
    query_feat = extractor.extract(query_img)
    save_pickle(query_feat, 'query_cnn_feature.pkl')
