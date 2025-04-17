import os
import shutil
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sentence_transformers import SentenceTransformer
import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import joblib
import requests
from io import BytesIO
from sklearn.preprocessing import normalize
from tqdm import tqdm
import faiss
import json

# GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Remove Previous Cache
cache_dir = "cache"
os.makedirs(cache_dir, exist_ok=True)

# Data loading
df = pd.read_json(
    'data/marketing_sample_for_amazon_com-amazon_fashion_products__20200201_20200430__30k_data.ldjson', 
    lines=True
)
df = df.fillna('')

# Text Embeddings
model = SentenceTransformer('all-MiniLM-L6-v2', device=device)

if not os.path.exists(os.path.join(cache_dir, 'text_embeddings.npy')):
    df['text_blob'] = (
        df['product_name'].astype(str) + ' ' +
        df['meta_keywords'].astype(str) + ' ' +
        df['parent___child_category__all'].astype(str)
    )

    print("Encoding text embeddings ----------------------")
    text_embeddings = model.encode(df['text_blob'].tolist(), show_progress_bar=True, device=device)
    text_embeddings = normalize(text_embeddings, norm='l2')
    np.save(os.path.join(cache_dir, 'text_embeddings.npy'), text_embeddings)
else:
    text_embeddings = np.load(os.path.join(cache_dir, 'text_embeddings.npy'))

# Numerical Embeddings
print("Encoding text embeddings ----------------------")
if not os.path.exists(os.path.join(cache_dir, 'numeric_matrix.npy')):
    df['product_price'] = pd.to_numeric(df['sales_price'], errors='coerce').fillna(0)
    df['item_weight'] = pd.to_numeric(df['weight'], errors='coerce').fillna(0)
    df['browse_nodes'] = pd.to_numeric(df['browsenode'], errors='coerce').fillna(0) # Test this

    scaler = MinMaxScaler()
    numeric_matrix = scaler.fit_transform(df[['product_price', 'item_weight', 'browse_nodes']])
    numeric_matrix = normalize(numeric_matrix, norm='l2')

    np.save(os.path.join(cache_dir, 'numeric_matrix.npy'), numeric_matrix)
    joblib.dump(scaler, os.path.join(cache_dir, 'numeric_scaler.pkl'))
else:
    numeric_matrix = np.load(os.path.join(cache_dir, 'numeric_matrix.npy'))

# Image Embeddings
image_embedding_path = os.path.join(cache_dir, 'image_embeddings.npy')
if not os.path.exists(image_embedding_path):

    # Transforming the images
    
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


    # Load ResNet and remove final classification layer to get embeddings
    resnet = models.resnet50(pretrained=True)
    resnet = torch.nn.Sequential(*list(resnet.children())[:-1])  # Remove final fc layer
    resnet = resnet.to(device)
    resnet.eval()


    # Pull images and get embedding

    def get_image_embedding(image_url):
        try:
            response = requests.get(image_url, timeout=5)
            if response.status_code != 200:
                raise Exception(f"Status code: {response.status_code}")
            img = Image.open(BytesIO(response.content)).convert('RGB')
            img = transform(img).unsqueeze(0).to(device)
    
            with torch.no_grad():
                embedding = resnet(img).squeeze()  # [2048, 1, 1]
                embedding = embedding.view(-1)     # [2048]
            return embedding.cpu().numpy()
        except Exception as e:
            print(f"[WARN] Failed to process image ({image_url}): {e}")
            return np.zeros(2048)  


    print("Encoding image embeddings ----------------------")
    image_embeddings = []

    # TQDM for loading

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing Images"):
        image_url = row.get('medium', '')
        embedding = get_image_embedding(image_url) if image_url else np.zeros(2048) # 2048-dim FALLBACK
        image_embeddings.append(embedding)

    image_embeddings = normalize(np.array(image_embeddings), norm='l2')
    np.save(image_embedding_path, image_embeddings)
else:
    image_embeddings = np.load(image_embedding_path)

# Combine all Embeddings
combined_embeddings = np.hstack([text_embeddings, numeric_matrix, image_embeddings]) 

# FAISS Index
index_path = os.path.join(cache_dir, 'faiss_index.idx')
id_map_path = os.path.join(cache_dir, 'id_map.json')

if not os.path.exists(index_path):
    print("Creating FAISS index...")
    dim = combined_embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(combined_embeddings)
    faiss.write_index(index, index_path)
    print("FAISS index saved.")

    with open(id_map_path, 'w') as f:
        json.dump(df['uniq_id'].tolist(), f)
else:
    print("FAISS index already exists.")
