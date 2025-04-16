from fastapi import FastAPI, HTTPException
import numpy as np
import faiss
import joblib
import json


app = FastAPI()

# Load embeddings, index and id map
cache_dir = "cache"
text_embeddings = np.load(f'{cache_dir}/text_embeddings.npy')
numeric_embeddings = np.load(f'{cache_dir}/numeric_matrix.npy')
image_embeddings = np.load(f'{cache_dir}/image_embeddings.npy')
scaler = joblib.load(f'{cache_dir}/numeric_scaler.pkl')

index_path = f'{cache_dir}/faiss_index.idx'
id_map_path = f'{cache_dir}/id_map.json'

index = faiss.read_index(index_path)

with open(id_map_path, 'r') as f:
    id_map = json.load(f)

# FAISS similar products
def find_similar_products(product_id: str, num_similar: int):
    try:
        # Find the index of the input product using its uniq_id
        if product_id not in id_map:
            raise HTTPException(status_code=404, detail="Product ID not found")
        product_idx = id_map.index(product_id)
        
        # Combine the embeddings for the product
        combined_embeddings = np.hstack([text_embeddings[product_idx], numeric_embeddings[product_idx], image_embeddings[product_idx]])
        
        # Query FAISS (Replaced calculate_similarities)
        k = num_similar + 1  # Including the product itself
        D, I = index.search(np.array([combined_embeddings]), k)
        
        # FAISS generates the product itself also
        similar_ids = [id_map[i] for i in I[0][1:]]
        return similar_ids
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Endpoint
@app.get("/find_similar_products")
def get_similar_products(product_id: str, num_similar: int):
    try:
        similar_products = find_similar_products(product_id, num_similar)
        return similar_products
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)