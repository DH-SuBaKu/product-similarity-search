# Product Similarity Search – SAP ML Internship Assignment

This project implements a product similarity search microservice using text, numeric, and image embeddings in multimodal similarity. It exposes a FastAPI endpoint that returns similar products based on a given product ID.

## Folder Structure

```
product-similarity-assignment/
├── app.py                      # FastAPI microservice to serve similarity search
├── generate_embeddings.py      # Script to generate and save embeddings
├── requirements.txt            # For running the FastAPI app
├── requirements-embed.txt      # For running the embedding generation script
├── Dockerfile                  # Docker config for the API
├── cache/                      # Cached data and models
│   ├── text_embeddings.npy
│   ├── numeric_matrix.npy
│   ├── image_embeddings.npy
│   ├── numeric_scaler.pkl
│   ├── faiss_index.idx
│   └── id_map.json
```

---

## How to Run the FastAPI App

### Option 1: Run Locally with Python

1. Create a virtual environment:

   ```bash
   python -m venv venv
   source venv/bin/activate   # or venv\Scripts\activate on Windows
   ```

2. Install requirements:

   ```bash
   pip install -r requirements.txt
   ```

3. Start the app:

   ```bash
   uvicorn app:app --reload
   ```

4. Test it at: [http://localhost:8000/find_similar_products?product_id=YOUR_ID&num_similar=5](http://localhost:8000/find_similar_products?product_id=YOUR_ID&num_similar=5)

    With sample product id here -> http://127.0.0.1:8000/find_similar_products?product_id=a7cf70174a0011640d5a39fcff067612&num_similar=5

---

### Option 2: Run with Docker

1. Build the image:

   ```bash
   docker build -t similarity-app .
   ```

2. Run the container:

   ```bash
   docker run -p 8000:8000 similarity-app
   ```

3. Test it at: [http://localhost:8000/find_similar_products?product_id=YOUR_ID&num_similar=5](http://localhost:8000/find_similar_products?product_id=YOUR_ID&num_similar=5)

    With sample product id here -> http://127.0.0.1:8000/find_similar_products?product_id=a7cf70174a0011640d5a39fcff067612&num_similar=5

---

## How Embeddings Are Computed

> Run this only if you want to regenerate embeddings from scratch using `generate_embeddings.py`. NOTE: This deletes the previous embeddings.

```bash
pip install -r requirements-embed.txt
python generate_embeddings.py
```

- **Text**: SentenceTransformer (`all-MiniLM-L6-v2`)
- **Numeric**: Price, weight, and browse node features (MinMax scaled)
- **Image**: ResNet50 + global average pooled features

Each embedding is L2-normalized and saved to `.npy` files under `/cache`.

---

## How Search Works

- All three embeddings are combined into a single vector
- FAISS is used to index and search over combined vectors
- The `id_map.json` maps FAISS indices back to product IDs
- Querying `/find_similar_products` returns the top N most similar product IDs

---

## Notes

- The `cache/` folder is included with precomputed embeddings and FAISS index.
- No training is done and everything is based on pretrained models and simple similarity.

---

## References

- FAISS - https://github.com/facebookresearch/faiss
