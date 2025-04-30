import os
import json
import jsonlines
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct

COLLECTION_NAME = "products"
BATCH_SIZE = 100

def create_client():
    return QdrantClient("localhost", port=6333)

def prepare_text_for_embedding(product):
    text_parts = []

    if product.get("title"):
        text_parts.append(product["title"])

    if product.get("about_this_item") and isinstance(product["about_this_item"], list):
        text_parts.extend(product["about_this_item"])

    if product.get("product_description"):
        text_parts.append(product["product_description"])

    return "\n".join(text_parts)

def main():
    print("Loading embedding model...")
    model = SentenceTransformer('all-MiniLM-L6-v2')

    client = create_client()

    collections = client.get_collections().collections
    collection_names = [collection.name for collection in collections]
    vector_size = model.get_sentence_embedding_dimension()
    
    if COLLECTION_NAME not in collection_names:
        print(f"Creating collection '{COLLECTION_NAME}'...")
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
        )

    points = []
    ids = []
    count = 0
    
    print("Reading data and creating embeddings...")
    with jsonlines.open("dpdata.jsonl") as reader:
        for item in tqdm(reader):
            text_to_embed = prepare_text_for_embedding(item)
            
            if not text_to_embed:
                continue
            
            embedding = model.encode(text_to_embed)
            
            point = PointStruct(
                id=count,
                vector=embedding.tolist(),
                payload=item
            )
            
            points.append(point)
            ids.append(count)
            count += 1
            
            if len(points) >= BATCH_SIZE:
                client.upsert(
                    collection_name=COLLECTION_NAME,
                    points=points
                )
                print(f"Uploaded batch of {len(points)} points. Total: {count}")
                points = []
    
    if points:
        client.upsert(
            collection_name=COLLECTION_NAME,
            points=points
        )
        print(f"Uploaded final batch of {len(points)} points. Total: {count}")
    
    print(f"Indexing complete! {count} products indexed.")

if __name__ == "__main__":
    main() 