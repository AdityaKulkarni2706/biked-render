import numpy as np
import torch
from sentence_transformers import SentenceTransformer, util
from PIL import Image

class BikeSearchEngine:
    def __init__(self, emb_path = r"C:\Users\Adi\Downloads\all_embeddings.npy", prm_path = r"C:\Users\Adi\Downloads\all_parametric.npy"):
        """Initializes the engine, loads data, and sets up the model."""
        print("Initializing BikeSearchEngine...")
        
        # 1. Load the data
        self.params = np.load(prm_path)
        self.og_embs = np.load(emb_path).astype("float32")
        
        # 2. Load the model
        self.model = SentenceTransformer('clip-ViT-B-32')
        print("Engine ready!")

    def search_by_text(self, text_query: str, top_k: int = 3) -> list:
        """Takes a text string and returns the indices and scores of the best matches."""
        # Encode the text
        text_embedding = self.model.encode([text_query]).astype("float32")
        
        # Calculate similarity
        cos_scores = util.cos_sim(self.og_embs, text_embedding).squeeze()
        
        # Get the Top K matches
        top_results = torch.topk(cos_scores, k=top_k)
        
        # Format the output into a clean list of dictionaries
        results = []
        for score, idx in zip(top_results.values, top_results.indices):
            results.append({
                "index": idx.item(),
                "score": score.item(),
                "parameters": self.params[idx.item()]
            })
            
        return results

    def search_by_image(self, image_path_or_obj, top_k: int = 3) -> list:
        """Takes an image upload and returns the closest parametric bikes."""
        # CLIP can encode PIL images directly!
        image = Image.open(image_path_or_obj)
        image_embedding = self.model.encode(image).astype("float32")
        
        # Calculate similarity (same logic as text)
        cos_scores = util.cos_sim(self.og_embs, image_embedding).squeeze()
        top_results = torch.topk(cos_scores, k=top_k)
        
        results = []
        for score, idx in zip(top_results.values, top_results.indices):
            results.append({
                "index": idx.item(),
                "score": score.item(),
                "parameters": self.params[idx.item()]
            })
            
        return results
        
    def get_parameters(self, index: int):
        """Utility to grab parameters if the UI just passes an ID."""
        return self.params[index]
    
# test if class works
if __name__ == "__main__":
    # Initialize the engine
    engine = BikeSearchEngine(
        emb_path=r"C:\Users\Adi\Downloads\all_embeddings.npy",
        prm_path=r"C:\Users\Adi\Downloads\all_parametric.npy"
    )

    # -------------------------
    # Test 1: Text-based search
    # -------------------------
    print("\n=== TEXT SEARCH TEST ===")
    text_query = "aggressive road bike with steep head tube angle"
    text_results = engine.search_by_text(text_query, top_k=3)

    for i, res in enumerate(text_results, 1):
        print(f"\nResult {i}")
        print(f"Index: {res['index']}")
        print(f"Score: {res['score']:.4f}")
        print(f"Parameters shape: {res['parameters'].shape}")

    # -------------------------
    # Test 2: Image-based search
    # -------------------------
    print("\n=== IMAGE SEARCH TEST ===")
    image_path = r"Biked_Reference_Data\output\bcad\initial_base_bike.png"  # CHANGE THIS

    try:
        image_results = engine.search_by_image(image_path, top_k=3)

        for i, res in enumerate(image_results, 1):
            print(f"\nResult {i}")
            print(f"Index: {res['index']}")
            print(f"Score: {res['score']:.4f}")
            print(f"Parameters shape: {res['parameters'].shape}")

    except Exception as e:
        print("Image search failed:", e)

    # -------------------------
    # Test 3: Direct parameter fetch
    # -------------------------
    print("\n=== PARAMETER FETCH TEST ===")
    test_index = text_results[0]["index"]
    params = engine.get_parameters(test_index)
    print(f"Fetched parameters for index {test_index}")
    print(f"Shape: {params.shape}")