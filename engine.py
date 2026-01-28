import numpy as np
import uuid
import hashlib
import os
import pickle

# The file where we save our data
DB_FILE = "aether_data.pkl"

class DistributedVectorEngine:
    def __init__(self, num_shards=2):
        self.num_shards = num_shards
        # Initialize empty shards
        self.shards = [{} for _ in range(num_shards)]
        # Try to load existing data
        self._load_from_disk()

    def _get_shard_index(self, vector_id: str):
        """Determines which shard receives the data based on the ID."""
        hash_val = int(hashlib.sha256(vector_id.encode()).hexdigest(), 16)
        return hash_val % self.num_shards

    def insert(self, vector: list):
        vector_id = str(uuid.uuid4())
        shard_idx = self._get_shard_index(vector_id)
        
        # Store as float32 to save memory
        self.shards[shard_idx][vector_id] = np.array(vector, dtype=np.float32)
        
        # PERSISTENCE: Save immediately after write
        self._save_to_disk()
        
        print(f"[Distributed Log] Vector {vector_id} -> Stored on Shard {shard_idx} (Saved to Disk)")
        return vector_id

    def search(self, query_vector: list, limit: int = 3):
        query_vec = np.array(query_vector, dtype=np.float32)
        all_results = []

        print(f"[Distributed Log] Broadcasting search to {self.num_shards} shards (Vectorized)...")

        # SCATTER: Send query to ALL shards
        for idx, shard in enumerate(self.shards):
            shard_results = self._search_single_shard(shard, query_vec)
            all_results.extend(shard_results)

        # GATHER: Sort combined results by score (Highest first)
        all_results.sort(key=lambda x: x[1], reverse=True)
        return all_results[:limit]

    def _search_single_shard(self, store, query_vec):
        """OPTIMIZED: Matrix Multiplication"""
        if not store:
            return []
        
        vec_ids = list(store.keys())
        matrix = np.array(list(store.values()))
        
        # Matrix Dot Product (The HPC Step)
        dot_products = np.dot(matrix, query_vec)
        matrix_norms = np.linalg.norm(matrix, axis=1)
        query_norm = np.linalg.norm(query_vec)
        
        with np.errstate(divide='ignore', invalid='ignore'):
            scores = dot_products / (matrix_norms * query_norm)
        
        scores = np.nan_to_num(scores)
        return list(zip(vec_ids, scores))

    def _save_to_disk(self):
        """Serializes the shard data to a file."""
        try:
            with open(DB_FILE, 'wb') as f:
                pickle.dump(self.shards, f)
        except Exception as e:
            print(f"[System Error] Failed to save data: {e}")

    def _load_from_disk(self):
        """Loads data from file if it exists."""
        if os.path.exists(DB_FILE):
            try:
                with open(DB_FILE, 'rb') as f:
                    self.shards = pickle.load(f)
                total_vecs = sum(len(s) for s in self.shards)
                print(f"[System] Successfully loaded {total_vecs} vectors from {DB_FILE}")
            except Exception as e:
                print(f"[System Error] Corrupt data file, starting fresh: {e}")

# --- TEST BLOCK ---
if __name__ == "__main__":
    db = DistributedVectorEngine()
    print("Engine initialized.")