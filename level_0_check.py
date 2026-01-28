import numpy as np

def calculate_similarity(vec_a, vec_b):
    a = np.array(vec_a)
    b = np.array(vec_b)
    
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    
    return dot_product / (norm_a * norm_b)

# --- TEST ---
king = [1, 1, 1]
queen = [1, 1, 0.9]
table = [0, 0, 1]

print(f"Similarity (King vs Queen): {calculate_similarity(king, queen):.4f}")
print(f"Similarity (King vs Table): {calculate_similarity(king, table):.4f}")