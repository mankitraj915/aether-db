
# Aether: Distributed Vector Database

A high-performance, distributed vector database built from scratch in Python. Aether is designed to handle high-dimensional vector similarity search with **horizontal scalability** and **SIMD-optimized** compute.

## 🚀 Key Features

* **Distributed Sharding:** Implements a Leader-Follower architecture using **Consistent Hashing** to distribute data across multiple logical shards, ensuring linear scalability.
* **Vectorized Execution:** Replaces iterative Python loops with **NumPy Matrix Multiplication**, leveraging CPU SIMD instructions for batch similarity calculations.
* **Persistence:** Custom serialization engine using binary pickling to ensure data durability across server restarts.
* **RESTful API:** Fully documented FastAPI interface for `Upsert` and `Search` operations.

## 🛠️ Architecture

Aether follows a **Scatter-Gather** pattern:

1.  **Ingress:** API receives a query vector.
2.  **Scatter:** The Coordinator broadcasts the query to all active Shards in parallel.
3.  **Compute:** Each Shard performs a vectorized Dot Product search on its local dataset.
4.  **Gather:** Results are aggregated, sorted, and returned to the client.

## ⚡ Technical Highlights

| Feature | Implementation Detail |
| :--- | :--- |
| **Search Algo** | Cosine Similarity (Vectorized) |
| **Load Balancing** | SHA-256 Hashing (Mod N) |
| **Concurrency** | Non-blocking I/O (FastAPI/Uvicorn) |
| **Storage** | In-Memory with Disk Persistence (Binary Protocol) |

## 📦 Installation & Usage

### 1. Setup
```bash
# Clone the repo
git clone [https://github.com/yourusername/aether-db.git](https://github.com/yourusername/aether-db.git)
cd aether-db

# Install dependencies
pip install numpy fastapi uvicorn
