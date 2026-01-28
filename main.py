from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from engine import DistributedVectorEngine  # Updated import

app = FastAPI(title="Aether Distributed Vector DB")

# Initialize the Distributed Engine with 2 simulated shards
db = DistributedVectorEngine(num_shards=2)

class VectorPayload(BaseModel):
    vector: list[float]

@app.get("/")
def home():
    return {"message": "Aether Distributed DB is Online. Use /docs to test."}

@app.post("/upsert")
def upsert_vector(payload: VectorPayload):
    try:
        vid = db.insert(payload.vector)
        return {"status": "success", "id": vid}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/search")
def search_vector(payload: VectorPayload):
    try:
        results = db.search(payload.vector)
        
        # Clean up numpy types for JSON response
        clean_results = [
            {"id": r[0], "score": float(r[1])} for r in results
        ]
        return {"matches": clean_results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))