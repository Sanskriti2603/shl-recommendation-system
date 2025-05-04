from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.responses import JSONResponse
import pandas as pd
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import math

# ✅ Must be defined early
app = FastAPI()

# ✅ Load resources
model = SentenceTransformer('all-mpnet-base-v2')
index = faiss.read_index('models/faiss_index_mpnet.index')
df = pd.read_pickle('models/df_mpnet.pkl')

class Query(BaseModel):
    text: str

@app.post("/recommend")
def recommend(query: Query):
    embedding = model.encode([query.text]).astype("float32")
    _, I = index.search(embedding, 10)

    results = []
    for i in I[0]:
        item = df.iloc[i]

        result = {
            "Assessment Name": item["Assessment Name"],
            "URL": item["URL"],
            "Duration": float(item["Assessment Length"]) if not pd.isna(item["Assessment Length"]) else None,
            "Remote Testing Support": item["Remote Testing Support"],
            "Adaptive/IRT Support": item["Adaptive/IRT Support"],
            "Test Type(s)": item["Test Type Categories"]
        }

        # Filter out NaN, inf, -inf
        for k, v in result.items():
            if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
                result[k] = None

        results.append(result)

    return JSONResponse(content={"recommendations": results})
