import pandas as pd
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import json

# Load updated model with better retrieval quality
model = SentenceTransformer('all-mpnet-base-v2')

# Load flattened, clean dataset
df = pd.read_csv("../data/shl_flattened.csv")

# Structured combined text
df['combined_text'] = (
    "Assessment Name: " + df['Assessment Name'].fillna('') + '. ' +
    "Description: " + df['Description'].fillna('') + '. ' +
    "Job Title: " + df['Job Title'].fillna('') + '. ' +
    "Test Types: " + df['Test Type Categories'].astype(str)
)

# Encode using mpnet
embeddings = model.encode(df['combined_text'].tolist(), show_progress_bar=True)
embeddings = np.array(embeddings).astype("float32")

# Build FAISS index
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)

# Save index and df
faiss.write_index(index, "../models/faiss_index_mpnet.index")
df.to_pickle("../models/df_mpnet.pkl")
