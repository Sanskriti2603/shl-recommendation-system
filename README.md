#SHL Assessment Recommendation System

This project is an **AI-powered recommendation engine** that suggests the most relevant SHL assessments based on a job description or user query. It uses **mpnet embeddings**, **FAISS vector search**, and is deployed with **FastAPI + Streamlit**.

---

## 🔍 Features

- ✅ Web scraping using **BeautifulSoup** to extract assessment data from SHL
- ✅ Cleaned and flattened JSON dataset (~500+ assessments)
- ✅ Embedding with **sentence-transformers (all-mpnet-base-v2)**
- ✅ Semantic search using **FAISS index**
- ✅ REST API built with **FastAPI**
- ✅ Frontend UI with **Streamlit**
- ✅ Deployed backend on **Render**
- ✅ Deployed frontend on **Streamlit Cloud** or **Hugging Face Spaces**

---

## 🛠️ Folder Structure

```
shl_recommendation_project/
├── app/                  # FastAPI backend
│   └── main.py
├── notebook/            # Data prep and index building
│   ├── prepare_data.py
│   ├── build_index.py
│   └── evaluate.py
├── models/              # FAISS index and embeddings
│   ├── faiss_index_mpnet.index
│   └── df_mpnet.pkl
├── data/                # Processed CSVs from scraped JSON
│   └── shl_flattened.csv
├── streamlit_ui.py      # Streamlit frontend UI
├── requirements.txt     # All dependencies
├── render.yaml          # Render deployment config
└── README.md            # This file
```

---

## ⚙️ How It Works

1. **Scraping**: SHL product catalog scraped using BeautifulSoup → stored as structured JSON
2. **Data Prep**: Flattened into CSV, cleaned, and embedded using mpnet
3. **Indexing**: FAISS index created on embeddings
4. **API**: `/recommend` endpoint returns top matches for a text query
5. **UI**: Streamlit app calls the FastAPI backend and shows assessment cards

---

## 🚀 Deployment

### 🔹 Backend (FastAPI) — [Render.com](https://render.com)
- `render.yaml` included
- Start Command:
  ```bash
  uvicorn app.main:app --host 0.0.0.0 --port 10000
  ```

### 🔹 Frontend (Streamlit UI) — [Streamlit Cloud](https://streamlit.io/cloud)
- Upload `streamlit_ui.py` and `requirements.txt`
- API URL must point to the Render backend

---

## 🧪 Example Usage (API)

```json
POST /recommend
{
  "text": "Looking for a sales assessment for client-facing roles"
}
```

Returns a list of recommended assessments with links and metadata.

---

## 📦 Dependencies
```txt
faiss-cpu
pandas
numpy
beautifulsoup4
sentence-transformers
streamlit
requests
fastapi
uvicorn
```

---

## 🙋‍♀️ Maintainer
**Sanskriti Singh**  
Created during SHL AI Internship Application Project
