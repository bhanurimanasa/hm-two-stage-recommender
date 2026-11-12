import os, time, pickle
import numpy as np, pandas as pd, faiss
from fastapi import FastAPI, HTTPException

DATA = os.getenv("DATA_DIR", "serving_data")

preds    = pickle.load(open(f"{DATA}/preds_v2.pkl", "rb"))
popular  = pickle.load(open(f"{DATA}/top12_popular.pkl", "rb"))
item_ids = np.load(f"{DATA}/item_ids.npy")
item_vecs = np.load(f"{DATA}/item_vecs_v2.npy").astype("float32")
meta = pd.read_parquet(f"{DATA}/articles_small.parquet").set_index("article_id").to_dict("index")

index = faiss.IndexFlatIP(item_vecs.shape[1])
index.add(item_vecs)
pos = {int(a): i for i, a in enumerate(item_ids)}

app = FastAPI(title="H&M Two-Stage Recommender", version="2.0")

def describe(ids):
    out = []
    for a in ids:
        m = meta.get(int(a), {})
        out.append({"article_id": int(a), "name": m.get("prod_name"),
                    "type": m.get("product_type_name"), "colour": m.get("colour_group_name")})
    return out

@app.get("/health")
def health():
    return {"status": "ok", "users_with_recs": len(preds), "items_indexed": index.ntotal}

@app.get("/recommend/{customer_id}")
def recommend(customer_id: str, k: int = 12):
    t0 = time.perf_counter()
    if customer_id in preds:
        items, source = preds[customer_id][:k], "two_stage_v2"
    else:
        items, source = popular[:k], "popular_fallback"     # cold-start user
    return {"customer_id": customer_id, "source": source, "items": describe(items),
            "latency_ms": round((time.perf_counter() - t0) * 1000, 2)}

@app.get("/similar/{article_id}")
def similar(article_id: int, k: int = 12):
    if article_id not in pos:
        raise HTTPException(status_code=404, detail="Unknown article_id")
    t0 = time.perf_counter()
    _, nn = index.search(item_vecs[pos[article_id]][None, :], k + 1)
    ids = [int(item_ids[j]) for j in nn[0] if int(item_ids[j]) != article_id][:k]
    return {"article_id": article_id, "query_item": describe([article_id])[0],
            "similar": describe(ids), "latency_ms": round((time.perf_counter() - t0) * 1000, 2)}
