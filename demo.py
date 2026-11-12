import requests, streamlit as st

API = "http://localhost:8000"
st.set_page_config(page_title="H&M Recommender", layout="wide")
st.title("H&M Two-Stage Recommender")
st.caption("Stage 1: repurchase · bought-together · ALS · popular · two-tower  →  Stage 2: LightGBM ranker")

try:
    h = requests.get(f"{API}/health", timeout=3).json()
    st.success(f"API online · {h['users_with_recs']:,} users · {h['items_indexed']:,} items indexed")
except Exception:
    st.error("API is not running. Start it with: python -m uvicorn app.main:app --port 8000")
    st.stop()

def show(items, cols=4):
    for row in range(0, len(items), cols):
        for c, it in zip(st.columns(cols), items[row:row + cols]):
            with c.container(border=True):
                st.markdown(f"**{it['name']}**")
                st.write(f"{it['type']} · {it['colour']}")
                st.caption(f"ID {it['article_id']}")

tab1, tab2 = st.tabs(["Recommend for a customer", "Find similar items"])

with tab1:
    cid = st.text_input("Customer ID", "00039306476aaf41a07fed942884f16b30abfa83a2a8bea972019098d6406793")
    k = st.slider("Number of items", 4, 12, 12, key="k1")
    if st.button("Get recommendations"):
        r = requests.get(f"{API}/recommend/{cid}", params={"k": k}).json()
        st.info(f"Source: {r['source']}  ·  latency: {r['latency_ms']} ms")
        show(r["items"])

with tab2:
    aid = st.number_input("Article ID", value=909370001, step=1)
    k2 = st.slider("Number of items", 4, 12, 8, key="k2")
    if st.button("Find similar items"):
        resp = requests.get(f"{API}/similar/{int(aid)}", params={"k": k2})
        if resp.status_code == 404:
            st.warning("Unknown article ID")
        else:
            r = resp.json(); q = r["query_item"]
            st.write(f"Similar to: **{q['name']}** ({q['type']}, {q['colour']})")
            st.info(f"FAISS search latency: {r['latency_ms']} ms")
            show(r["similar"])