\# H\&M Two-Stage Recommendation Engine



An end-to-end, production-style recommender system built on the H\&M Personalized Fashion

Recommendations dataset (Kaggle). It uses the same two-stage design as large-scale industry

recommenders: fast \*\*candidate generation\*\* followed by a feature-rich \*\*learning-to-rank\*\* model,

served through a \*\*FastAPI\*\* service with a \*\*Streamlit\*\* demo.



\## Architecture



```

&#x20;                   STAGE 1 - Candidate generation (\~100-120 items per user)

&#x20;Customer  ──►  Repurchase | Bought-together | ALS (implicit) | Popular | Two-tower + FAISS

&#x20;history                                    │

&#x20;                                           ▼

&#x20;                   STAGE 2 - LightGBM LambdaRank (\~35 features)

&#x20;                   user · item · user-item · age-group trends · two-tower similarity

&#x20;                                           │

&#x20;                                           ▼

&#x20;                   Top-12  ──►  FastAPI (/recommend, /similar)  ──►  Streamlit demo

&#x20;                   (cold-start users → popular fallback)

```



\## Results



Time-based evaluation: trained on data up to 2020-09-15, tested on the following week

(2020-09-16 to 2020-09-22, \~69k customers). Metric: MAP@12 (the competition metric).



| Model | MAP@12 | Recall@12 |

|---|---|---|

| Popularity baseline | 0.00875 | 0.0255 |

| Buy-again + popular baseline | 0.02657 | 0.0537 |

| Two-stage v1 (4 sources + LightGBM) | 0.02758 | 0.0576 |

| \*\*Two-stage v2 (+ age-group, trend, repurchase features)\*\* | \*\*0.02855\*\* | \*\*0.0599\*\* |

| Two-stage v3 (+ two-tower source and score) | 0.02838 | 0.0596 |



\*\*Best model: v2, +7.5% MAP@12 over a strong buy-again baseline.\*\*



\### Candidate generation (Stage 1 recall)



| Source | Recall | Avg. candidates |

|---|---|---|

| ALS | 0.0400 | 35 |

| Two-tower (PyTorch + FAISS) | 0.0291 | 35 |

| 4 sources combined | 0.1429 | 97 |

| 4 sources + two-tower | \*\*0.1523\*\* | 121 |



\### Serving latency (local CPU)

\- `/recommend`: \~0.03 ms (pre-computed batch recommendations)

\- `/similar`: \~9-18 ms (real-time FAISS search over 40,058 item vectors)



\## Key learnings

\- \*\*A time-based split is essential.\*\* A random split leaks future purchases and inflates scores.

\- \*\*Strong baselines matter.\*\* In fashion, "buy again" alone beats popularity 3x.

\- \*\*Stage 1 recall caps the whole system.\*\* The ranker can only reorder what retrieval finds.

\- \*\*Better retrieval ≠ better ranking.\*\* The two-tower model raised candidate recall by 6.6%,

&#x20; but final MAP stayed flat (v3), because more candidates add more negatives and the ranker

&#x20; was trained on only 30k users.

\- \*\*Popularity (logQ) correction\*\* in the two-tower loss stops the model from only learning best-sellers.

\- \*\*Repurchase recency\*\* (`ui\_days\_since`) is the most important ranking feature.



\## Project structure

```

├── app/main.py          # FastAPI service: /health, /recommend, /similar

├── demo.py              # Streamlit demo UI

├── notebooks/           # Full pipeline: data, baselines, candidates, two-tower, ranker

├── Dockerfile

└── requirements.txt

```



\## How to run

1\. Run the notebook in Google Colab to download the data from Kaggle and create the

&#x20;  `serving\_data/` folder (data is not included because of the competition rules).

2\. Start the API:

&#x20;  ```

&#x20;  pip install -r requirements.txt

&#x20;  python -m uvicorn app.main:app --port 8000

&#x20;  ```

&#x20;  Open http://localhost:8000/docs

3\. Start the demo (in a second terminal):

&#x20;  ```

&#x20;  pip install streamlit requests

&#x20;  python -m streamlit run demo.py

&#x20;  ```

&#x20;  Open http://localhost:8501

4\. Or with Docker:

&#x20;  ```

&#x20;  docker build -t recsys-api .

&#x20;  docker run -p 8000:8000 recsys-api

&#x20;  ```



\## Tech stack

Python · Pandas · implicit (ALS) · PyTorch (two-tower) · FAISS · LightGBM (LambdaRank) ·

FastAPI · Streamlit · Docker · Google Colab



\## Next steps

\- Train the ranker on several weeks and more users to make use of the two-tower signal

\- MLflow experiment tracking and a scheduled retraining pipeline (Prefect/Airflow)

\- Feature store (Redis/Feast) for real-time features

\- Online A/B testing plan

