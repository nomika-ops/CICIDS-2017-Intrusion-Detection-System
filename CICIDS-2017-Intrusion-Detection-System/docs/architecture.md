# Initial application architecture

```text
User uploads network-flow CSV
            |
            v
       Streamlit UI
            |
            v
    CSV validation layer
            |
            v
   (next) preprocessing pipeline
            |
            v
      (next) classifier
            |
            v
  prediction + confidence output
            |
            v
   dashboard / CSV export
```

## Design decisions

- Start with offline CSV inference, not live packet capture. The supplied project is based on CICIDS-2017 CSV flow records; it does not include a verified live-capture pipeline.
- Keep UI code separate from data validation and model inference so the interface can later be replaced with FastAPI without rewriting the ML logic.
- Do not integrate the archived `.pkl` models until their matching preprocessing and label-encoding artifacts are recovered and validated.
- Treat binary and multiclass classifiers as separate deployment choices. Choose one for the first end-to-end release; add the other after the first path is reproducible.
