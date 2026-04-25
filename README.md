# LLM Evaluation Dashboard

A production-style LLM evaluation pipeline that runs prompts through multiple 
language models and scores their responses on custom quality metrics.

Built to demonstrate real-world GenAI engineering skills — model integration, 
automated evaluation, semantic scoring, and interactive visualization.

---

## Live Demo

[View Dashboard](https://llm-evaluation-dashboard-omnk9o5vwihthr39c8vx4y.streamlit.app)

---

## What It Does

- Sends a dataset of questions to 3 different LLM models simultaneously
- Automatically scores each response on 3 quality metrics
- Visualizes results in an interactive comparison dashboard
- Identifies which model performs best per metric and overall

---

## Models Compared

| Model | Provider | Size | Type |
|---|---|---|---|
| Llama 3.3 70B | Groq Cloud | 70B params | Cloud API |
| Llama 3.1 8B | Groq Cloud | 8B params | Cloud API |
| Qwen 2.5 1.5B | Ollama (local) | 1.5B params | Runs locally |

---

## Evaluation Metrics

**Faithfulness** — How accurately does the response reflect the reference answer?  
Scored using cosine similarity between sentence embeddings (all-MiniLM-L6-v2).

**Relevance** — How well does the response actually answer the question?  
Scored using semantic similarity between the response and the question.

**Toxicity** — Does the response contain harmful or inappropriate content?  
Scored using Detoxify, a BERT-based toxicity classifier.

---

## Tech Stack

| Layer | Tool |
|---|---|
| LLM APIs | Groq (Llama 3.3 70B, Llama 3.1 8B) |
| Local model | Ollama + Qwen 2.5 1.5B |
| Semantic scoring | sentence-transformers (all-MiniLM-L6-v2) |
| Toxicity scoring | Detoxify |
| Dashboard | Streamlit + Plotly |
| Data | Pandas |

---

## Project Structure
