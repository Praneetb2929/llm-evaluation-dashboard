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
llm-eval-dashboard/

├── dataset.json         # evaluation questions with reference answers

├── model_runner.py      # calls all 3 models, collects responses

├── scorer.py            # computes faithfulness, relevance, toxicity

├── main.py             # orchestrates the full pipeline

├── dashboard.py        # Streamlit visualization app

├── requirements.txt    # dependencies

└── .gitignore          # keeps secrets out of version control

---

## How to Run Locally

**1. Clone the repo**
```bash
git clone https://github.com/Praneetb2929/llm-evaluation-dashboard.git
cd llm-evaluation-dashboard
```

**2. Create virtual environment**
```bash
python -m venv venv
venv\Scripts\activate       # Windows
source venv/bin/activate    # Mac/Linux
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. Add API keys**

Create a `.env` file in the root folder:
GROQ_API_KEY=your_groq_key_here
HF_TOKEN=your_hf_token_here
Get free keys from:
- Groq — console.groq.com
- HuggingFace — huggingface.co/settings/tokens

**5. Start Ollama (for local model)**
```bash
ollama pull qwen2.5:1.5b
ollama serve
```

**6. Run the evaluation pipeline**
```bash
python main.py
```

**7. Launch the dashboard**
```bash
streamlit run dashboard.py
```

---

## Key Findings

- All three models scored within 1 point of each other (80.3–80.7/100) on 
  factual Q&A tasks, showing that smaller models are competitive with larger 
  ones on well-defined knowledge questions
- Qwen 2.5 1.5B running locally matched cloud-hosted 70B models on relevance, 
  suggesting over-parameterization for simple factual retrieval
- Toxicity scores were uniformly high (clean) across all models on the 
  technical dataset used

---

## Why This Project Matters

Most engineers stop at "call the API and display the output." This project 
goes further — it treats LLM output as data to be measured, compared, and 
understood systematically. That's what production AI teams actually do.

Skills demonstrated:
- Multi-model API integration (cloud + local)
- Automated NLP evaluation (embeddings, classifiers)
- Local LLM deployment with Ollama
- Secrets management and secure deployment
- Interactive data visualization

---

## Author

**Praneet Biswal**  
[GitHub](https://github.com/Praneetb2929)

---

## License

MIT
