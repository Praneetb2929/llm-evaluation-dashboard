import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="LLM Eval Dashboard", layout="wide")
st.title("LLM Evaluation Dashboard")

df = pd.read_csv("eval_results.csv")
models = ["llama70b", "groq", "ollama"]
metrics = ["faithfulness", "relevance", "toxicity"]

# Sidebar filters
selected_metrics = st.sidebar.multiselect("Metrics", metrics, default=metrics)
selected_models = st.sidebar.multiselect("Models", models, default=models)

# Summary scores
st.subheader("Average scores by model")
summary = {}
for model in selected_models:
    scores = [df[f"{model}_{m}"].mean() for m in selected_metrics]
    summary[model] = round(sum(scores) / len(scores), 1)

cols = st.columns(len(selected_models))
for i, (model, score) in enumerate(summary.items()):
    cols[i].metric(model.upper(), f"{score}/100")

# Radar chart
st.subheader("Metric radar chart")
fig = go.Figure()
for model in selected_models:
    values = [df[f"{model}_{m}"].mean() for m in selected_metrics]
    fig.add_trace(go.Scatterpolar(r=values, theta=selected_metrics, fill="toself", name=model))
fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0,100])))
st.plotly_chart(fig, use_container_width=True)

# Per-question breakdown
st.subheader("Per-question breakdown")
for _, row in df.iterrows():
    with st.expander(f"Q{row['id']}: {row['question']}"):
        for model in selected_models:
            st.markdown(f"**{model.upper()}**")
            st.write(row[model])
            score_cols = st.columns(len(selected_metrics))
            for j, metric in enumerate(selected_metrics):
                score_cols[j].metric(metric, row[f"{model}_{metric}"])
        st.caption(f"Reference: {row['reference']}")