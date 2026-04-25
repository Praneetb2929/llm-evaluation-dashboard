from sentence_transformers import SentenceTransformer, util
from detoxify import Detoxify

embedder = SentenceTransformer("all-MiniLM-L6-v2")   # free, runs locally, ~80MB
toxifier = Detoxify("original")                        # free toxicity classifier

def score_faithfulness(response, reference):
    # cosine similarity between response and reference embeddings
    emb_r = embedder.encode(response, convert_to_tensor=True)
    emb_ref = embedder.encode(reference, convert_to_tensor=True)
    score = util.cos_sim(emb_r, emb_ref).item()
    return round(score * 100, 1)   # convert to 0-100

def score_relevance(response, question):
    # how much does the response relate to the question
    emb_resp = embedder.encode(response, convert_to_tensor=True)
    emb_q = embedder.encode(question, convert_to_tensor=True)
    score = util.cos_sim(emb_resp, emb_q).item()
    return round(score * 100, 1)

def score_toxicity(response):
    result = toxifier.predict(response)
    # result["toxicity"] is 0.0 to 1.0
    # we return toxicity as a LOW-is-good score: 100 - (toxicity * 100)
    tox_raw = result["toxicity"]
    return round((1 - tox_raw) * 100, 1)   # 100 = clean, 0 = toxic

def score_all(results):
    for item in results:
        for model in ["llama70b", "groq", "ollama"]:
            resp = item[model]
            item[f"{model}_faithfulness"] = score_faithfulness(resp, item["reference"])
            item[f"{model}_relevance"] = score_relevance(resp, item["question"])
            item[f"{model}_toxicity"] = score_toxicity(resp)
    return results