import os, requests

def call_openai(prompt, model="gpt-4o-mini"):
    r = requests.post(
        "https://api.openai.com/v1/chat/completions",
        headers={"Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}", "Content-Type": "application/json"},
        json={"model": model, "messages": [{"role": "user", "content": prompt}], "temperature": 0}
    )
    data = r.json()
    if "error" in data:
        raise Exception(data["error"])
    return data["choices"][0]["message"]["content"]

def ask(query, chunks, retrieve_fn):
    retrieved = retrieve_fn(query)
    context = "\n\n".join(f"[page {c['page']}] {c['text']}" for c in retrieved)
    prompt = f"""Answer the question using only the context below. Cite the page number(s) in square brackets. If the answer isn't in the context, say you don't know.

Context:
{context}

Question: {query}
Answer:"""
    return call_openai(prompt), retrieved
