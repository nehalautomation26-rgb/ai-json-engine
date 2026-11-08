from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
import requests

app = FastAPI(title="Raw Text to Clean JSON API")

# Environment variable se key read karega (Render par set karenge)
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

class TextRequest(BaseModel):
    raw_text: str

@app.get("/")
def read_root():
    return {"status": "online", "message": "API Engine Active"}

@app.post("/extract-json")
def extract_json(payload: TextRequest):
    if not payload.raw_text:
        raise HTTPException(status_code=400, detail="raw_text field is required.")
    
    if not GROQ_API_KEY:
        raise HTTPException(status_code=500, detail="GROQ_API_KEY environment variable is not set.")

    system_prompt = (
        "You are an expert JSON parser. Extract key entities (names, emails, phones, "
        "company, role, key_points) from the given input text. Return ONLY a valid JSON object "
        "without markdown fences, commentary, or extra text."
    )
    
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    
    body = {
        "model": "llama-3.3-70b-versatile",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": payload.raw_text}
        ],
        "temperature": 0.1
    }
    
    response = requests.post("https://api.groq.com/openai/v1/chat/completions", headers=headers, json=body)
    
    if response.status_code != 200:
        raise HTTPException(status_code=500, detail=f"Groq LLM Engine Error: {response.text}")
        
    res_data = response.json()
    extracted_content = res_data["choices"][0]["message"]["content"]
    
    return {"status": "success", "data": extracted_content}