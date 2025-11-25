# app.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, Dict
from dotenv import load_dotenv
from google import genai
import os

# --- Load config & client ---
load_dotenv()
API_KEY = os.getenv("GOOGLE_API_KEY")
if not API_KEY:
    raise RuntimeError("Missing GOOGLE_API_KEY in .env")

client = genai.Client(api_key=API_KEY)
MODEL_ID = os.getenv("MODEL_ID", "models/gemini-2.0-flash")  # nhanh & rẻ cho dịch

# --- FastAPI app ---
app = FastAPI(
    title="Translator API",
    version="1.0.0",
    description="FastAPI + Gemini translation backend"
)

# CORS (cho phép frontend gọi)
allow_origins = os.getenv("ALLOW_ORIGINS", "*").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=allow_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Schemas ---
class TranslateReq(BaseModel):
    text: str = Field(..., description="Nguồn cần dịch")
    source: str = Field("auto", description="Mã ngôn ngữ nguồn, 'auto' để tự nhận diện")
    target: str = Field(..., description="Mã ngôn ngữ đích, ví dụ 'vi','en','ja'")
    style: str = Field("neutral", description="neutral|formal|casual|concise")
    glossary: Optional[Dict[str, str]] = Field(
        default=None, description="Từ điển thuật ngữ: nguồn -> đích"
    )

class TranslateResp(BaseModel):
    translation: str
    model: str = MODEL_ID

# --- Prompt chuẩn dịch ---
SYSTEM_PROMPT = """You are a professional translator.
- Preserve meaning, tone, punctuation, inline code, URLs, variables.
- Keep list/markdown structure.
- Do not add explanations; return only the translation text.
"""

def build_prompt(req: TranslateReq) -> str:
    gloss = ""
    if req.glossary:
        pairs = "\n".join(f"- {k} -> {v}" for k, v in req.glossary.items())
        gloss = f"\nUse this glossary consistently:\n{pairs}\n"
    return (
        f"{SYSTEM_PROMPT}\n"
        f"Source language: {req.source}\n"
        f"Target language: {req.target}\n"
        f"Style: {req.style}{gloss}\n\n"
        "Translate the following text:\n"
        "---SOURCE---\n"
        f"{req.text}\n"
        "---END---"
    )

# --- Routes ---
@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_ID}

@app.post("/translate", response_model=TranslateResp)
def translate(req: TranslateReq):
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Empty text")

    prompt = build_prompt(req)
    try:
        resp = client.models.generate_content(model=MODEL_ID, contents=prompt)
        out = (getattr(resp, "text", "") or "").strip()
        if not out:
            raise RuntimeError("Empty model response")
        return TranslateResp(translation=out)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Translation error: {e}")

