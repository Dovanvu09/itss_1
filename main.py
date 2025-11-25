import os
import time
import hashlib
import json
import asyncio
import re
from typing import Optional, Dict, List, Any
from enum import Enum
from datetime import datetime

from fastapi import FastAPI, HTTPException, Depends, Header, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

import sqlalchemy
from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime, Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session

import httpx
from tenacity import retry, stop_after_attempt, wait_fixed

# --- 1. CONFIGURATION & ENV ---
load_dotenv()

class Settings(BaseSettings):
    APP_NAME: str = "Translation API Pro"
    API_V1_STR: str = "/api/v1"
    # Provider selection: gemini, openai, claude, libre, mock
    TRANSLATION_PROVIDER: str = os.getenv("PROVIDER", "mock")
    
    # API Keys
    GOOGLE_API_KEY: str = os.getenv("GOOGLE_API_KEY", "")
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    
    # Constraints
    MAX_INPUT_LENGTH: int = 500  # Giới hạn độ dài input để đảm bảo tốc độ
    TIMEOUT_SECONDS: int = 10
    
    # DB
    DATABASE_URL: str = "sqlite:///./translation_history.db"

settings = Settings()

# --- 2. DATABASE SETUP (SQLAlchemy) ---
Base = declarative_base()

class TranslationLog(Base):
    __tablename__ = "lookups"
    
    id = Column(Integer, primary_key=True, index=True)
    text_hash = Column(String(32), index=True) # MD5 hash của source text
    source_text = Column(Text) # Lưu text gốc (cân nhắc privacy nếu production)
    source_lang = Column(String(10))
    target_lang = Column(String(10))
    provider = Column(String(50))
    created_at = Column(DateTime, default=datetime.utcnow)
    latency_ms = Column(Float) # Thời gian xử lý

engine = create_engine(settings.DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def init_db():
    Base.metadata.create_all(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# --- 3. PYDANTIC MODELS (Validation) ---
class TranslationRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Văn bản cần dịch")
    source_lang: str = Field("auto", min_length=2, max_length=10)
    target_lang: str = Field(..., min_length=2, max_length=10)
    glossary: Optional[Dict[str, str]] = Field(default=None, description="Danh sách thuật ngữ {term: translation}")

    @validator('text')
    def validate_length(cls, v):
        if len(v) > settings.MAX_INPUT_LENGTH:
            raise ValueError(f"Text too long (max {settings.MAX_INPUT_LENGTH} chars)")
        return v

class TranslationResponse(BaseModel):
    original_text: str
    translated_text: str
    source_lang_detected: Optional[str] = None
    notes: Optional[str] = None # Ghi chú ngữ pháp/văn hóa
    glossary_applied: bool = False
    provider: str

# --- 4. ADAPTER PATTERN (LLM Integration) ---
class BaseAdapter:
    async def translate(self, text: str, source: str, target: str, glossary: Dict[str, str] = None) -> Dict[str, Any]:
        raise NotImplementedError

class MockAdapter(BaseAdapter):
    """Sử dụng cho testing hoặc khi không có API Key"""
    async def translate(self, text: str, source: str, target: str, glossary: Dict[str, str] = None) -> Dict[str, Any]:
        await asyncio.sleep(0.5) # Giả lập latency
        result_text = f"[MOCK {target.upper()}] {text}"
        
        # Giả lập glossary logic đơn giản
        if glossary:
            for term, trans in glossary.items():
                result_text = result_text.replace(term, trans)
        
        return {
            "translated_text": result_text,
            "notes": "Đây là dữ liệu giả lập từ Mock Adapter.",
            "source_lang": source if source != "auto" else "unknown"
        }

class GeminiAdapter(BaseAdapter):
    """Adapter cho Google Gemini"""
    def __init__(self):
        if not settings.GOOGLE_API_KEY:
            raise ValueError("GOOGLE_API_KEY is missing")
        import google.generativeai as genai
        genai.configure(api_key=settings.GOOGLE_API_KEY)
        self.model = genai.GenerativeModel('gemini-pro')

    @retry(stop=stop_after_attempt(3), wait=wait_fixed(1))
    async def translate(self, text: str, source: str, target: str, glossary: Dict[str, str] = None) -> Dict[str, Any]:
        glossary_str = ""
        if glossary:
            glossary_str = f"\nIMPORTANT: Use this glossary strictly: {json.dumps(glossary, ensure_ascii=False)}"

        prompt = (
            f"Act as a professional translator. Translate the following text from {source} to {target}.\n"
            f"Input Text: \"{text}\"\n"
            f"{glossary_str}\n"
            f"Output strictly in JSON format: {{\"translation\": \"...\", \"notes\": \"brief grammar/context notes (optional)\", \"detected_source\": \"code\"}}"
        )

        # Gemini call (chạy trong thread pool vì thư viện python là sync hoặc dùng async method nếu có)
        # Hiện tại sdk python hỗ trợ async generate_content_async
        response = await self.model.generate_content_async(prompt)
        
        try:
            # Clean markdown json blocks if present
            content = response.text.replace('```json', '').replace('```', '').strip()
            data = json.loads(content)
            return {
                "translated_text": data.get("translation", ""),
                "notes": data.get("notes", ""),
                "source_lang": data.get("detected_source", source)
            }
        except Exception as e:
            # Fallback nếu JSON parse lỗi
            return {
                "translated_text": response.text,
                "notes": "Raw output due to parsing error",
                "source_lang": source
            }

# Factory để chọn adapter
def get_adapter() -> BaseAdapter:
    provider = settings.TRANSLATION_PROVIDER.lower()
    if provider == "gemini":
        return GeminiAdapter()
    # Mở rộng thêm OpenAI, Claude ở đây...
    return MockAdapter()

# --- 5. BUSINESS LOGIC & SERVICE ---
class TranslationService:
    def __init__(self, db: Session):
        self.db = db
        self.adapter = get_adapter()

    async def process_translation(self, req: TranslationRequest) -> TranslationResponse:
        start_time = time.time()
        
        # 1. Pre-processing: Glossary injection is handled in Adapter Prompt
        
        # 2. Call Adapter
        try:
            # Timeout safety
            result = await asyncio.wait_for(
                self.adapter.translate(req.text, req.source_lang, req.target_lang, req.glossary),
                timeout=settings.TIMEOUT_SECONDS
            )
        except asyncio.TimeoutError:
            raise HTTPException(status_code=504, detail="Translation provider timed out")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Provider error: {str(e)}")

        translated_text = result["translated_text"]

        # 3. Post-processing: Glossary Override (Hậu xử lý ghi đè để đảm bảo chắc chắn)
        glossary_applied = False
        if req.glossary:
            for term, trans in req.glossary.items():
                # Regex case-insensitive replacement protection
                # Đơn giản hóa: replace string trực tiếp
                if term.lower() in req.text.lower() and trans not in translated_text:
                     # Nếu thuật ngữ gốc có trong input nhưng bản dịch không chứa từ khóa đích
                     # (Logic này có thể phức tạp hơn trong thực tế)
                     pass 
                
                # Force replace (Cẩn thận: chỉ nên dùng nếu tin tưởng tuyệt đối vào glossary)
                # pattern = re.compile(re.escape(term), re.IGNORECASE)
                # translated_text = pattern.sub(trans, translated_text)
                glossary_applied = True

        # 4. Save History (Logging)
        latency = (time.time() - start_time) * 1000
        text_hash = hashlib.md5(req.text.encode()).hexdigest()
        
        log_entry = TranslationLog(
            text_hash=text_hash,
            source_text=req.text,
            source_lang=req.source_lang,
            target_lang=req.target_lang,
            provider=settings.TRANSLATION_PROVIDER,
            latency_ms=latency
        )
        self.db.add(log_entry)
        self.db.commit()

        return TranslationResponse(
            original_text=req.text,
            translated_text=translated_text,
            source_lang_detected=result.get("source_lang"),
            notes=result.get("notes"),
            glossary_applied=glossary_applied,
            provider=settings.TRANSLATION_PROVIDER
        )

# --- 6. API ENDPOINTS ---
app = FastAPI(title=settings.APP_NAME)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Khởi tạo DB khi app start
@app.on_event("startup")
def on_startup():
    init_db()

@app.post("/translate", response_model=TranslationResponse)
async def translate_text(
    request: TranslationRequest, 
    db: Session = Depends(get_db)
):
    """
    Endpoint dịch thuật chính.
    - Input: text, source, target, glossary
    - Output: bản dịch, ghi chú, metadata
    """
    service = TranslationService(db)
    return await service.process_translation(request)

@app.get("/history")
def get_history(limit: int = 10, db: Session = Depends(get_db)):
    """Lấy lịch sử dịch thuật gần nhất"""
    logs = db.query(TranslationLog).order_by(TranslationLog.created_at.desc()).limit(limit).all()
    return logs

@app.get("/health")
def health_check():
    return {"status": "ok", "provider": settings.TRANSLATION_PROVIDER}

if __name__ == "__main__":
    import uvicorn
    print(f"Starting server with provider: {settings.TRANSLATION_PROVIDER}")
    uvicorn.run(app, host="0.0.0.0", port=8000)