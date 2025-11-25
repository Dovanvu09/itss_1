import os
import time
import hashlib
import json
import asyncio
import re
import traceback  # Thư viện để in chi tiết lỗi
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
    APP_NAME: str = "Translation API Pro (Auto-Discovery Mode)"
    API_V1_STR: str = "/api/v1"
    TRANSLATION_PROVIDER: str = os.getenv("PROVIDER", "mock")
    
    # API Keys
    GOOGLE_API_KEY: str = os.getenv("GOOGLE_API_KEY", "")
    
    # Constraints
    MAX_INPUT_LENGTH: int = 500
    TIMEOUT_SECONDS: int = 30 
    
    # DB
    DATABASE_URL: str = "sqlite:///./translation_history.db"

settings = Settings()

# --- KIỂM TRA KEY NGAY KHI CHẠY ---
print("-" * 50)
print(f"📡 Provider đang chọn: {settings.TRANSLATION_PROVIDER}")
if settings.GOOGLE_API_KEY:
    print(f"🔑 Google API Key: ĐÃ TÌM THẤY (Bắt đầu bằng: {settings.GOOGLE_API_KEY[:5]}...)")
else:
    print("❌ Google API Key: KHÔNG TÌM THẤY (Kiểm tra lại file .env)")
print("-" * 50)

# --- 2. DATABASE SETUP ---
Base = declarative_base()

class TranslationLog(Base):
    __tablename__ = "lookups"
    id = Column(Integer, primary_key=True, index=True)
    text_hash = Column(String(32), index=True)
    source_text = Column(Text)
    source_lang = Column(String(10))
    target_lang = Column(String(10))
    provider = Column(String(50))
    created_at = Column(DateTime, default=datetime.utcnow)
    latency_ms = Column(Float)

engine = create_engine(settings.DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def init_db():
    Base.metadata.create_all(bind=engine)

def get_db():
    db = SessionLocal()
    try: yield db
    finally: db.close()

# --- 3. MODELS ---
class TranslationRequest(BaseModel):
    text: str = Field(..., min_length=1)
    source_lang: str = "auto"
    target_lang: str
    glossary: Optional[Dict[str, str]] = None

    @validator('text')
    def validate_length(cls, v):
        if len(v) > settings.MAX_INPUT_LENGTH:
            raise ValueError(f"Text too long (max {settings.MAX_INPUT_LENGTH})")
        return v

class TranslationResponse(BaseModel):
    original_text: str
    translated_text: str
    source_lang_detected: Optional[str] = None
    notes: Optional[str] = None
    glossary_applied: bool = False
    provider: str

# --- 4. ADAPTER ---
class BaseAdapter:
    async def translate(self, text: str, source: str, target: str, glossary: Dict[str, str] = None) -> Dict[str, Any]:
        raise NotImplementedError

class MockAdapter(BaseAdapter):
    async def translate(self, text, source, target, glossary=None):
        await asyncio.sleep(0.5)
        return {
            "translated_text": f"[MOCK {target.upper()}] {text}",
            "notes": "Debug Mode: Mock Adapter",
            "source_lang": source
        }

class GeminiAdapter(BaseAdapter):
    def __init__(self):
        if not settings.GOOGLE_API_KEY:
            raise ValueError("❌ LỖI: Thiếu GOOGLE_API_KEY trong file .env")
        self.api_key = settings.GOOGLE_API_KEY
        self.selected_model = None # Sẽ được tìm thấy khi gọi lần đầu

    async def _find_working_model(self):
        """Hàm tự động hỏi Google xem Key này dùng được model nào"""
        print("🔍 Đang dò tìm model khả dụng cho Key của bạn...")
        list_models_url = f"https://generativelanguage.googleapis.com/v1beta/models?key={self.api_key}"
        
        async with httpx.AsyncClient() as client:
            try:
                resp = await client.get(list_models_url, timeout=10)
                if resp.status_code != 200:
                    print(f"❌ Không thể lấy danh sách model. Code: {resp.status_code}")
                    print(f"Chi tiết lỗi: {resp.text}")
                    # Nếu lỗi 400/403 ở đây nghĩa là Key hỏng hoặc chưa bật API
                    raise Exception(f"List Models Failed: {resp.status_code}")
                
                data = resp.json()
                models = data.get('models', [])
                
                # Lọc ra các model hỗ trợ generateContent
                valid_models = [
                    m['name'].replace('models/', '') 
                    for m in models 
                    if 'generateContent' in m.get('supportedGenerationMethods', [])
                ]
                
                if not valid_models:
                    raise Exception("Không tìm thấy model nào hỗ trợ generateContent cho Key này!")
                
                print(f"✅ Các model tìm thấy: {valid_models}")
                
                # Ưu tiên chọn Flash hoặc Pro
                for m in valid_models:
                    if 'flash' in m and '1.5' in m:
                        return m
                for m in valid_models:
                    if 'gemini-pro' in m:
                        return m
                
                # Nếu không có cái ưu tiên, lấy cái đầu tiên
                return valid_models[0]
                
            except Exception as e:
                print(f"⚠️ Lỗi khi dò model: {e}")
                # Fallback cuối cùng nếu không dò được
                return "gemini-1.5-flash"

    async def translate(self, text: str, source: str, target: str, glossary: Dict[str, str] = None) -> Dict[str, Any]:
        # Nếu chưa chọn được model, thì đi tìm
        if not self.selected_model:
            self.selected_model = await self._find_working_model()
            print(f"🎯 Đã chốt dùng model: {self.selected_model}")

        glossary_str = ""
        if glossary:
            glossary_str = f"Glossary: {json.dumps(glossary)}."

        prompt = (
            f"Translate the following text from {source} to {target}.\n"
            f"Text: \"{text}\"\n{glossary_str}\n"
            f"Return ONLY a JSON object with keys: translation, notes, detected_source."
        )

        payload = {
            "contents": [{
                "parts": [{"text": prompt}]
            }],
            "generationConfig": {
                "response_mime_type": "application/json"
            }
        }

        # Tạo URL với model đã tìm được
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.selected_model}:generateContent?key={self.api_key}"

        print(f"⏳ Đang gọi Google Gemini ({self.selected_model})...")
        
        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(url, json=payload, timeout=settings.TIMEOUT_SECONDS)
                
                if response.status_code != 200:
                    print(f"❌ Lỗi API: {response.status_code}")
                    print(f"Chi tiết: {response.text}")
                    # Nếu model tự tìm mà vẫn lỗi 404, thử reset để tìm lại lần sau
                    if response.status_code == 404:
                         self.selected_model = None 
                    raise Exception(f"Google Error {response.status_code}: {response.text}")

                data = response.json()
                try:
                    candidates = data.get('candidates', [])
                    if not candidates:
                         return {"translated_text": text, "notes": "Blocked/Empty", "source_lang": source}

                    raw_content = candidates[0]['content']['parts'][0]['text']
                    parsed = json.loads(raw_content)
                    return {
                        "translated_text": parsed.get("translation", ""),
                        "notes": parsed.get("notes", ""),
                        "source_lang": parsed.get("detected_source", source)
                    }
                except (KeyError, IndexError, json.JSONDecodeError) as e:
                    print(f"⚠️ Lỗi đọc JSON trả về: {e}")
                    return {
                        "translated_text": str(data), 
                        "notes": "Parsing Error", 
                        "source_lang": source
                    }

            except Exception as e:
                print(f"❌ Request Failed: {e}")
                raise e

def get_adapter() -> BaseAdapter:
    if settings.TRANSLATION_PROVIDER.lower() == "gemini":
        return GeminiAdapter()
    return MockAdapter()

# --- 5. SERVICE ---
class TranslationService:
    def __init__(self, db: Session):
        self.db = db
        self.adapter = get_adapter()

    async def process_translation(self, req: TranslationRequest) -> TranslationResponse:
        start_time = time.time()
        
        try:
            result = await self.adapter.translate(req.text, req.source_lang, req.target_lang, req.glossary)
        except Exception as e:
            print(f"❌ SERVICE ERROR: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

        try:
            latency = (time.time() - start_time) * 1000
            log_entry = TranslationLog(
                text_hash=hashlib.md5(req.text.encode()).hexdigest(),
                source_text=req.text,
                source_lang=req.source_lang,
                target_lang=req.target_lang,
                provider=settings.TRANSLATION_PROVIDER,
                latency_ms=latency
            )
            self.db.add(log_entry)
            self.db.commit()
        except Exception as db_err:
            print(f"⚠️ Lỗi lưu Database: {db_err}")

        return TranslationResponse(
            original_text=req.text,
            translated_text=result["translated_text"],
            source_lang_detected=result.get("source_lang"),
            notes=result.get("notes"),
            glossary_applied=False,
            provider=settings.TRANSLATION_PROVIDER
        )

# --- 6. ENDPOINTS ---
app = FastAPI(title=settings.APP_NAME)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

@app.on_event("startup")
def on_startup():
    init_db()

@app.post("/translate", response_model=TranslationResponse)
async def translate_text(request: TranslationRequest, db: Session = Depends(get_db)):
    print(f"\n📩 [INCOMING] {request.text} -> {request.target_lang}")
    try:
        service = TranslationService(db)
        return await service.process_translation(request)
    except Exception as e:
        print("❌ UNHANDLED EXCEPTION:")
        traceback.print_exc()
        raise e

@app.get("/health")
def health_check():
    return {"status": "ok", "provider": settings.TRANSLATION_PROVIDER}

if __name__ == "__main__":
    import uvicorn
    print("🚀 Đang khởi động Server (Auto-Discovery Mode)...")
    uvicorn.run(app, host="0.0.0.0", port=8000)