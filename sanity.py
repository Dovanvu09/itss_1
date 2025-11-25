# sanity.py
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI
import os

load_dotenv(dotenv_path=Path(__file__).with_name(".env"))
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

r = client.responses.create(
    model="gpt-4o-mini",
    input=[{"role":"user","content":"Say 'pong' if you can hear me."}]
)
print("OUTPUT:", r.output_text)
