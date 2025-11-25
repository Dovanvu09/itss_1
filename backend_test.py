import httpx
import time
import sys

API_URL = "http://localhost:8000/translate"

def main():
    print("\n--- 🚀 TERMINAL TRANSLATOR CLIENT 🚀 ---")
    print("Gõ 'exit' để thoát.\n")

    while True:
        text = input("✍️  Nhập văn bản cần dịch: ").strip()
        if text.lower() == 'exit': break
        if not text: continue

        payload = {
            "text": text,
            "source_lang": "auto",
            "target_lang": "vi",
            "glossary": {"AI": "Trí tuệ nhân tạo"} # Test thử glossary
        }

        try:
            print("⏳ Đang gửi yêu cầu...", end="\r")
            start = time.time()
            
            # Gửi request
            response = httpx.post(API_URL, json=payload, timeout=10)
            data = response.json()
            
            latency = (time.time() - start) * 1000
            
            # Xóa dòng đang chờ
            sys.stdout.write('\x1b[2K\r')
            
            if response.status_code == 200:
                print(f"✅ KẾT QUẢ ({int(latency)}ms):")
                print(f"   Input:  {data['original_text']}")
                print(f"   Output: \033[92m{data['translated_text']}\033[0m") # Màu xanh lá
                print(f"   Nguồn:  {data['provider']}")
                print("-" * 30)
            else:
                print(f"❌ LỖI: {data}")

        except Exception as e:
            print(f"\n❌ Không kết nối được Server: {e}")
            print("👉 Bạn đã chạy 'python main.py' chưa?")

if __name__ == "__main__":
    main()


