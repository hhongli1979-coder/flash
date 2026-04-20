
import os
import time
import requests
from openai import OpenAI

API_KEY = os.environ.get("RUNPOD_API_KEY", "")
PUBLIC_ENDPOINT_ID = "qwen3-32b-awq"
BASE_URL = f"https://api.runpod.ai/v2/{PUBLIC_ENDPOINT_ID}"


def ask_openai(question: str, max_tokens: int = 512) -> str:
    """使用 OpenAI SDK（同步阻塞）呼叫模型"""
    client = OpenAI(
        api_key=API_KEY,
        base_url=f"{BASE_URL}/openai/v1",
    )
    response = client.chat.completions.create(
        model="Qwen/Qwen3-32B-AWQ",
        messages=[{"role": "user", "content": question}],
        max_tokens=max_tokens,
        temperature=0.7,
        top_p=1,
    )
    return response.choices[0].message.content


def ask_requests(question: str, max_tokens: int = 512) -> str:
    """使用 requests（非同步提交 + 輪詢）呼叫模型"""
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}",
    }
    data = {
        "input": {
            "messages": [
                {"role": "assistant", "content": ""},
                {"role": "user", "content": question},
            ],
            "sampling_params": {
                "max_tokens": max_tokens,
                "temperature": 0.7,
                "seed": -1,
                "top_k": -1,
                "top_p": 1,
            },
        }
    }

    # 提交任務
    resp = requests.post(f"{BASE_URL}/run", headers=headers, json=data, timeout=30)
    resp.raise_for_status()
    job_id = resp.json()["id"]
    print(f"Job submitted: {job_id}")

    # 輪詢結果（最多等 3 分鐘）
    for i in range(36):
        time.sleep(5)
        r = requests.get(
            f"{BASE_URL}/status/{job_id}", headers=headers, timeout=30
        )
        r.raise_for_status()
        result = r.json()
        status = result["status"]
        print(f"[{i+1}] {status}")
        if status == "COMPLETED":
            tokens = result["output"][0]["choices"][0]["tokens"]
            return "".join(tokens)
        elif status == "FAILED":
            raise RuntimeError(f"Job failed: {result}")

    raise TimeoutError("Polling timeout after 3 minutes")


if __name__ == "__main__":
    question = "What is Runpod?"

    print("=== OpenAI SDK ===")
    print(ask_openai(question))

    print("\n=== requests ===")
    print(ask_requests(question))
