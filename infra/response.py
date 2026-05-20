import time
import pandas as pd

from openai import OpenAI
from google import genai

class Inference:
    def __init__(self, gemini_token: str = None,
                 qwen_token: str = None,
                 gemini_model: str = None,
                 qwen_model: str = None):
        # Loading gemini
        self.gemini_client = genai.Client(api_key = gemini_token)

        # Loading qwen
        self.qwen_client = OpenAI(
            api_key=qwen_token,
            base_url="https://api.groq.com/openai/v1"
        )

        self.gemini_model = gemini_model
        self.qwen_model = qwen_model

    def call_gemini(self, prompt: str) -> str:
        response = self.gemini_client.models.generate_content(
            model = self.gemini_model,
            contents = prompt,
            config = {"max_output_tokens": 50}
        )

        #para to sa mga none shi
        try:
            return response.text.strip() if response.text else "[REFUSED]"
        except Exception:
            return "[REFUSED]"

    def call_qwen(self, prompt: str) -> str:
        response = self.qwen_client.chat.completions.create(
            model = self.qwen_model,
            messages = [{"role": "user", "content": prompt}],
            max_tokens = 150,
            extra_body = {"reasoning_effort": "none"}
        )

        # Ensuring against nonetype responses
        content = response.choices[0].message.content

        return content.strip() if content != None else "[REFUSED]"

    def generate_responses(self, df: pd.DataFrame):
        rows = []

        for i, (_, row) in enumerate(df.iterrows()):
            # Gemini inference
            try:
                response_gemini = self.call_gemini(row["translated_prompt"])

            except Exception as e:
                print(f"[{i+1}/{len(df)}] Gemini FAILED | {row['language']}: {e}")
                response_gemini = None

            # Qwen inference
            try:
                response_qwen = self.call_qwen(row["translated_prompt"])

            except Exception as e:
                print(f"[{i+1}/{len(df)}] Qwen FAILED | {row['language']}: {e}")
                response_qwen = None

            rows.append({
                "source": row["source"],
                "category": row["category"],
                "original_prompt": row["original_prompt"],
                "language": row["language"],
                "translated_prompt": row["translated_prompt"],
                "response_gemini": response_gemini,
                "response_qwen": response_qwen,
            })

            print(f"[{i+1}/{len(df)}] {row['language']} done")
            time.sleep(4)

        df_responses = pd.DataFrame(rows)

        return df_responses