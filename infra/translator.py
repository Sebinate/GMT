import pandas as pd

from transformers import MarianMTModel, MarianTokenizer

class Translator:
    def __init__(self, language: dict):
        self.LANGUAGES = language
        self.models = {}
        self.tokenizers = {}

        for lang_key, model_name in language.items():
            print(f"Loading model for {lang_key}")
            self.tokenizers[lang_key] = MarianTokenizer.from_pretrained(model_name)
            self.models[lang_key] = MarianMTModel.from_pretrained(model_name)
            print(f"{lang_key} model loaded.")

    def translatetinator(self, prompt: str, lang_key: str) -> str:
        tokenizer = self.tokenizers[lang_key]
        model = self.models[lang_key]

        inputs = tokenizer([prompt], return_tensors="pt", padding=True, truncation=True, max_length=512)
        translated = model.generate(**inputs)
        return tokenizer.decode(translated[0], skip_special_tokens=True)

    def translate(self, df: pd.DataFrame):
        rows = []

        for i, (_, row) in enumerate(df.iterrows()):
            for lang_key in self.LANGUAGES.keys():
                try:
                    translated = self.translatetinator(row["forbidden_prompt"], lang_key)
                    rows.append({
                        "source": row["source"],
                        "category": row["category"],
                        "original_prompt": row["forbidden_prompt"],
                        "language": lang_key,
                        "translated_prompt": translated
                    })
                    print(f"[{i+1}/{len(df)}] {lang_key}: {translated[:60]}...")

                except Exception as e:
                    print(f"[{i+1}/{len(df)}] {lang_key} FAILED: {e}")
                    rows.append({
                        "source": row["source"],
                        "category": row["category"],
                        "original_prompt": row["forbidden_prompt"],
                        "language": lang_key,
                        "translated_prompt": None
                    })

        df_translated = pd.DataFrame(rows)

        return df_translated