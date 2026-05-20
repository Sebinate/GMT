import streamlit as st
import os
from dotenv import load_dotenv

from infra.reader import reader
from infra.translator import Translator
from infra.response import Inference
from infra.voting import LlamaJudge, SEAGuardClassifier, get_verdict
from utils.utils import save_file

load_dotenv()

#Initializing contants
# ================================================================
# | TODO: Create a separate file for constants                   |
# ================================================================
groq_token = os.getenv('groq_token')
gemini_token = os.getenv('gemini_token')
sealion_token = os.getenv('sealion_token')

LANGUAGES = {
    "tagalog": "Helsinki-NLP/opus-mt-en-tl",
    "cebuano": "Helsinki-NLP/opus-mt-en-ceb",
    "ilocano": "Helsinki-NLP/opus-mt-en-ilo"
}

st.title('GMT: Guardrailing Multilingual Toxicity')
st.divider()

file_source = st.selectbox(label='Select Data source', options = ['SGBench', 'StrongREJECT'])

st.header('Upload File here')

uploaded_file = st.file_uploader(
    "Choose a file to upload",
    type=["csv", "json"],
)
st.divider()

if file_source and uploaded_file:
    file = reader(file_source, uploaded_file)
    st.subheader('Actual File')
    st.dataframe(file.head(10), use_container_width = True)
    st.divider()
    le_translator = Translator(language = LANGUAGES).translate(df = file)
    
    inferencer = Inference(gemini_token = gemini_token,
                       qwen_token = groq_token,
                       gemini_model = "gemini-3.1-flash-lite-preview",
                       qwen_model = "qwen/qwen3-32b").\
                        generate_responses(le_translator)
    
    seaguard = SEAGuardClassifier(api_key = sealion_token).classify_dataframe(inferencer)
    
    judge = LlamaJudge(groq_token = groq_token).classify_dataframe(seaguard)
    
    st.subheader('Annotated Data')
    
    judge['verdict_gemini'] = judge.apply(lambda row: get_verdict(row, 'gemini'), axis=1)
    judge['verdict_qwen'] = judge.apply(lambda row: get_verdict(row, 'qwen'), axis=1)
    
    st.dataframe(judge.head(10), use_container_width = True)
    
    final = judge.to_csv(index=False).encode('utf-8')
    
    st.download_button(label = 'Download Annotated Data', data = final,
                       file_name = 'annotated_data.csv', mime = 'text/csv')