import streamlit as st
from transformers import pipeline

st.set_page_config(page_title="VoxBrief AI", page_icon="🤖")

@st.cache_resource
def get_summarizer():
    return pipeline("summarization", model="facebook/bart-large-cnn")

st.title("🤖 VoxBrief: Neural Document Summarizer")

input_text = st.text_area("Input Document Content", placeholder="Paste high-volume text here...", height=250)

mode = st.radio("Compression Level", ["Concise", "Detailed", "Key Bullets"], horizontal=True)

if st.button("Generate Intelligence"):
    summarizer = get_summarizer()
    with st.spinner("AI processing in progress..."):
        result = summarizer(input_text, max_length=150, min_length=50, do_sample=False)
        st.subheader("Summary Result")
        st.info(result[0]['summary_text'])