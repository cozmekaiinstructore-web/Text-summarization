import streamlit as st
from transformers import pipeline
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lex_rank import LexRankSummarizer
import pyttsx3


st.set_page_config(page_title="Text Summarization App", layout="centered")
st.title("📝 Text Summarization using NLP")
st.write("""
This application summarizes long paragraphs or articles into concise versions using both 
**Extractive (LexRank)** and **Abstractive (BART)** NLP techniques.
""")


@st.cache_resource
def load_model():
    return pipeline("summarization", model="facebook/bart-large-cnn")

summarizer = load_model()


st.subheader("Enter Text or Upload a File")

text_input = st.text_area("✍️ Paste or type your text here:", height=250)

uploaded_file = st.file_uploader("📁 Or upload a text file", type=["txt"])
if uploaded_file is not None:
    text_input = uploaded_file.read().decode("utf-8")
    st.success("✅ File uploaded successfully!")


if st.button("🔹 Generate Extractive Summary"):
    if text_input.strip():
        try:
            parser = PlaintextParser.from_string(text_input, Tokenizer("english"))
            summarizer_lex = LexRankSummarizer()
            summary_sentences = summarizer_lex(parser.document, 3)
            extractive_summary = " ".join(str(sentence) for sentence in summary_sentences)

            if extractive_summary:
                st.subheader("Extractive Summary (LexRank):")
                st.write(extractive_summary)
            else:
                st.warning("The text is too short for extractive summarization.")
        except Exception as e:
            st.error(f"Error: {e}")
    else:
        st.warning("Please enter or upload text first.")


if st.button("🔸 Generate Abstractive Summary"):
    if text_input.strip():
        with st.spinner("Generating summary... Please wait."):
            text_chunk = text_input[:1024]
            summary = summarizer(text_chunk, max_length=120, min_length=30, do_sample=False)
        st.subheader("Abstractive Summary (BART):")
        st.write(summary[0]['summary_text'])
    else:
        st.warning("Please enter or upload text first.")


if st.button("🔊 Read Summary Aloud"):
    try:
        engine = pyttsx3.init()
        engine.say("Here is your summarized text.")
        engine.runAndWait()
        st.info("✅ Reading completed! (Text-to-speech executed)")
    except Exception as e:
        st.error(f"Text-to-speech error: {e}")


st.markdown("---")
st.markdown("**Developed by:** Cozmek AIE | NLP Project | © 2025")
