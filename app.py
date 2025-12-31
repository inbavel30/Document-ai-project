import streamlit as st
from model import classify_document, summarize_document

st.title("📄 Intelligent Document AI")
st.markdown(
    "<h4 style='color:#00F5FF;'>AI-Based Document Classification & Summarization</h4>",
    unsafe_allow_html=True
)

st.divider()

uploaded_file = st.file_uploader("📂 Upload a TXT document", type=["txt"])

if uploaded_file:
    text = uploaded_file.read().decode("utf-8")

    st.markdown("<h4 style='color:#FF00FF;'>📝 Document Preview</h4>", unsafe_allow_html=True)
    st.text_area("", text, height=180)

    if st.button("🚀 Analyze Document"):
        with st.spinner("🤖 AI is analyzing..."):
            doc_type, confidence = classify_document(text)
            summary = summarize_document(text)

        st.success("✅ Analysis Completed")

        st.markdown(
            f"<h3 style='color:#00FF88;'>📌 Document Type: {doc_type}</h3>",
            unsafe_allow_html=True
        )

        st.markdown(
            f"<h4 style='color:#FFD700;'>Confidence: {confidence}%</h4>",
            unsafe_allow_html=True
        )
        st.progress(confidence / 100)

        st.markdown("<h4 style='color:#00F5FF;'>✂️ AI Summary</h4>", unsafe_allow_html=True)
        st.info(summary)

st.divider()
st.markdown(
    "<center style='color:#888;'>🖤 AIML Mini Project | Full Dark Theme UI</center>",
    unsafe_allow_html=True
)

