import streamlit as st
import os
import config
from backend import RAGSystem
import logging
from transformers import logging as hf_logging

hf_logging.set_verbosity_error()


# 1. SETUP ANTARMUKA (UI SETUP)

st.set_page_config(page_title="RAG Multimodal System", layout="wide")

# CSS Custom untuk merapikan margin atas
st.markdown("""
    <style>
        .block-container {padding-top: 1rem;}
        h1 {margin-bottom: 0px;}
        .stButton button {width: 100%;}
    </style>
""", unsafe_allow_html=True)

st.markdown("<h1 style='text-align: center;'>Image-based RAG System</h1>", unsafe_allow_html=True)
st.caption(f"Backend: {config.CLIP_MODEL_NAME} | {config.VLM_MODEL_NAME} | Cache: {config.MODELS_CACHE_DIR}")


# 2. LOAD SYSTEM (CACHING)

@st.cache_resource
def get_rag_system():
    try:
        return RAGSystem()
    except FileNotFoundError as e:
        st.error(str(e))
        st.stop()
    except Exception as e:
        st.error(f"Failed to load system: {e}")
        st.stop()

with st.spinner("Initializing System & Models... (Check ./models_cache folder)"):
    rag = get_rag_system()


# 3. LOGIKA TAMPILAN (DISPLAY LOGIC)

def display_results(results):
    if results:
        best = results[0]
        c1, c2 = st.columns([1, 1.5])

        with c1:
            # PERBAIKAN: Menggunakan width='stretch' sesuai standar Streamlit 2025
            st.image(best['path'], caption=f"Top Result: {best['label']}", width="stretch")
            st.metric("Similarity Score", f"{best['score']:.4f}")

        with c2:
            st.subheader("🤖 Qwen2-VL Description:")
            with st.spinner("Generating explanation..."):
                # Mengirim label sebagai konteks agar deskripsi lebih akurat
                desc = rag.generate_description(best['path'], best['label'])
            st.success(desc)

        if len(results) > 1:
            st.divider()
            st.write("### Similar Images:")
            cols = st.columns(len(results) - 1)
            for i, res in enumerate(results[1:]):
                with cols[i]:
                    # PERBAIKAN: Menggunakan width='stretch' di galeri juga
                    st.image(res['path'], width="stretch")
                    st.caption(f"**{res['label']}**\n({res['score']:.2f})")
    else:
        st.warning("No results found.")


# 4. NAVIGASI TABS & INPUT

tab1, tab2 = st.tabs(["🔤 Search by Text", "🖼️ Search by Image"])

with tab1:
    text_query = st.text_input("Describe what you are looking for...",
                               placeholder="e.g., A golden retriever playing in the grass")

    if st.button("Search Text", key="btn_txt"):
        if text_query:
            results = rag.search(text_query)
            display_results(results)

with tab2:
    uploaded = st.file_uploader("Upload reference image", type=['jpg', 'png', 'jpeg'])
    if uploaded:
        st.image(uploaded, width=250, caption="Your Input")
        if st.button("Search Similar Images", key="btn_img"):
            # Save temp file untuk diproses backend
            os.makedirs("save", exist_ok=True)
            temp_path = os.path.join("save", "temp_query.jpg")
            with open(temp_path, "wb") as f:
                f.write(uploaded.getbuffer())

            results = rag.search(temp_path)
            display_results(results)

st.divider()
st.markdown("<center><small>Final Project RAG System</small></center>", unsafe_allow_html=True)