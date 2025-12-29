# TinyImageNet RAG Multimodal System

Sistem **Retrieval-Augmented Generation (RAG) Multimodal** yang dibangun untuk melakukan pencarian dan deskripsi citra secara cerdas menggunakan dataset **TinyImageNet**.

Sistem ini menggabungkan kekuatan **CLIP** untuk pencarian visual (retrieval) dan **Qwen2-VL** (Vision-Language Model) untuk menghasilkan deskripsi tekstual yang akurat berdasarkan konteks gambar yang ditemukan.

---

## 👥 Anggota Kelompok

| Nama | NIM |
| :--- | :--- |
| **Naza Nadhana Afdha** | 202110370311522 |
| **Yashinta Indrastuti** | 202110370311502 |
| **Muhammad Yurdan Asy Shadzili** | 202110370311455 |

---

## 🚀 Fitur Utama

1.  **Multimodal Retrieval (Pencarian Cerdas)**
    * Mencari gambar menggunakan query **Teks** ("cari kucing mesir").
    * Mencari gambar menggunakan query **Gambar** (Image-to-Image Search).
    * Menggunakan **FAISS** untuk pencarian vektor kecepatan tinggi.

2.  **Generative Description (AI Explanation)**
    * Menggunakan **Qwen2-VL** untuk melihat dan mendeskripsikan gambar hasil pencarian secara detail.
    * Menerapkan teknik RAG dimana model diberikan konteks label asli untuk meningkatkan akurasi deskripsi.

3.  **Evaluasi**
    * Modul evaluasi otomatis untuk menghitung metrik: **Recall@1**, **Recall@5**, **MRR** (Mean Reciprocal Rank), dan **LIR** (Label Inclusion Rate).

---

## 🛠️ Teknologi yang Digunakan

* **Bahasa Pemrograman:** Python 3.10+
* **Vector Database:** FAISS (Facebook AI Similarity Search)
* **Embedding Model:** CLIP (Sentence-Transformers)
* **Generative Model (VLM):** Qwen2-VL-2B-Instruct (Hugging Face)
* **Framework:** PyTorch, Transformers
* **Dataset:** TinyImageNet-200

---

## 📂 Struktur Folder

```text
TinyImageNet-RAG-Multimodal-System/
├── app.py                  # Antarmuka Web (Streamlit/Gradio)
├── backend.py              # Logika Utama RAG (Load Model & Search)
├── config.py               # Konfigurasi Path & Parameter
├── evaluation.py           # Script Pengujian Akurasi (Recall & MRR)
├── indexer.py              # Script untuk membuat index FAISS dari Dataset
├── download_data.py        # Script download dataset otomatis
├── Dataset/                # Folder penyimpanan gambar TinyImageNet
└── vector_db/              # Folder penyimpanan hasil indexing (FAISS .bin, .json)






