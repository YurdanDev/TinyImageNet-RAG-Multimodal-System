# TinyImageNet RAG Multimodal System

Sistem **Retrieval-Augmented Generation (RAG) Multimodal** yang dibangun untuk melakukan pencarian dan deskripsi citra secara cerdas menggunakan dataset **TinyImageNet**.

Sistem ini menggabungkan kekuatan **CLIP** untuk pencarian visual (retrieval) dan **Qwen2-VL** (Vision-Language Model) untuk menghasilkan deskripsi tekstual yang akurat berdasarkan konteks gambar yang ditemukan.


dataset <img width="1639" height="912" alt="image" src="https://github.com/user-attachments/assets/e984b1a5-b043-4d63-8047-552a5c3d2f92" />

metodology & pipeline RAG<img width="1658" height="921" alt="image" src="https://github.com/user-attachments/assets/44a2a49e-3954-4892-a639-2a2e7c57f516" />

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



pip install torch torchvision transformers sentence-transformers faiss-cpu pillow requests tqdm numpy accelerate qwen-vl-utils streamlit

Langkah 1: Konfigurasi Sistem
Memastikan parameter path dan model sudah sesuai.

File: config.py

Tindakan: Periksa file ini untuk memastikan path dataset dan konfigurasi model sudah benar. Tidak perlu dijalankan, hanya diverifikasi.

Langkah 2: Akuisisi Data
Mengunduh dataset TinyImageNet-200 dan mengekstraknya ke folder proyek.

Bash

python download_data.py
Langkah 3: Restrukturisasi Data Training
Memperbaiki struktur folder training agar sesuai dengan format standar ImageFolder.

Bash

python fix_train.py
Langkah 4: Restrukturisasi Data Validasi
Mengelompokkan gambar validasi (val) ke dalam sub-folder kelas masing-masing (sangat krusial untuk evaluasi akurasi).

Bash

python fix_val.py
Langkah 5: Verifikasi Integritas Data
Melakukan pengecekan akhir untuk memastikan jumlah gambar dan struktur folder sudah valid sebelum diproses lebih lanjut.

Bash

python check_data.py
Langkah 6: Inisialisasi Backend
File: backend.py

Keterangan: File ini berisi logika inti (Class RAGSystem). Pastikan file ini ada di root folder karena akan dipanggil oleh modul lain.

Langkah 7: Indexing (Vektorisasi)
Mengubah ribuan gambar dataset menjadi vektor embedding menggunakan model CLIP dan menyimpannya ke dalam FAISS Vector Database.

Bash

python indexer.py
> Output: File vector_db/tiny_imagenet_rag_index.bin akan terbentuk.

Langkah 8: Setup Model LLM
Mengunduh atau memverifikasi kesiapan model Qwen2-VL (Vision Language Model) agar siap digunakan untuk inferensi.

Bash

python setup_models_LLM.py
Langkah 9: Menjalankan Aplikasi (Interface)
Menjalankan antarmuka web berbasis Streamlit untuk demonstrasi pencarian dan deskripsi gambar secara interaktif.

Bash

streamlit run app.py
Langkah 10: Evaluasi Akademik
Menjalankan skrip pengujian otomatis untuk menghitung metrik performa: Recall@1, Recall@5, MRR, dan LIR.

Bash

python evaluation.py

