# 🖼️ TinyImageNet RAG Multimodal System

Sistem **Retrieval-Augmented Generation (RAG) Multimodal** untuk pencarian dan deskripsi citra secara cerdas menggunakan dataset **TinyImageNet-200**.

Sistem ini mengintegrasikan:

* **CLIP** untuk *visual & text retrieval* berbasis embedding,
* **FAISS** sebagai *vector database* berperforma tinggi,
* **Qwen2-VL** (*Vision-Language Model*) untuk menghasilkan deskripsi gambar yang kontekstual dan akurat.

Pendekatan RAG memungkinkan model generatif menerima **konteks hasil retrieval** (label/metadata) sehingga kualitas deskripsi menjadi lebih presisi dan relevan.

---

## 📊 Dataset

Dataset yang digunakan adalah **TinyImageNet-200**, terdiri dari 200 kelas dengan resolusi gambar 64×64.

🔗 **Sumber Dataset (Kaggle):**
[https://www.kaggle.com/datasets/akash2sharma/tiny-imagenet](https://www.kaggle.com/datasets/akash2sharma/tiny-imagenet)

<p align="center">
  <img src="https://github.com/user-attachments/assets/44a2a49e-3954-4892-a639-2a2e7c57f516" width="85%" />
</p>

---

## 🧠 Metodologi & Pipeline RAG

Pipeline sistem mengikuti alur **Multimodal RAG** sebagai berikut:

1. **Input Query**

   * Query teks (Text-to-Image Search)
   * Query gambar (Image-to-Image Search)
2. **Embedding**

   * Query dan gambar dienkode menggunakan **CLIP**
3. **Retrieval**

   * Pencarian vektor dilakukan menggunakan **FAISS**
4. **Augmentation**

   * Label dan metadata hasil retrieval dijadikan konteks
5. **Generation**

   * **Qwen2-VL** menghasilkan deskripsi citra berbasis konteks (RAG)

<p align="center">
  <img src="https://github.com/user-attachments/assets/e984b1a5-b043-4d63-8047-552a5c3d2f92" width="85%" />
</p>

---

## 👥 Anggota Kelompok

| Nama                             | NIM             |
| -------------------------------- | --------------- |
| **Naza Nadhana Afdha**           | 202110370311522 |
| **Yashinta Indrastuti**          | 202110370311502 |
| **Muhammad Yurdan Asy Shadzili** | 202110370311455 |

---

## 🚀 Fitur Utama

### 1️⃣ Multimodal Retrieval (Pencarian Cerdas)

* 🔍 Text-to-Image Search (contoh: *"cari kucing mesir"*)
* 🖼️ Image-to-Image Search
* ⚡ Menggunakan **FAISS** untuk pencarian vektor skala besar

### 2️⃣ Generative Description (AI Explanation)

* 🤖 Deskripsi gambar otomatis menggunakan **Qwen2-VL**
* 🧩 Menggunakan konteks label hasil retrieval (RAG-based)
* 📄 Output berupa narasi visual yang detail dan kontekstual

### 3️⃣ Evaluasi Sistem

* 📈 Evaluasi otomatis menggunakan metrik:

  * **Recall@1**
  * **Recall@5**
  * **MRR (Mean Reciprocal Rank)**
  * **LIR (Label Inclusion Rate)**

---

## 🛠️ Teknologi yang Digunakan

* **Bahasa Pemrograman** : Python 3.10+
* **Embedding Model** : CLIP (Sentence-Transformers)
* **Vector Database** : FAISS
* **Vision-Language Model** : Qwen2-VL-2B-Instruct
* **Framework** : PyTorch, Hugging Face Transformers
* **Dataset** : TinyImageNet-200
* **Web Interface** : Streamlit

---

## 📂 Struktur Folder Proyek

```text
TinyImageNet-RAG-Multimodal-System/
│
├── app.py                 # Antarmuka Web (Streamlit)
├── backend.py             # Logika inti RAG (retrieval & generation)
├── config.py              # Konfigurasi path & parameter model
├── indexer.py             # Pembuatan index FAISS dari dataset
├── evaluation.py          # Evaluasi performa (Recall, MRR, LIR)
├── download_data.py       # Download dataset TinyImageNet otomatis
├── fix_train.py           # Restrukturisasi data training
├── fix_val.py             # Restrukturisasi data validasi
├── check_data.py          # Validasi integritas dataset
│
├── Dataset/               # Dataset TinyImageNet
└── vector_db/             # Penyimpanan FAISS index & metadata
```

---

## ⚙️ Instalasi Dependensi

```bash
pip install torch torchvision transformers sentence-transformers \
            faiss-cpu pillow requests tqdm numpy \
            accelerate qwen-vl-utils streamlit
```

---

## ▶️ Cara Menjalankan Sistem

### Langkah 1 — Konfigurasi Sistem

* **File**: `config.py`
* Pastikan path dataset dan model sudah benar
* Tidak perlu dijalankan, hanya diverifikasi

---

### Langkah 2 — Akuisisi Dataset

Mengunduh dan mengekstrak dataset TinyImageNet-200

```bash
python download_data.py
```

---

### Langkah 3 — Restrukturisasi Data Training

Menyesuaikan struktur folder training agar kompatibel dengan `ImageFolder`

```bash
python fix_train.py
```

---

### Langkah 4 — Restrukturisasi Data Validasi

Mengelompokkan data validasi ke dalam folder kelas (krusial untuk evaluasi)

```bash
python fix_val.py
```

---

### Langkah 5 — Verifikasi Integritas Data

Memastikan struktur dan jumlah data sudah valid

```bash
python check_data.py
```

---

### Langkah 6 — Inisialisasi Backend

* **File**: `backend.py`
* Berisi class utama `RAGSystem`
* Digunakan oleh aplikasi dan modul evaluasi

---

### Langkah 7 — Indexing (Vektorisasi Dataset)

Mengonversi seluruh gambar menjadi embedding CLIP dan menyimpannya ke FAISS

```bash
python indexer.py
```

**Output:**

```text
vector_db/tiny_imagenet_rag_index.bin
```
```text
vector_db/tiny_imagenet_rag_metadata.json
```

---

### Langkah 8 — Setup Model Vision-Language (LLM)

Menyiapkan model **Qwen2-VL** untuk inferensi

```bash
python setup_models_LLM.py
```

---

### Langkah 9 — Menjalankan Aplikasi Web

Menjalankan antarmuka Streamlit untuk demo interaktif

```bash
streamlit run app.py
```

---

### Langkah 10 — Evaluasi

Menghitung performa sistem menggunakan metrik retrieval

```bash
python evaluation.py
```

---

## 📌 Catatan

* Sistem dirancang untuk **Final Project Temu Kembali Citra dan pembelajaran multimodal RAG**
* Dapat dikembangkan untuk dataset skala besar atau domain lain
* Cocok sebagai dasar pengembangan **Multimodal Search Engine**

---

✨ *TinyImageNet RAG Multimodal System — Bridging Vision & Language with Retrieval-Augmented Intelligence*


