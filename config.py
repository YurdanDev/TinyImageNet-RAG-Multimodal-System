import os
import torch

# 1. MANAJEMEN DIREKTORI & PATH

# Menentukan root directory secara dinamis
try:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    BASE_DIR = os.getcwd() # Fallback untuk Jupyter/Colab

# --- FOLDER DATASET ---
# Folder ini akan diisi oleh script 'download_data.py' dari Kaggle
DATASET_DIR = os.path.join(BASE_DIR, "Dataset")

TRAIN_DIR = os.path.join(DATASET_DIR, "Train")
VAL_DIR   = os.path.join(DATASET_DIR, "Val")
WORDS_FILE = os.path.join(DATASET_DIR, "words.txt")

# --- FOLDER SISTEM RAG ---
# Tempat menyimpan file vektor (FAISS) dan cache model agar tidak download ulang
VECTOR_DB_DIR = os.path.join(BASE_DIR, "vector_db")
MODELS_CACHE_DIR = os.path.join(BASE_DIR, "models_cache")

# Membuat folder otomatis jika belum ada
target_folders = [DATASET_DIR, TRAIN_DIR, VAL_DIR, VECTOR_DB_DIR, MODELS_CACHE_DIR]
for folder in target_folders:
    os.makedirs(folder, exist_ok=True)

# 2. KONFIGURASI DATABASE VEKTOR

# File database disimpan
INDEX_FILE = os.path.join(VECTOR_DB_DIR, "tiny_imagenet_rag_index.bin")
METADATA_FILE = os.path.join(VECTOR_DB_DIR, "tiny_imagenet_rag_metadata.json")

# 3. KONFIGURASI MODEL AI

# Model Embedding (Pengubah Gambar ke Angka)
CLIP_MODEL_NAME = 'clip-ViT-B-32'

# Model Generatif (Pemberi Deskripsi)
VLM_MODEL_NAME = "Qwen/Qwen2-VL-2B-Instruct"

# Pengaturan Hardware (Otomatis pakai GPU T4 di Colab)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Parameter Pencarian
TOP_K = 5  # Jumlah kemiripan yang ditampilkan

# 4. STATUS LOG

print(f"⚙️  Konfigurasi Sistem Dimuat (Mode: No-Auth).")
print(f"   - Device       : {DEVICE}")
print(f"   - Base Dir     : {BASE_DIR}")
print(f"   - Vector DB    : {VECTOR_DB_DIR}")
print(f"   - Dataset Path : {DATASET_DIR}")