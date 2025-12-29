import os
import sys
import json
import logging
from transformers import logging as hf_logging

# Mengatur environment variable untuk menghindari deadlock pada tokenizer
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Import library pihak ketiga
import torch
import faiss
import numpy as np
from PIL import Image
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

# Import konfigurasi lokal
import config

# Konfigurasi Logging HuggingFace
hf_logging.set_verbosity_error()

# Konstanta
BATCH_SIZE = 64

# 1. FUNGSI UTILITAS DATASET

def load_class_mapping(txt_path):
    """
    Membaca file metadata (words.txt) untuk memetakan ID Kelas ke Nama Label.

    Args:
        txt_path (str): Path lokasi file words.txt.

    Returns:
        dict: Mapping {class_id: human_readable_label}.
    """
    print(f"📖 Membaca mapping kelas dari: {txt_path}")
    mapping = {}

    if not os.path.exists(txt_path):
        print(f"⚠️ Peringatan: File {txt_path} tidak ditemukan.")
        return mapping

    try:
        with open(txt_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    # parts[0] = id (n01443537), parts[1] = nama (goldfish, ...)
                    mapping[parts[0]] = parts[1]
    except Exception as e:
        print(f"❌ Error membaca mapping: {e}")

    return mapping

def get_image_paths(root_dir, split_name="Data"):
    """
    Memindai direktori secara rekursif untuk mendapatkan path semua gambar.

    Args:
        root_dir (str): Folder root dataset (misal: Train/ atau Val/).
        split_name (str): Nama split untuk log (opsional).

    Returns:
        list: Daftar tuple [(path_gambar, class_id), ...].
    """
    print(f"🔍 Memindai gambar di folder {split_name} ({root_dir})...")
    image_paths = []
    valid_exts = ('.jpg', '.jpeg', '.png')

    if not os.path.exists(root_dir):
        print(f"⚠️ Folder {root_dir} tidak ditemukan.")
        return []

    # Walk melalui struktur direktori folder_kelas/gambar.jpg
    for class_folder in os.listdir(root_dir):
        class_path = os.path.join(root_dir, class_folder)

        # Pastikan yang dibaca adalah direktori
        if not os.path.isdir(class_path):
            continue

        for root, _, files in os.walk(class_path):
            for file in files:
                if file.lower().endswith(valid_exts):
                    full_path = os.path.join(root, file)
                    image_paths.append((full_path, class_folder))

    print(f"   ✅ Ditemukan {len(image_paths)} gambar di {split_name}.")
    return image_paths

# 2. PROSES UTAMA (INDEXING)

def main():
    print(f"🚀 Memulai proses indexing pada device: {config.DEVICE}")

    # A. Inisialisasi Model Embedding (CLIP)
    try:
        model = SentenceTransformer(
            config.CLIP_MODEL_NAME,
            device=config.DEVICE,
            cache_folder=config.MODELS_CACHE_DIR
        )
    except Exception as e:
        print(f"❌ Gagal memuat model: {e}")
        return

    # B. Persiapan Data
    class_map = load_class_mapping(config.WORDS_FILE)

    # Gabungkan data Train dan Validation
    train_imgs = get_image_paths(config.TRAIN_DIR, "Train")
    val_imgs = get_image_paths(config.VAL_DIR, "Validation")
    all_images = train_imgs + val_imgs

    print(f"Σ  Total Semua Gambar: {len(all_images)}")
    if not all_images:
        print("❌ Tidak ada gambar untuk diproses.")
        return

    embeddings = []
    metadata = []

    # C. Batch Processing (Encoding)
    print("⚙️  Memproses Embedding (Batch Processing)...")

    # Loop dengan step sebesar BATCH_SIZE
    for i in tqdm(range(0, len(all_images), BATCH_SIZE), desc="Indexing"):
        batch_files = all_images[i : i + BATCH_SIZE]
        batch_images = []
        batch_meta = []

        # Load gambar dalam batch
        for img_path, class_id in batch_files:
            try:
                # Convert RGB penting untuk menangani gambar grayscale/RGBA
                img = Image.open(img_path).convert('RGB')
                batch_images.append(img)

                # Simpan metadata terkait
                batch_meta.append({
                    "path": img_path,
                    "class_id": class_id,
                    "label": class_map.get(class_id, class_id)
                })
            except Exception as e:
                # Skip gambar corrupt
                continue

        # Jika batch memiliki gambar valid, lakukan encoding
        if batch_images:
            with torch.no_grad():
                # Encode gambar menjadi vektor
                batch_emb = model.encode(
                    batch_images,
                    convert_to_numpy=True,
                    show_progress_bar=False
                )

            # Normalisasi L2 untuk pencarian berbasis Cosine Similarity
            faiss.normalize_L2(batch_emb)

            embeddings.append(batch_emb)
            metadata.extend(batch_meta)

    # D. Penyimpanan Index FAISS & Metadata
    if embeddings:
        final_embeddings = np.vstack(embeddings)
        print(f"📊 Dimensi Matrix Akhir: {final_embeddings.shape}")

        # Membuat Index Flat Inner Product (Cocok untuk normalized vectors)
        d = final_embeddings.shape[1]
        index = faiss.IndexFlatIP(d)
        index.add(final_embeddings)

        # Pastikan direktori output tersedia
        output_folder = os.path.dirname(config.INDEX_FILE)
        if output_folder and not os.path.exists(output_folder):
            os.makedirs(output_folder, exist_ok=True)
            print(f"📁 Membuat folder output: {output_folder}")

        print(f"💾 Menyimpan index vektor ke: {config.INDEX_FILE}")
        faiss.write_index(index, config.INDEX_FILE)

        print(f"📝 Menyimpan metadata JSON ke: {config.METADATA_FILE}")
        with open(config.METADATA_FILE, 'w') as f:
            json.dump(metadata, f, indent=4)

        print("\n🎉 SUKSES! Database Vector berhasil dibuat.")

if __name__ == "__main__":
    main()