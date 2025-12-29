import time
import os
import random
import numpy as np
from tqdm import tqdm
import config
from backend import RAGSystem
import gc
import torch

# 1. KONFIGURASI EVALUASI

# Jika pakai Colab Pro, silakan kembalikan ke 10000
SAMPLE_SIZE_RETRIEVAL = 1000

# Dibatasi agar evaluasi tidak memakan waktu berjam-jam (karena Qwen berat).
SAMPLE_SIZE_GENERATIVE = 20

# Jumlah dokumen/gambar teratas yang diambil saat retrieval
TOP_K = 5


# 2. FUNGSI UTILITAS (HELPER)

def load_tiny_imagenet_mapping(root_dir):
    """
    Memuat file metadata 'words.txt' dari TinyImageNet.
    """
    mapping = {}
    possible_paths = [
        os.path.join(os.path.dirname(config.VAL_DIR), 'words.txt'),
        os.path.join(os.path.dirname(os.path.dirname(config.VAL_DIR)), 'words.txt'),
        './words.txt',
        '/content/tiny-imagenet-200/words.txt',
        '/content/Dataset/words.txt'
    ]

    found_path = None
    for p in possible_paths:
        if os.path.exists(p):
            found_path = p
            break

    if found_path:
        print(f"📂 Loading Label Mapping from: {found_path}")
        try:
            with open(found_path, 'r') as f:
                for line in f:
                    parts = line.strip().split('\t')
                    if len(parts) >= 2:
                        class_id = parts[0]
                        human_name = parts[1].split(',')[0].strip().lower()
                        mapping[class_id] = human_name
        except:
            pass
    else:
        print("⚠️ WARNING: File 'words.txt' tidak ditemukan!")

    return mapping


# 3. METRIK EVALUASI

def calculate_mrr(results, target_class_id):
    for i, res in enumerate(results):
        if target_class_id in res['path']:
            return 1.0 / (i + 1)
    return 0.0


def calculate_lir(description, target_names):
    if not description or not target_names:
        return 0
    desc_clean = description.lower()
    if isinstance(target_names, str):
        return 1 if target_names in desc_clean else 0
    return 0


# 4. PROGRAM UTAMA (MAIN LOOP)

def run_evaluation():
    # Bersihkan memori dulu
    gc.collect()
    torch.cuda.empty_cache()

    print("\n" + "=" * 50)
    print("🚀 MEMULAI ACADEMIC EVALUATION (FINAL)")
    print("=" * 50)

    # A. Inisialisasi Sistem & Mapping
    try:
        rag = RAGSystem()
        id_to_name = load_tiny_imagenet_mapping(config.VAL_DIR)
    except Exception as e:
        print(f"❌ Error Initialization: {e}")
        return

    # B. Persiapan Data (Scanning Dataset) - BAGIAN INI DIPERBAIKI
    val_samples = []
    print(f"📂 Scanning Folder Validasi: {config.VAL_DIR}")

    if os.path.exists(config.VAL_DIR):
        # Ambil hanya folder (hindari file nyasar seperti .DS_Store)
        classes = [d for d in os.listdir(config.VAL_DIR) if os.path.isdir(os.path.join(config.VAL_DIR, d))]

        for class_id in tqdm(classes, desc="Indexing Classes"):
            class_path = os.path.join(config.VAL_DIR, class_id)

            # Support .jpg, .png, DAN .jpeg (penting!)
            images = [f for f in os.listdir(class_path)
                      if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

            for img in images:
                val_samples.append({
                    'path': os.path.join(class_path, img),
                    'class_id': class_id
                })

    print(f"📊 Total Data Validasi Ditemukan: {len(val_samples)} gambar.")

    if not val_samples:
        print("❌ Data validasi kosong. Pastikan path benar.")
        return

    # C. Random Sampling
    real_sample_count = min(len(val_samples), SAMPLE_SIZE_RETRIEVAL)
    test_set = random.sample(val_samples, real_sample_count)
    print(f"🧪 Melakukan pengujian pada {len(test_set)} sampel acak.")

    # D. Container Metrik
    metrics = {
        'top1_hits': 0, 'top5_hits': 0,
        'mrr_sum': 0.0, 'precision_sum': 0.0,
        'lir_hits': 0
    }

    # E. Loop Evaluasi
    debug_print_count = 0

    for i, sample in enumerate(tqdm(test_set, desc="Benchmarking")):

        # --- PHASE 1: RETRIEVAL ---
        results = rag.search(sample['path'], top_k=TOP_K)

        if not results: continue

        target_id = sample['class_id']

        # Hitung Metrics Retrieval
        if target_id in results[0]['path']:
            metrics['top1_hits'] += 1

        relevant_count = 0
        found_in_top5 = False
        for res in results:
            if target_id in res['path']:
                found_in_top5 = True
                relevant_count += 1

        if found_in_top5: metrics['top5_hits'] += 1
        metrics['precision_sum'] += (relevant_count / TOP_K)
        metrics['mrr_sum'] += calculate_mrr(results, target_id)

        # --- PHASE 2: GENERATIVE (Subset Only) ---
        if i < SAMPLE_SIZE_GENERATIVE:
            try:
                target_human_name = id_to_name.get(target_id, target_id)
                desc = rag.generate_description(results[0]['path'], target_human_name)

                is_hit = calculate_lir(desc, target_human_name)
                metrics['lir_hits'] += is_hit

                # Debug Print
                if debug_print_count < 100:
                    print("\n" + "-" * 30)
                    print(f"🔍 DEBUG SAMPLE #{debug_print_count + 1}")
                    print(f"   ID Folder   : {target_id}")
                    print(f"   Target Name : {target_human_name}")
                    print(f"   Qwen Output : {desc[:100]}...")  # Limit text
                    print(f"   LIR Status  : {'✅ HIT' if is_hit else '❌ MISS'}")
                    print("-" * 30)
                    debug_print_count += 1

                # Bersihkan cache GPU agar tidak OOM
                torch.cuda.empty_cache()

            except Exception as e:
                pass

                # F. Laporan Akhir
    n_ret = len(test_set)
    # Hindari pembagian dengan nol jika loop generative gagal total
    n_gen = max(debug_print_count, 1)

    print("\n╔════════════ FINAL REPORT ════════════╗")
    print(f"║ Recall@1    : {(metrics['top1_hits'] / n_ret) * 100:6.2f} %           ║")
    print(f"║ Recall@5    : {(metrics['top5_hits'] / n_ret) * 100:6.2f} %           ║")
    print(f"║ Precision@5 : {(metrics['precision_sum'] / n_ret) * 100:6.2f} %           ║")
    print(f"║ MRR         : {metrics['mrr_sum'] / n_ret:6.4f}              ║")
    print("╠══════════════════════════════════════╣")
    print(f"║ LIR (Generative): {(metrics['lir_hits'] / n_gen) * 100:6.2f} %        ║")
    print("╚══════════════════════════════════════╝")


if __name__ == "__main__":
    run_evaluation()