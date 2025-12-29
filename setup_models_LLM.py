import os
import logging
import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from transformers import logging as hf_logging

# Import modul konfigurasi lokal
import config

# Konfigurasi logging untuk menekan pesan warning yang tidak kritikal
hf_logging.set_verbosity_error()

def download_qwen():
    """
    Fungsi utilitas untuk mengunduh model Qwen2-VL dan menyimpannya ke cache lokal.

    Proses ini dioptimalkan untuk efisiensi memori (RAM) dengan tidak memuat
    model ke dalam VRAM GPU selama proses pengunduhan berlangsung.
    """
    print(f"📂 Direktori Cache Target: {config.MODELS_CACHE_DIR}")
    print("🚀 Memulai Proses Download Qwen2-VL...")

    # Memastikan direktori cache tersedia
    os.makedirs(config.MODELS_CACHE_DIR, exist_ok=True)

    try:
        # ---------------------------------------------------------
        # Tahap 1: Mengunduh Processor
        # ---------------------------------------------------------
        print("⏳ [1/2] Mengunduh Processor (Image & Text Handler)...")
        AutoProcessor.from_pretrained(
            config.VLM_MODEL_NAME,
            cache_dir=config.MODELS_CACHE_DIR,
            use_fast=True
        )

        # ---------------------------------------------------------
        # Tahap 2: Mengunduh Bobot Model (Weights)
        # ---------------------------------------------------------
        print("⏳ [2/2] Mengunduh Bobot Model (File Besar)...")

        # Pemuatan model hanya ke CPU/Disk untuk caching.
        # device_map="auto" dihilangkan untuk mencegah alokasi VRAM GPU
        # saat hanya bertujuan untuk mengunduh file.
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            config.VLM_MODEL_NAME,
            torch_dtype=torch.float16,
            cache_dir=config.MODELS_CACHE_DIR,
            low_cpu_mem_usage=True,
        )

        # Mekanisme pembersihan memori setelah download selesai
        del model
        torch.cuda.empty_cache()

        print("\n✅ SUKSES! Model Qwen2-VL berhasil didownload.")
        print(f"📍 Tersimpan di: {config.MODELS_CACHE_DIR}")

    except Exception as e:
        print(f"\n❌ TERJADI KESALAHAN SAAT DOWNLOAD: {e}")
        print("   Saran: Periksa koneksi internet atau kapasitas penyimpanan.")

if __name__ == "__main__":
    download_qwen()