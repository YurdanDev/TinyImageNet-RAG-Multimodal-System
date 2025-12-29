import os
import json
import logging
import torch
import faiss
import numpy as np
from PIL import Image
from sentence_transformers import SentenceTransformer
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from transformers import logging as hf_logging

# Import konfigurasi lokal
import config

# Konfigurasi Logging: Menekan pesan warning yang tidak kritikal
hf_logging.set_verbosity_error()

class RAGSystem:
    """
    Sistem utama Retrieval-Augmented Generation (RAG) yang menangani:
    1. Inisialisasi Model (CLIP & Qwen2-VL).
    2. Pemuatan Database Vektor (FAISS).
    3. Proses Pencarian (Retrieval).
    4. Proses Generasi Deskripsi (Generation).
    """

    def __init__(self):
        print("🛠️  Inisialisasi RAG System...")

        # ---------------------------------------------------------
        # 1. Validasi Keberadaan Database
        # ---------------------------------------------------------
        if not os.path.exists(config.INDEX_FILE) or not os.path.exists(config.METADATA_FILE):
            raise FileNotFoundError(
                "❌ Database Vector belum ditemukan! Harap jalankan 'indexer.py' terlebih dahulu."
            )

        # ---------------------------------------------------------
        # 2. Memuat Model Retrieval (CLIP)
        # ---------------------------------------------------------
        # Digunakan untuk mengubah query (teks/gambar) menjadi vektor.
        self.clip_model = SentenceTransformer(
            config.CLIP_MODEL_NAME,
            device=config.DEVICE,
            cache_folder=config.MODELS_CACHE_DIR
        )

        # ---------------------------------------------------------
        # 3. Memuat Index FAISS & Metadata
        # ---------------------------------------------------------
        # FAISS untuk pencarian vektor cepat, Metadata untuk info label/path.
        self.index = faiss.read_index(config.INDEX_FILE)

        with open(config.METADATA_FILE, 'r') as f:
            self.metadata = json.load(f)

        # ---------------------------------------------------------
        # 4. Memuat Model Generative (Qwen2-VL)
        # ---------------------------------------------------------
        # Diload terakhir karena memakan VRAM paling besar.
        print("⏳ Loading Qwen2-VL Model (Vision-Language Model)...")

        self.processor = AutoProcessor.from_pretrained(
            config.VLM_MODEL_NAME,
            cache_dir=config.MODELS_CACHE_DIR,
            use_fast=True
        )

        # Menggunakan torch_dtype=float16 untuk efisiensi memori GPU
        self.vlm_model = Qwen2VLForConditionalGeneration.from_pretrained(
            config.VLM_MODEL_NAME,
            torch_dtype=torch.float16,
            device_map="auto",
            cache_dir=config.MODELS_CACHE_DIR
        )

        print("✅ Sistem RAG Siap Digunakan!")

    def search(self, query, top_k=config.TOP_K):
        """
        Melakukan pencarian gambar berdasarkan query teks atau gambar input.

        Args:
            query (str): Path file gambar ATAU string teks pencarian.
            top_k (int): Jumlah hasil teratas yang diambil.

        Returns:
            list: Daftar dictionary berisi path gambar, label, dan skor kemiripan.
        """
        # A. Deteksi Tipe Query
        # Jika query adalah string path file yang valid -> Image Search
        if isinstance(query, str) and os.path.exists(query):
            img = Image.open(query).convert('RGB')
            query_emb = self.clip_model.encode([img], convert_to_numpy=True)
        # Jika query adalah teks biasa -> Text Search
        else:
            query_emb = self.clip_model.encode([query], convert_to_numpy=True)

        # B. Normalisasi & Pencarian Vektor
        faiss.normalize_L2(query_emb)
        scores, indices = self.index.search(query_emb, top_k)

        # C. Format Output
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.metadata):
                item = self.metadata[idx]
                results.append({
                    "path": item['path'],
                    "label": item['label'],
                    "score": float(score)
                })
        return results

    def generate_description(self, image_path, label):
        """
        Menghasilkan deskripsi visual menggunakan model Qwen2-VL.

        Args:
            image_path (str): Lokasi file gambar.
            label (str): Label kelas (sebagai konteks tambahan prompt).

        Returns:
            str: Deskripsi teks yang dihasilkan model.
        """
        try:
            image = Image.open(image_path).convert("RGB")

            # Prompt Engineering: Memberikan konteks kategori untuk hasil lebih akurat
            prompt = f"Describe this image in detail. The image category is '{label}'."

            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": prompt},
                    ],
                }
            ]

            # Preprocessing Input
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = self.processor(
                text=[text],
                images=[image],
                padding=True,
                return_tensors="pt"
            ).to(self.vlm_model.device)

            # Proses Generasi (Inference)
            generated_ids = self.vlm_model.generate(**inputs, max_new_tokens=200)

            # Post-processing Output (Decoding)
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = self.processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )

            return output_text[0]

        except Exception as e:
            return f"Error generating description: {str(e)}"