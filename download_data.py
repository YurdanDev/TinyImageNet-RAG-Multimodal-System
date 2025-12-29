import kagglehub
import shutil
import os
import config
from tqdm import tqdm


def setup_dataset():
    print("🚀 Memulai Download Dataset dari Kaggle...")

    # 1. DOWNLOAD VIA KAGGLEHUB
    path = kagglehub.dataset_download("akash2sharma/tiny-imagenet")
    print(f"✅ Download Selesai di: {path}")

    # Path sumber (biasanya ada subfolder tiny-imagenet-200)
    source_root = os.path.join(path, "tiny-imagenet-200")
    if not os.path.exists(source_root):
        source_root = path  # Jaga-jaga jika strukturnya langsung

    # 2. PINDAHKAN KE FOLDER PROYEK (Dataset/)
    print("📂 Menata ulang struktur folder...")

    # Mapping folder sumber -> target (Sesuai config.py)
    # Kaggle pakai huruf kecil ('train'), Config kita pakai huruf besar ('Train')
    moves = {
        "train": config.TRAIN_DIR,
        "val": config.VAL_DIR,
        "words.txt": config.WORDS_FILE
    }

    for src_name, target_dir in moves.items():
        src_path = os.path.join(source_root, src_name)

        # Hapus target lama jika ada (biar bersih)
        if os.path.exists(target_dir):
            if os.path.isdir(target_dir):
                shutil.rmtree(target_dir)
            else:
                os.remove(target_dir)

        # Pindahkan/Copy
        if os.path.exists(src_path):
            print(f"   -> Memindahkan {src_name} ke {target_dir}")
            shutil.move(src_path, target_dir)
        else:
            print(f"⚠️ Warning: {src_name} tidak ditemukan di source.")

    # 3. RESTRUKTURISASI DATA VALIDASI

    val_img_dir = os.path.join(config.VAL_DIR, "images")
    val_annot_file = os.path.join(config.VAL_DIR, "val_annotations.txt")

    if os.path.exists(val_img_dir) and os.path.exists(val_annot_file):
        print("🔨 Melakukan Restrukturisasi Folder Validasi (Flat -> Class Folders)...")

        # Baca file anotasi (Format: nama_file.jpg \t id_kelas \t ...)
        with open(val_annot_file, 'r') as f:
            lines = f.readlines()

        for line in tqdm(lines, desc="Sorting Val Images"):
            parts = line.strip().split('\t')
            if len(parts) < 2: continue

            img_name = parts[0]
            class_id = parts[1]

            # Buat folder kelas di dalam VAL_DIR (misal: Dataset/Val/n01443537)
            class_dir = os.path.join(config.VAL_DIR, class_id)
            os.makedirs(class_dir, exist_ok=True)

            # Pindahkan gambar dari folder 'images' ke folder kelas
            src_img = os.path.join(val_img_dir, img_name)
            dst_img = os.path.join(class_dir, img_name)

            if os.path.exists(src_img):
                shutil.move(src_img, dst_img)

        # Hapus folder 'images' kosong dan file anotasi (opsional, biar rapi)
        shutil.rmtree(val_img_dir)
        # os.remove(val_annot_file) # Simpan anotasi jika perlu debug
        print("✅ Struktur Validasi Selesai Diperbaiki.")

    print("\n🎉 Dataset Siap Digunakan!")
    print(f"   Train: {config.TRAIN_DIR}")
    print(f"   Val  : {config.VAL_DIR}")


if __name__ == "__main__":
    setup_dataset()