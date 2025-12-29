import os
import shutil
from tqdm import tqdm
import config


def fix_validation_structure():
    print("🔧 MEMULAI PERBAIKAN STRUKTUR VALIDASI...")

    val_dir = config.VAL_DIR
    images_dir = os.path.join(val_dir, "images")
    annot_file = os.path.join(val_dir, "val_annotations.txt")

    # 1. Cek apakah file anotasi ada
    if not os.path.exists(annot_file):
        # Coba cari di parent directory jika tidak ada di dalam Val
        parent_annot = os.path.join(os.path.dirname(val_dir), "val_annotations.txt")
        if os.path.exists(parent_annot):
            shutil.copy(parent_annot, annot_file)
            print(f"   -> File anotasi disalin dari: {parent_annot}")
        else:
            print("❌ ERROR: File 'val_annotations.txt' tidak ditemukan!")
            print("   Pastikan dataset TinyImageNet ter-download dengan lengkap.")
            return

    # 2. Cek apakah folder 'images' ada (Format Kaggle Raw)
    # Jika folder 'images' tidak ada, mungkin gambar ada langsung di root Val, atau sudah tertata.
    source_dir = images_dir if os.path.exists(images_dir) else val_dir

    print(f"📂 Membaca anotasi dari: {annot_file}")

    # Mapping: Nama File -> ID Kelas
    img_to_class = {}
    with open(annot_file, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                img_to_class[parts[0]] = parts[1]

    print(f"   -> Ditemukan {len(img_to_class)} data anotasi.")

    # 3. Pindahkan gambar ke folder kelas masing-masing
    moved_count = 0
    missing_count = 0

    # List semua file gambar di source_dir
    all_files = [f for f in os.listdir(source_dir) if f.lower().endswith('.jpg')]

    if len(all_files) == 0:
        print("⚠️ Folder Validasi terlihat kosong atau sudah tertata. Mengecek subfolder...")
        # Cek apakah sudah tertata (ada folder n0xxxx)
        subdirs = [d for d in os.listdir(val_dir) if os.path.isdir(os.path.join(val_dir, d)) and d.startswith('n')]
        if len(subdirs) > 0:
            print(f"✅ Struktur Validasi Tampaknya SUDAH BENAR ({len(subdirs)} kelas terdeteksi).")
            return
        else:
            print("❌ Tidak ada gambar jpg ditemukan di root Val maupun folder images.")
            return

    print("🚀 Memindahkan gambar ke folder kelas...")
    for filename in tqdm(all_files):
        if filename in img_to_class:
            class_id = img_to_class[filename]

            # Buat folder kelas (misal: Dataset/Val/n01443537)
            target_folder = os.path.join(val_dir, class_id)
            os.makedirs(target_folder, exist_ok=True)

            # Pindahkan file
            src_path = os.path.join(source_dir, filename)
            dst_path = os.path.join(target_folder, filename)

            try:
                shutil.move(src_path, dst_path)
                moved_count += 1
            except Exception as e:
                print(f"   Gagal pindah {filename}: {e}")
        else:
            missing_count += 1

    # 4. Menghapus
    if os.path.exists(images_dir) and not os.listdir(images_dir):
        os.rmdir(images_dir)
        print("🗑️  Folder 'images' kosong telah dihapus.")

    print("\n✅ PERBAIKAN SELESAI!")
    print(f"   - Gambar dipindahkan : {moved_count}")
    print(f"   - Gambar tanpa label : {missing_count}")
    print(f"   - Lokasi Validasi    : {val_dir}")


if __name__ == "__main__":
    fix_validation_structure()