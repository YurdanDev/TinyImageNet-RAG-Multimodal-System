import os
import shutil
from tqdm import tqdm
import config


def flatten_train_structure():
    print(f"🔧 Memperbaiki struktur folder Training di: {config.TRAIN_DIR}")

    if not os.path.exists(config.TRAIN_DIR):
        print("❌ Folder Train tidak ditemukan!")
        return

    # Ambil semua folder kelas (n01443537, dst)
    classes = [d for d in os.listdir(config.TRAIN_DIR) if os.path.isdir(os.path.join(config.TRAIN_DIR, d))]

    count_moved = 0

    for cls in tqdm(classes, desc="Merapikan Folder"):
        class_path = os.path.join(config.TRAIN_DIR, cls)
        images_subdir = os.path.join(class_path, 'images')

        # Cek apakah ada subfolder 'images'
        if os.path.exists(images_subdir) and os.path.isdir(images_subdir):
            # Pindahkan semua file di dalamnya naik satu level
            for filename in os.listdir(images_subdir):
                src = os.path.join(images_subdir, filename)
                dst = os.path.join(class_path, filename)

                shutil.move(src, dst)
                count_moved += 1

            # Hapus folder 'images' yang sudah kosong
            os.rmdir(images_subdir)

    print(f"\n✅ Selesai! {count_moved} gambar berhasil dipindahkan.")
    print("Sekarang struktur folder Train sudah benar (Flat).")


if __name__ == "__main__":
    flatten_train_structure()