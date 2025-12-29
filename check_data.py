import os
import random
import matplotlib.pyplot as plt
from PIL import Image
import config


def load_class_names():
    """Memuat mapping dari ID Kelas (n01443537) ke Nama (Goldfish) """
    mapping = {}
    if os.path.exists(config.WORDS_FILE):
        with open(config.WORDS_FILE, 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    mapping[parts[0]] = parts[1]
    return mapping


def visualize_samples(base_dir, split_name="Train", num_samples=5):
    """
    Mengambil sampel acak dari dataset dan menampilkannya beserta metadata.
    """
    print(f"\n🔍 Memeriksa Sampel Data: {split_name} Set")

    if not os.path.exists(base_dir):
        print(f"❌ Direktori {base_dir} tidak ditemukan!")
        return

    # 1. Ambil daftar semua kelas
    classes = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    if not classes:
        print("❌ Tidak ada folder kelas ditemukan.")
        return

    print(f"   📊 Total Kelas ditemukan: {len(classes)}")

    # 2. Setup Plotting
    fig, axes = plt.subplots(1, num_samples, figsize=(15, 4))
    if num_samples == 1: axes = [axes]  # Handle jika cuma 1 sampel

    class_mapping = load_class_names()

    # 3. Ambil Sampel Acak
    for i in range(num_samples):
        # Pilih kelas acak
        random_class = random.choice(classes)
        class_path = os.path.join(base_dir, random_class)

        # Pilih gambar acak dari kelas tersebut
        images = [f for f in os.listdir(class_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        if not images:
            continue

        random_image = random.choice(images)
        img_full_path = os.path.join(class_path, random_image)

        # Load Gambar
        try:
            img = Image.open(img_full_path)

            # Metadata
            human_label = class_mapping.get(random_class, "Unknown Label")
            img_size = img.size

            # Tampilkan di Grafik
            axes[i].imshow(img)
            axes[i].axis('off')

            # Tulis Metadata di Judul
            title_text = f"{human_label}\nID: {random_class}\nSize: {img_size}"
            axes[i].set_title(title_text, fontsize=9, color='darkblue')

        except Exception as e:
            print(f"❌ Error loading image: {e}")

    plt.suptitle(f"Sampel Visualisasi Dataset ({split_name})", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Cek Data Training
    visualize_samples(config.TRAIN_DIR, split_name="TRAIN Set", num_samples=5)

    # Cek Data Validation (Penting untuk memastikan script download tadi berhasil menata folder)
    visualize_samples(config.VAL_DIR, split_name="VALIDATION Set", num_samples=5)