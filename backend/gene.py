import os

TRAIN_PATH = "data/train"     # Folder jisme tumhare class folders hain
OUTPUT_FILE = "model/labels.txt"   # Jahan output save hoga

def generate_labels():
    if not os.path.exists(TRAIN_PATH):
        print(f"❌ Error: '{TRAIN_PATH}' not found!")
        return

    # Get sorted folder names
    folder_names = sorted(os.listdir(TRAIN_PATH))

    # Ensure output directory exists
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

    with open(OUTPUT_FILE, "w") as f:
        for idx, folder in enumerate(folder_names):
            f.write(f"{idx} {folder}\n")

    print("✅ labels.txt generated successfully!")
    print(f"📁 Saved to: {OUTPUT_FILE}")
    print("📌 Total Classes:", len(folder_names))

if __name__ == "__main__":
    generate_labels()
