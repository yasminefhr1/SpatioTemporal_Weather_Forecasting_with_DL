import os
import re
import argparse
import requests
from tqdm import tqdm

API_URL = "https://www.data.gouv.fr/api/1/datasets/6569b3d7d193b4daf2b43edc/"

KEYWORDS = [
    "periode_1950-2023",
    "periode_2024-2026",
    "1950-2023",
    "2024-2026",

]

def download_data(out_dir: str):
    print("Récupération du jeu de données...")

    try:
        response = requests.get(API_URL, timeout=30)
        response.raise_for_status()
        metadata = response.json()
    except Exception as e:
        print(f"Erreur lors de l'accès à l'API : {e}")
        return

    resources = metadata.get("resources", [])
    print(f"Analyse de {len(resources)} fichiers...")

    to_download = []
    for res in resources:
        title = (res.get("title") or "").lower()
        url = res.get("url") or ""

        if any(k.lower() in title for k in KEYWORDS) and url:
            to_download.append(res)

    print(f"Démarrage du téléchargement de {len(to_download)} fichiers...")

    os.makedirs(out_dir, exist_ok=True)

    for res in tqdm(to_download):
        file_url = res.get("url")
        titre_fichier = res.get("title", "unknown_file")

        safe_filename = re.sub(r"[^a-zA-Z0-9_.-]", "_", titre_fichier)
        if not safe_filename.endswith(".csv.gz"):
            safe_filename += ".csv.gz"

        output_path = os.path.join(out_dir, safe_filename)

        if os.path.exists(output_path):
            continue

        try:
            with requests.get(file_url, stream=True, timeout=60) as r:
                r.raise_for_status()
                with open(output_path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
        except Exception as e:
            print(f"Erreur téléchargement {titre_fichier}: {e}")

    print("\nTéléchargement terminé")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=str, default="data/raw")
    args = parser.parse_args()
    download_data(args.out_dir)