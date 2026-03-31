import os
import requests
import re
from tqdm import tqdm

# url récupérée depuis
# https://explore.data.gouv.fr/fr/datasets/6569b3d7d193b4daf2b43edc/#/resources/f7e77d52-1496-4761-b395-288793e63155
API_URL="https://www.data.gouv.fr/api/1/datasets/6569b3d7d193b4daf2b43edc/"

# on va récupérer tous les fichiers des départements pour la période 1950/2023 et pour 2024-aujourd'hui
fichiers=["periode_1950-2023", "periode_2024"]

def download_data():
    print("Récupération du jeu de données...")
    try:
        response=requests.get(API_URL)
        response.raise_for_status()
        metadata=response.json()
    except Exception as e:
        print(f"Erreur lors de l'accès à l'API : {e}")
        return

    resources=metadata.get('resources', [])
    print(f"Analyse de {len(resources)} fichiers...")

    # filter les fichiers à télécharger
    to_download = []
    for res in resources:
        title=res.get('title', '')
        # vérifier que le titre correspond aux fichiers à récupérer
        if any(fichier in title for fichier in fichiers):
            to_download.append(res)

    print(f"Démarrage du téléchargement de {len(to_download)} fichiers...")

    for res in tqdm(to_download):
        file_url=res.get('url')
        titre_fichier=res.get('title')

        # vérification du nom de fichier
        safe_filename = re.sub(r'[^a-zA-Z0-9_\-\.]', '_', titre_fichier)
        if not safe_filename.endswith('.csv.gz'):
             safe_filename += '.csv.gz'
             
        # crée le dossier "data/raw" s'il n'existe pas
        os.makedirs("data/raw", exist_ok=True)
        output_path = os.path.join("data/raw", safe_filename)

        # si le fichier n'est pas déjà enregistré, on le télécharge
        if not os.path.exists(output_path):
            try:
                # utilisation de stream pour ne pas surcharger la mémoire
                with requests.get(file_url, stream=True) as r:
                    r.raise_for_status()
                    with open(output_path, 'wb') as f:
                        for chunk in r.iter_content(chunk_size=8192):
                            f.write(chunk)
            except Exception as e:
                print(f"Erreur téléchargement {titre_fichier}: {e}")

        else:
            # rien à faire, le fichier existe déjà : on passe au suivant
            pass

    print(f"\nTéléchargement terminé")

if __name__ == "__main__":
    download_data()