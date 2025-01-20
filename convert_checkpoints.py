import os
from pathlib import Path
import argparse
import requests
from huggingface_hub import create_repo, create_branch, HfApi, HfFolder

# Funzione per verificare se una directory è vuota
def is_dir_empty(directory):
    return not any(directory.iterdir())

# Funzione per scaricare un file
def download_file(url, dest):
    try:
        response = requests.get(url)
        response.raise_for_status()
        with open(dest, 'wb') as f:
            f.write(response.content)
    except requests.RequestException as e:
        print(f"Error downloading {url}: {e}")

# Funzione per estrarre il numero da una stringa
def extract_number(s):
    return int(''.join(filter(str.isdigit, s)))

# Funzione principale
def main():
    # Parser degli argomenti
    parser = argparse.ArgumentParser(description="Convert checkpoints to Hugging Face format")
    parser.add_argument("-d", "--checkpoints_dir", type=str, help="Directory containing the checkpoints")
    parser.add_argument("-o", "--output_dir", type=str, help="Directory for the output")
    parser.add_argument("-n", "--config_name", type=str, help="Name of the configuration file")
    parser.add_argument("-p", "--pythia_type", type=str, help="Pythia type (e.g., 14M, 160M, etc.)")
    parser.add_argument("-r", "--repo_name", type=str, help="Name of the Hugging Face repository")
    parser.add_argument("--last_step_in_training", type=bool, default=True, help="Whether to ignore the last step in training (default: True)")
    args = parser.parse_args()

    # Parametri
    checkpoints_dir = Path(args.checkpoints_dir)
    output_dir = Path(args.output_dir)
    config_name = args.config_name
    pythia_type = args.pythia_type.upper()
    repo_name = args.repo_name
    last_step_in_training = args.last_step_in_training

    # Tipi validi di PYTHIA
    valid_types = ["14M", "31M", "70M", "160M", "410M", "1B", "1.4B", "2.8B", "6.9B", "12B"]
    if pythia_type not in valid_types:
        print(f"Error: PYTHIA_TYPE must be one of {', '.join(valid_types)}")
        exit(1)

    # Verifica che CHECKPOINTS_DIR esista
    if not checkpoints_dir.is_dir():
        print(f"Error: Directory {checkpoints_dir} does not exist.")
        exit(1)

    # Crea OUTPUT_DIR se non esiste
    output_dir.mkdir(parents=True, exist_ok=True)

    # Crea il repository su Hugging Face
    print(f"Creating Hugging Face repository: {repo_name}")
    api = HfApi()
    token = HfFolder.get_token()
    try:
        repo_url = create_repo(repo_id=repo_name, token=token, exist_ok=True)
    except Exception as e:
        print(f"Error creating repository: {e}")
        exit(1)

    # Scarica i file di configurazione corretti una sola volta
    print("Downloading configuration files...")
    urls = {
        "special_tokens_map.json": f"https://huggingface.co/EleutherAI/pythia-{pythia_type}/raw/main/special_tokens_map.json",
        "tokenizer_config.json": f"https://huggingface.co/EleutherAI/pythia-{pythia_type}/raw/main/tokenizer_config.json"
    }
    config_files = {}
    for file_name, url in urls.items():
        dest_path = output_dir / file_name
        download_file(url, dest_path)
        config_files[file_name] = dest_path

    # Ordina le sottocartelle
    step_dirs = sorted([d for d in checkpoints_dir.iterdir() if d.is_dir()], key=lambda x: extract_number(x.name))

    # Ignora l'ultimo step se specificato
    last_step_dir = None
    if last_step_in_training and step_dirs:
        last_step_dir = step_dirs.pop(-1)

    # Ciclo sulle sottocartelle ordinate
    for step_dir in step_dirs:
        step_dir_name = step_dir.name
        print(f"Processing {step_dir_name}...")

        step_output_dir = output_dir / step_dir_name

        # Salta se la directory di output non è vuota
        if step_output_dir.exists() and not is_dir_empty(step_output_dir):
            print(f"Directory {step_output_dir} is not empty. Skipping.")
            continue

        # Percorso del file di configurazione
        config_file = step_dir / "configs" / config_name
        if not config_file.is_file():
            print(f"Error: Config file {config_file} not found. Skipping.")
            continue

        # Esegui il comando Python con il percorso corretto
        print(f"Running conversion for {step_dir_name}...")
        os.system(
            f"python tools/convert_to_hf.py --input_dir {step_dir} --config_file {config_file} --output_dir {step_output_dir}"
        )

        # Copia i file scaricati nella directory di output
        for file_name, src_path in config_files.items():
            dest_path = step_output_dir / file_name
            dest_path.write_bytes(src_path.read_bytes())

        # Carica il modello su Hugging Face nel branch dello step
        print(f"Uploading to Hugging Face: {repo_name}, branch: {step_dir_name[7:]}")
        create_branch(repo_id=repo_url.repo_id, branch=step_dir_name[7:], token=token, exist_ok=True)
        api.upload_folder(
            folder_path=str(step_output_dir),
            repo_id=repo_url.repo_id,
            repo_type="model",
            revision=step_dir_name[7:],
            token=token
        )

        last_step_output_dir = step_output_dir

    # Carica l'ultimo step nel branch principale
    if last_step_dir:
        print(f"Uploading last step to Hugging Face: {repo_name}, branch: main")
        api.upload_folder(
            folder_path=str(last_step_output_dir),
            repo_id=repo_url.repo_id,
            repo_type="model",
            revision="main",
            token=token
        )

    # Rimuovi i file scaricati dalla cartella principale
    print("Cleaning up downloaded files...")
    for file_name, src_path in config_files.items():
        if src_path.exists():
            src_path.unlink()

    print("Processing completed.")

if __name__ == "__main__":
    main()



