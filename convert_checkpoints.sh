#!/bin/bash

# Verifica che i parametri siano stati forniti
echo "Verifica parametri..."
if [ "$#" -ne 4 ]; then
    echo "Uso: $0 <CHECKPOINTS-DIR> <OUTPUT-DIR> <CONFIG-NAME> <PYTHIA-TYPE>"
    exit 1
fi

# Assegna i parametri a variabili
CHECKPOINTS_DIR=$1
OUTPUT_DIR=$2
CONFIG_NAME=$3
PYTHIA_TYPE=$4

# Verifica che PYTHIA_TYPE sia di un valore corretto
VALID_TYPES=("14M" "31M" "70M" "160M" "410M" "1B" "1.4B" "2.8B" "6.9B" "12B")
if [[ ! " ${VALID_TYPES[@]} " =~ " $PYTHIA_TYPE " ]]; then
    echo "Errore: PYTHIA-TYPE deve essere uno dei seguenti valori: ${VALID_TYPES[*]}"
    exit 1
fi

# Verifica che CHECKPOINTS_DIR esista e sia una directory
echo "Verifica cartella input..."
if [ ! -d "$CHECKPOINTS_DIR" ]; then
    echo "Errore: La directory $CHECKPOINTS_DIR non esiste."
    exit 1
fi

# Crea OUTPUT_DIR se non esiste
echo "Verifica cartella output..."
mkdir -p "$OUTPUT_DIR"


# Ciclo su tutte le sottocartelle di CHECKPOINTS_DIR
echo "Inizio elaborazione..."
for STEP_DIR in "$CHECKPOINTS_DIR"/*/; do
    # Rimuove eventuali slash finali dal nome della sottocartella
    STEP_DIR_NAME=$(basename "$STEP_DIR")

    echo "Elaboro $STEP_DIR_NAME..."

    # Costruisce il percorso al file di configurazione
    CONFIG_FILE="$CHECKPOINTS_DIR/$STEP_DIR_NAME/configs/$CONFIG_NAME"

    # Verifica che il file di configurazione esista
    if [ ! -f "$CONFIG_FILE" ]; then
        echo "Errore: File di configurazione $CONFIG_FILE non trovato. Procedo con la prossima directory."
        continue
    fi

    # Esegui il comando python
    echo "Eseguo conversione per $STEP_DIR_NAME...\n"
    sudo python tools/convert_to_hf.py \
        --input_dir "$CHECKPOINTS_DIR/$STEP_DIR_NAME" \
        --config_file "$CHECKPOINTS_DIR/$STEP_DIR_NAME/configs/$CONFIG_NAME" \
        --output_dir "$OUTPUT_DIR/$STEP_DIR_NAME"

    echo "\n\n"

    # Rimuove file special_tokens_map.json e tokenizer_config.json (bug noto di Pythia)
    sudo rm -f "$OUTPUT_DIR/$STEP_DIR_NAME/special_tokens_map.json"
    sudo rm -f "$OUTPUT_DIR/$STEP_DIR_NAME/tokenizer_config.json"

    # Scarica i file di configurazione corretti
    sudo wget -q -O "$OUTPUT_DIR/$STEP_DIR_NAME/special_tokens_map.json" \
        "https://huggingface.co/EleutherAI/pythia-$PYTHIA_TYPE/raw/main/special_tokens_map.json"
    sudo wget -q -O "$OUTPUT_DIR/$STEP_DIR_NAME/tokenizer_config.json" \
        "https://huggingface.co/EleutherAI/pythia-$PYTHIA_TYPE/raw/main/tokenizer_config.json"

    echo "\n\n"

done

echo "Elaborazione completata."

