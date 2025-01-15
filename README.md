# pythia-retrain
A simple guide to train the Pythia suite with custom data (on the DGX)

Source: 
 - [EleutherAI/pythia: Reproducing Training](https://github.com/EleutherAI/pythia/tree/main?tab=readme-ov-file#reproducing-training)
 - [EleutherAI/gpt-neox: Using Custom Data](https://github.com/EleutherAI/gpt-neox/tree/v1.0?tab=readme-ov-file#using-custom-data)


## 1. Setup the GPT-NeoX Environment

- Go to the [DGX Portainer](https://10.20.30.114:9443/#!/auth), enter the *Images* tab and search for my pre-built image *pythia:latest*
  - If you don't find it, proceed with the next steps, otherwise go to next section of this guide  
- Download from this repository the *addon_files.zip* archive
- Go to the [DGX Portainer](https://10.20.30.114:9443/#!/auth), enter the *Images* tab and *Build a new image*
- Copy and Paste the DockerFile contained in this repository in the *Web Editor* and upload the *addon_files.zip* archives
- Choose a significant *name*, like *pythia:latest*, and build the image


## 2. Create the container

- Go to the *Containers* tab and *Add Container*
  - Set the *Image* to **pythia:latest**
  - In *Command & logging*, set the *Console* to **Interactive & TTY**
  - In *Volumes*, *Map additional volumes* with *container* **/data** and *volume* **AirgData** (or other personal Volumes you have)
  - In *Env*, *Add an environment variable* with *name* **NVIDIA_VISIBLE_DEVICES** and *value* **0**
  - In *Runtime & Resources*, set the *Runtime* to **nvidia**
  - In *Runtime & Resources*, enable and choose a *GPU*
  - In *Runtime & Resources*, set the *Resources* according to [DGX guidelines](https://elemental-freesia-008.notion.site/GPU-server-DGX-A100-68f768412f2f48ce808937efbe44f796)
- Finally, *Deploy the container*


## 3. Prepare the data

- Start the container
- Download the scripts (in this guide we train a 160M Pythia)
  ```bash
  cd /data/pythia-retrain  
  git clone https://github.com/EleutherAI/gpt-neox.git
  cd gpt-neox
  git config --global --add safe.directory /data/pythia-retrain/gpt-neox

  mkdir ../input
  cd ../input
  
  sudo wget https://raw.githubusercontent.com/EleutherAI/pythia/refs/heads/main/models/160M/pythia-160m.yml
  sudo wget https://raw.githubusercontent.com/EleutherAI/pythia/refs/heads/main/utils/20B_tokenizer.json
- Prepare the data in a JSONL file, with one row per document, with the data in the *text* column 
- Move the JSONL file in the *input* directory
- Run the command:
  ```bash
  cd ../gpt-neox
  sudo python tools/preprocess_data.py \
            --input ../input/mydata.jsonl \
            --output-prefix  ../input/pythia_mydata_idxmaps/mydata\
            --vocab ../input/20B_tokenizer.json \
            --tokenizer-type HFTokenizer

## 4. Start the training

- Create the folder *checkpoints*
  ```bash
  mkdir ../checkpoints/mydata-pythia160m
  
- Mofify the Pythia Config file (input/pythia-160m.yml)
  ```json
  "train_micro_batch_size_per_gpu": XXX,    # make this a value that will fit within your GPU memory
  "gradient_accumulation_steps": YYY,       # make this a value to compensate to make the total batch size: XXX * YYY = 1024

  "train-data-paths": ["../input/pythia_mydata_idxmaps"], #point this to your folder containing the .bin and .idx file
  "valid-data-paths": ["../input/pythia_mydata_idxmaps"], #point this to your folder containing the .bin and .idx file
  "test-data-paths": ["../input/pythia_mydata_idxmaps"],  #point this to your folder containing the .bin and .idx file

  "tokenizer-type": "HFTokenizer",
  "vocab-file": "../input/20B_tokenizer.json", # point this to the tokenizer retrieved

  # Save the checkpoints
  "launcher": "slurm",
  "deepspeed_slurm": false,

  "save": "../checkpoints/mydata-pythia160m",
  "load": "../checkpoints/mydata-pythia160m",
  "checkpoint_validation_with_forward_pass": False,

- Modify L432 in *megatron/models/transformers.py*
    ```python
    qkv = torch.cat([query_layer, key_layer, value_layer], dim=1)

- Modify L270 in *megatron/checkpointing.py* to patch a bug:
  ```python
    iteration = state_dict.get("iteration")
    if iteration is None:
        iteration = state_dict.get("total_iters") # total_iters backward compatible with older checkpoints

- Run the training:
  ```bash
  python deepy.py train.py ../input/pythia-160m.yml  2>&1 | tee output.txt

## 4. Convert the Checkpoints to HuggingFace format

- Create the *output* directory
  ```bash
  mkdir ../output/mydata-pythia160m
  
- Modify the *tools/convert_to_hf.py* file L44 (and add the correct import):
  ```python
  from typing import List

  ) -> List[torch.Tensor]:


- Download the convertion_checkpoints.sh from this repository in the /gpt-neox directory:
- Execute the script:
  ```bash
  sudo chmod +x convert_checkpoints.sh
  sudo ./convert_checkpoints.sh ../checkpoints/mydata-pythia160m ../output/mydata-pythia160m pythia-160m.yml 160M
