import os
import shutil
import argparse

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Run lm_eval with specified model and output directory.")
parser.add_argument("-m", "--model_name", required=True, help="The model name to use")
parser.add_argument("-o", "--output_dir", required=True, help="The output directory to save results")
args = parser.parse_args()

model_name = args.model_name
output_dir = args.output_dir

# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)

# Define the list of values for s
STEPS = [
    0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1000, 2000, 
    3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000, 11000, 
    12000, 13000, 14000, 15000, 16000, 17000, 18000, 19000, 20000
]

TASKS = ["lambada_openai", "piqa", "winogrande", "wsc273", "arc_easy", "arc_challenge", "sciq", "logiqa"]

# Iterate through each value in the list
for s in STEPS:
    print(f"Running evaluation for step{s}...")

    # Build the command string
    command = (
        f"lm_eval --model hf "
        f"--model_args \"pretrained={model_name},revision=step{s},dtype=float\" "
        f"--tasks {','.join(TASKS)} "
        f"--device cuda:0 "
        f"--batch_size auto:4 "
        f"--output_path {output_dir}"
    )

    # Execute the command using os.system
    exit_code = os.system(command)
    if exit_code != 0:
        print(f"Error: Command failed for step{s} with exit code {exit_code}")
        break

    # Locate the generated JSON file
    try:
        subdirectories = [
            os.path.join(output_dir, d) for d in os.listdir(output_dir)
            if os.path.isdir(os.path.join(output_dir, d))
        ]
        json_file_path = None
        for subdirectory in subdirectories:
            for file in os.listdir(subdirectory):
                if file.endswith(".json"):
                    json_file_path = os.path.join(subdirectory, file)
                    break
            if json_file_path:
                break

        if json_file_path and os.path.exists(json_file_path):
            shutil.move(json_file_path, os.path.join(output_dir, f"step{s}.json"))

            # Remove the empty subdirectory
            os.rmdir(subdirectory)
        else:
            print(f"Error: JSON file not found in subdirectories for step{s}")
            break

    except Exception as e:
        print(f"Error while locating or moving JSON file for step{s}: {e}")
        break

print("All evaluations completed.")

