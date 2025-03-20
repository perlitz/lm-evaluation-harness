import argparse
import os

from loguru import logger
from transformers.models.bamba.convert_mamba_ssm_checkpoint import (
    convert_mamba_ssm_checkpoint_file_to_huggingface_model_file,
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download files from S3.")

    parser.add_argument(
        "--input_model_path",
        type=str,
        required=True,
        help="Path to the input model directory.",
    )
    parser.add_argument(
        "--model_save_dir",
        type=str,
        required=True,
        help="Path to the output model directory.",
    )

    args = parser.parse_args()

    input_model_path = args.input_model_path
    tokenizer_files = [f for f in os.listdir(input_model_path) if "tok" in f]
    if len(tokenizer_files) == 0:
        logger.info(f"No tokenizer files found for {input_model_path}, skipping")
        raise ValueError(f"No tokenizer files found for {input_model_path}, skipping")

    out_model_dir_path = args.model_save_dir + input_model_path + "-hf"
    convert_mamba_ssm_checkpoint_file_to_huggingface_model_file(
        input_model_path,
        "fp16",
        out_model_dir_path,
        save_model="sharded",
        tokenizer_path=input_model_path,
    )
    import shutil

    for tokenizer_file in tokenizer_files:
        shutil.copyfile(
            src=os.path.join(input_model_path, tokenizer_file),
            dst=os.path.join(out_model_dir_path, tokenizer_file),
        )
