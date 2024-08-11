import argparse
import logging
import os
import sys

import torch
from transformers import LlamaForCausalLM, LlamaTokenizer


sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from main import SARA, MoSARA, SARALlamaModel, load_and_adapt_model, test_model

logging.basicConfig(level=logging.INFO, filename='output.log', filemode='w')
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Run SARA or Mo-SARA model on AWS")
    parser.add_argument("--model_name", type=str, default="decapoda-research/llama-7b-hf", help="Name of the pretrained model")
    parser.add_argument("--use_mo_sara", action="store_true", help="Use Mo-SARA instead of SARA")
    parser.add_argument("--threshold", type=float, default=0.01, help="Threshold for SARA/Mo-SARA")
    parser.add_argument("--num_experts", type=int, default=5, help="Number of experts for Mo-SARA")
    parser.add_argument("--input_text", type=str, default="Once upon a time", help="Input text for model testing")
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Using device: {device}")

    model_type = "Mo-SARA" if args.use_mo_sara else "SARA"
    logger.info(f"Testing {model_type}:")

    try:
        model, tokenizer = load_and_adapt_model(args.model_name, args.use_mo_sara, args.threshold, args.num_experts, device)
        output = test_model(model, tokenizer, args.input_text, device)
        logger.info(f"{model_type} output: {output}")

        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Number of trainable parameters in {model_type}: {trainable_params}")
    except Exception as e:
        logger.error(f"Error running model: {e}")
        raise

if __name__ == "__main__":
    main()