"""Command-line interface for training."""

import argparse
import json
from pathlib import Path

from llm.training.config import DPOConfig, PretrainConfig, SFTConfig


def load_config_from_file(path: str, config_class):
    """Load config from JSON file."""
    with open(path, "r") as f:
        data = json.load(f)
    return config_class(**data)


def main():
    parser = argparse.ArgumentParser(description="LargeForgeAI Training CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # SFT command
    sft_parser = subparsers.add_parser("sft", help="Supervised fine-tuning")
    sft_parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-7B")
    sft_parser.add_argument("--dataset", type=str, required=True)
    sft_parser.add_argument("--output", type=str, default="./output/sft")
    sft_parser.add_argument("--config", type=str, help="Path to JSON config file")
    sft_parser.add_argument("--epochs", type=int, default=3)
    sft_parser.add_argument("--batch-size", type=int, default=1)
    sft_parser.add_argument("--lr", type=float, default=2e-5)
    sft_parser.add_argument("--lora-r", type=int, default=8)
    sft_parser.add_argument("--no-lora", action="store_true")
    sft_parser.add_argument("--no-4bit", action="store_true")

    # Pretrain command
    pretrain_parser = subparsers.add_parser("pretrain", help="Continued pretraining")
    pretrain_parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-7B")
    pretrain_parser.add_argument("--dataset", type=str, required=True)
    pretrain_parser.add_argument("--output", type=str, default="./output/pretrain")
    pretrain_parser.add_argument("--config", type=str, help="Path to JSON config file")
    pretrain_parser.add_argument("--epochs", type=int, default=1)
    pretrain_parser.add_argument("--batch-size", type=int, default=2)
    pretrain_parser.add_argument("--lr", type=float, default=1e-5)
    pretrain_parser.add_argument("--use-lora", action="store_true")

    # DPO command
    dpo_parser = subparsers.add_parser("dpo", help="DPO preference training")
    dpo_parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-7B")
    dpo_parser.add_argument("--dataset", type=str, required=True)
    dpo_parser.add_argument("--output", type=str, default="./output/dpo")
    dpo_parser.add_argument("--config", type=str, help="Path to JSON config file")
    dpo_parser.add_argument("--epochs", type=int, default=1)
    dpo_parser.add_argument("--batch-size", type=int, default=1)
    dpo_parser.add_argument("--lr", type=float, default=5e-6)
    dpo_parser.add_argument("--beta", type=float, default=0.1)

    args = parser.parse_args()

    if args.command == "sft":
        from llm.training.sft import train_sft

        if args.config:
            config = load_config_from_file(args.config, SFTConfig)
        else:
            config = SFTConfig(
                model_name=args.model,
                dataset_path=args.dataset,
                output_dir=args.output,
                num_train_epochs=args.epochs,
                per_device_train_batch_size=args.batch_size,
                learning_rate=args.lr,
                use_lora=not args.no_lora,
                load_in_4bit=not args.no_4bit,
            )
            config.lora.r = args.lora_r

        print(f"Starting SFT training: {config.model_name}")
        train_sft(config)
        print(f"Training complete. Model saved to {config.output_dir}")

    elif args.command == "pretrain":
        from llm.training.pretrain import train_pretrain

        if args.config:
            config = load_config_from_file(args.config, PretrainConfig)
        else:
            config = PretrainConfig(
                model_name=args.model,
                dataset_path=args.dataset,
                output_dir=args.output,
                num_train_epochs=args.epochs,
                per_device_train_batch_size=args.batch_size,
                learning_rate=args.lr,
                use_lora=args.use_lora,
            )

        print(f"Starting pretraining: {config.model_name}")
        train_pretrain(config)
        print(f"Training complete. Model saved to {config.output_dir}")

    elif args.command == "dpo":
        from llm.training.dpo import train_dpo

        if args.config:
            config = load_config_from_file(args.config, DPOConfig)
        else:
            config = DPOConfig(
                model_name=args.model,
                dataset_path=args.dataset,
                output_dir=args.output,
                num_train_epochs=args.epochs,
                per_device_train_batch_size=args.batch_size,
                learning_rate=args.lr,
                beta=args.beta,
            )

        print(f"Starting DPO training: {config.model_name}")
        train_dpo(config)
        print(f"Training complete. Model saved to {config.output_dir}")


if __name__ == "__main__":
    main()
