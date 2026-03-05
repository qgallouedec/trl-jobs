"""SFT (Supervised Fine-Tuning) command for SageMaker."""

import json
import time
from argparse import Namespace, _SubParsersAction
from importlib.resources import files
from typing import Optional

import yaml

# SageMaker SDK v3 imports
from sagemaker.train.model_trainer import ModelTrainer
from sagemaker.core.training.configs import SourceCode, Compute

from trl_sagemaker.auth import get_session, validate_credentials


# Default HuggingFace TRL training image

#TODO: use the latest image
#TODO: make it configurable
DEFAULT_TRL_IMAGE = "763104351884.dkr.ecr.us-east-1.amazonaws.com/huggingface-pytorch-training:2.8.0-transformers4.56.2-gpu-py312-cu129-ubuntu22.04-v1.2"
 


# Model-specific configurations mapping
# Key: (model_name, peft_mode)
# Value: config filename
CONFIGS = {
    # Qwen3-8B
    ("Qwen/Qwen3-8B", "no_peft"): "Qwen3-8B-p4d.yaml",
    ("Qwen/Qwen3-8B", "peft"): "Qwen3-8B-p4d-peft.yaml",
    # Qwen3-14B
    ("Qwen/Qwen3-14B", "peft"): "Qwen3-14B-p4d-peft.yaml",
    # Llama-3-8B
    ("meta-llama/Meta-Llama-3-8B", "peft"): "Llama-3-8B-p4d-peft.yaml",
    # SmolLM3-3B
    ("HuggingFaceTB/SmolLM3-3B", "no_peft"): "SmolLM3-3B-g5-12xlarge.yaml",
}


class SFTCommand:
    @staticmethod
    def register_subcommand(parser: _SubParsersAction) -> None:
        sft_parser = parser.add_parser("sft", help="Run SFT training on SageMaker")

        # Model & Data
        sft_parser.add_argument(
            "--model_name",
            type=str,
            required=True,
            help="HuggingFace model name (e.g., Qwen/Qwen3-8B)",
        )
        sft_parser.add_argument(
            "--dataset_name",
            type=str,
            required=True,
            help="HuggingFace dataset name or S3 path (s3://...)",
        )

        # Training configuration
        sft_parser.add_argument(
            "--peft",
            action="store_true",
            help="Use PEFT/LoRA instead of full fine-tuning",
        )
        sft_parser.add_argument(
            "--instance_type",
            type=str,
            default="ml.p4d.24xlarge",
            help="SageMaker instance type (default: ml.p4d.24xlarge)",
        )
        sft_parser.add_argument(
            "--instance_count",
            type=int,
            default=1,
            help="Number of training instances (default: 1)",
        )

        # SageMaker specific
        sft_parser.add_argument(
            "--role",
            type=str,
            help="SageMaker execution role ARN (uses default if not specified)",
        )
        sft_parser.add_argument(
            "--output_path",
            type=str,
            help="S3 output path for model artifacts (s3://bucket/prefix)",
        )
        sft_parser.add_argument(
            "--image",
            type=str,
            default=DEFAULT_TRL_IMAGE,
            help=f"Training Docker image (default: {DEFAULT_TRL_IMAGE})",
        )

        # Job control
        sft_parser.add_argument(
            "--max_runtime",
            type=int,
            default=86400,
            help="Maximum training time in seconds (default: 86400 = 24h)",
        )
        sft_parser.add_argument(
            "-d",
            "--detach",
            action="store_true",
            help="Don't wait for job completion, return immediately",
        )
        sft_parser.add_argument(
            "--job_name",
            type=str,
            help="Custom job name (auto-generated if not specified)",
        )

        sft_parser.set_defaults(func=SFTCommand)

    def __init__(self, args: Namespace, extra_args: list[str]) -> None:
        self.model_name: str = args.model_name
        self.dataset_name: str = args.dataset_name
        self.peft: bool = args.peft
        self.instance_type: str = args.instance_type
        self.instance_count: int = args.instance_count
        self.role: Optional[str] = args.role
        self.output_path: Optional[str] = args.output_path
        self.image: str = args.image
        self.max_runtime: int = args.max_runtime
        self.detach: bool = args.detach
        self.job_name: Optional[str] = args.job_name

        # Get AWS session from global args
        self.session = get_session(
            profile=getattr(args, "profile", None),
            region=getattr(args, "region", None),
        )

        # Load configuration if available
        self.config = self._load_config()

        # Parse extra_args and merge with config
        self.training_args = self._build_training_args(extra_args)

    def _load_config(self) -> dict:
        """Load model-specific configuration if available."""
        key = (self.model_name, "peft" if self.peft else "no_peft")

        if key in CONFIGS:
            config_file = CONFIGS[key]
            config_path = files("trl_sagemaker.configs.models").joinpath(config_file)
            with open(config_path, "r") as f:
                return yaml.safe_load(f)
        else:
            # No pre-configured setup, use defaults
            return {}

    def _build_training_args(self, extra_args: list[str]) -> dict:
        """Parse extra CLI args and merge with config."""
        # Start with loaded config
        args_dict = dict(self.config)

        # Add our own model/dataset args
        args_dict["model_name_or_path"] = self.model_name
        args_dict["dataset_name"] = self.dataset_name
        
        if self.peft:
            args_dict["use_peft"] = True

        # Parse extra_args (--key value format)
        overrides = {}
        i = 0
        while i < len(extra_args):
            if extra_args[i].startswith("--"):
                key = extra_args[i][2:]
                # Handle flags without values (bools)
                if i + 1 >= len(extra_args) or extra_args[i + 1].startswith("--"):
                    overrides[key] = True
                    i += 1
                else:
                    overrides[key] = extra_args[i + 1]
                    i += 2
            else:
                i += 1

        # Merge: CLI args override config
        args_dict.update(overrides)

        return args_dict

    def _build_trl_command(self) -> str:
        """Build the trl sft command with all arguments."""
        cmd_parts = ["trl", "sft"]
        
        for key, value in self.training_args.items():
            if isinstance(value, bool):
                # TRL CLI expects --flag true/false for some args
                cmd_parts.extend([f"--{key}", str(value).lower()])
            elif isinstance(value, list):
                # Lists should be space-separated for TRL CLI
                cmd_parts.append(f"--{key}")
                cmd_parts.extend([str(v) for v in value])
            elif isinstance(value, dict):
                # Dicts should be JSON
                cmd_parts.extend([f"--{key}", f"'{json.dumps(value)}'"])
            else:
                cmd_parts.extend([f"--{key}", str(value)])
        
        return " ".join(cmd_parts)

    def run(self) -> None:
        # Validate AWS credentials
        identity = validate_credentials(self.session)
        print(f"✓ Authenticated as: {identity}")

        # Generate job name if not provided
        job_name = self.job_name
        if not job_name:
            timestamp = time.strftime("%Y%m%d-%H%M%S", time.gmtime())
            model_short = self.model_name.split("/")[-1]
            job_name = f"{model_short}-sft-{timestamp}"

        # Build the TRL command
        trl_command = self._build_trl_command()

        print(f"🚀 Starting SFT training job...")
        print(f"   Job name: {job_name}")
        print(f"   Model: {self.model_name}")
        print(f"   Dataset: {self.dataset_name}")
        print(f"   Instance: {self.instance_type} x {self.instance_count}")
        print(f"   Training type: {'LoRA' if self.peft else 'Full fine-tuning'}")
        print(f"   Image: {self.image}")
        print(f"   Command: {trl_command}")

        # Configure source code (just the command, no local files)
        source_code = SourceCode(command=trl_command)

        # Configure compute
        compute = Compute(
            instance_type=self.instance_type,
            instance_count=self.instance_count,
        )

        # Create ModelTrainer
        trainer = ModelTrainer(
            training_image=self.image,
            source_code=source_code,
            compute=compute,
            role=self.role,
            base_job_name=job_name,
            sagemaker_session=self.session,
        )

        # Start training
        trainer.train(wait=not self.detach)

        print(f"✓ Job started: {job_name}")

        if self.detach:
            print("Running in detached mode. Monitor your job in the SageMaker console.")
        else:
            print("✓ Training complete!")
