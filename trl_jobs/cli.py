import json
import time
from argparse import ArgumentParser, Namespace, _SubParsersAction
from importlib.resources import files
from typing import Optional

import yaml
from huggingface_hub import HfApi, SpaceHardware
from huggingface_hub.utils import get_token_to_send, logging

import warnings

logger = logging.get_logger(__name__)


SUGGESTED_FLAVORS = [item.value for item in SpaceHardware if item.value != "zero-a10g"]

CONFIGS = {
    # Meta-Llama-3
    ("meta-llama/Meta-Llama-3-8B", "a100-large", "no_peft"): "Meta-Llama-3-8B-a100-large.yaml",
    ("meta-llama/Meta-Llama-3-8B-Instruct", "a100-large", "no_peft"): "Meta-Llama-3-8B-Instruct-a100-large.yaml",
    # Meta-Llama-3 peft
    ("meta-llama/Meta-Llama-3-8B", "a100-large", "peft"): "Meta-Llama-3-8B-a100-large-peft.yaml",
    ("meta-llama/Meta-Llama-3-8B-Instruct", "a100-large", "peft"): "Meta-Llama-3-8B-Instruct-a100-large-peft.yaml",
    # Open GPT-OSS
    ("openai/gpt-oss-20b", "a100-large", "peft"): "gpt-oss-20b-a100-large-peft.yaml",
    # Qwen3
    ("Qwen/Qwen3-0.6B-Base", "a100-large", "no_peft"): "Qwen3-0.6B-Base-a100-large.yaml",
    ("Qwen/Qwen3-0.6B", "a100-large", "no_peft"): "Qwen3-0.6B-a100-large.yaml",
    ("Qwen/Qwen3-1.7B-Base", "a100-large", "no_peft"): "Qwen3-1.7B-Base-a100-large.yaml",
    ("Qwen/Qwen3-1.7B", "a100-large", "no_peft"): "Qwen3-1.7B-a100-large.yaml",
    ("Qwen/Qwen3-4B-Base", "a100-large", "no_peft"): "Qwen3-4B-Base-a100-large.yaml",
    ("Qwen/Qwen3-4B", "a100-large", "no_peft"): "Qwen3-4B-a100-large.yaml",
    ("Qwen/Qwen3-8B-Base", "a100-large", "no_peft"): "Qwen3-8B-Base-a100-large.yaml",
    ("Qwen/Qwen3-8B", "a100-large", "no_peft"): "Qwen3-8B-a100-large.yaml",
    # Qwen3 peft
    ("Qwen/Qwen3-8B-Base", "a100-large", "peft"): "Qwen3-8B-Base-a100-large-peft.yaml",
    ("Qwen/Qwen3-8B", "a100-large", "peft"): "Qwen3-8B-a100-large-peft.yaml",
    ("Qwen/Qwen3-14B-Base", "a100-large", "peft"): "Qwen3-14B-Base-a100-large-peft.yaml",
    ("Qwen/Qwen3-14B", "a100-large", "peft"): "Qwen3-14B-a100-large-peft.yaml",
    ("Qwen/Qwen3-32B-Base", "a100-large", "peft"): "Qwen3-32B-Base-a100-large-peft.yaml",
    ("Qwen/Qwen3-32B", "a100-large", "peft"): "Qwen3-32B-a100-large-peft.yaml",
    # SmolLM3
    ("HuggingFaceTB/SmolLM3-3B-Base", "a100-large", "no_peft"): "SmolLM3-3B-Base-a100-large.yaml",
    ("HuggingFaceTB/SmolLM3-3B", "a100-large", "no_peft"): "SmolLM3-3B-a100-large.yaml",
}


class SFTCommand:
    @staticmethod
    def register_subcommand(parser: _SubParsersAction) -> None:
        sft_parser = parser.add_parser("sft", help="Run a SFT training job")
        sft_parser.add_argument(
            "--model_name",
            type=str,
            required=True,
            help="Model name (e.g., Qwen/Qwen3-4B-Base)",
        )
        sft_parser.add_argument(
            "--peft",
            action="store_true",
            help="Whether to use PEFT (LoRA) or not. Defaults to False.",
        )
        sft_parser.add_argument(
            "--flavor",
            default="a100-large",
            type=str,
            help=f"Flavor for the hardware, as in HF Spaces. Defaults to `a100-large`. Possible values: {', '.join(SUGGESTED_FLAVORS)}.",
        )
        sft_parser.add_argument(
            "--timeout",
            default="1h",
            type=str,
            help="Max duration: int/float with s (seconds, default), m (minutes), h (hours) or d (days).",
        )
        sft_parser.add_argument(
            "-d",
            "--detach",
            action="store_true",
            help="Run the Job in the background and print the Job ID.",
        )
        sft_parser.add_argument(
            "--namespace",
            type=str,
            help="The namespace where the Job will be created. Defaults to the current user's namespace.",
        )
        sft_parser.add_argument(
            "--token",
            type=str,
            help="A User Access Token generated from https://huggingface.co/settings/tokens",
        )
        sft_parser.set_defaults(func=SFTCommand)

    def __init__(self, args: Namespace, extra_args: list[str]) -> None:
        self.model_name: str = args.model_name
        self.peft: bool = args.peft
        self.flavor: Optional[SpaceHardware] = args.flavor
        self.timeout: Optional[str] = args.timeout
        self.detach: bool = args.detach
        self.namespace: Optional[str] = args.namespace
        self.token: Optional[str] = args.token

        # Check if the requested configuration exists
        key = (self.model_name, self.flavor, "peft" if self.peft else "no_peft")
        if key in CONFIGS:
            config_file = CONFIGS[key]
            # Load YAML file
            config_file = files("trl_jobs.configs").joinpath(config_file)
            with open(config_file, "r") as f:
                args_dict = yaml.safe_load(f)
        else:
            warnings.warn(
                f"❌ No configuration found for:\n"
                f"   • model: {self.model_name}\n"
                f"   • flavor: {self.flavor}\n"
                f"   • peft: {self.peft}\n\n"
                "⚠️  No optimal configuration found. The job will still be launched "
                "using the CLI flags provided by the user, but optimal performance "
                "is not guaranteed.\n"
                "👉 If you think this configuration should be supported, consider "
                "opening an issue or submitting a PR:\n"
                "https://github.com/huggingface/trl-jobs",
                UserWarning,
            )
            args_dict = {}
            args_dict["model_name_or_path"] = self.model_name
            args_dict["use_peft"] = self.peft

        # Add our own hub_model_id to avoid overwriting a previously trained model
        if "hub_model_id" not in args_dict:
            timestamp = time.strftime("%Y%m%d%H%M%S", time.gmtime())
            if self.namespace:
                args_dict["hub_model_id"] = f"{self.namespace}/{self.model_name.split('/')[-1]}-SFT-{timestamp}"
            else:
                args_dict["hub_model_id"] = f"{self.model_name.split('/')[-1]}-SFT-{timestamp}"

        # Same for run_name
        if "run_name" not in args_dict:
            args_dict["run_name"] = f"{self.model_name.split('/')[-1]}-SFT-{timestamp}"

        # Parse extra_args into a dictionary
        overrides = {}
        i = 0
        while i < len(extra_args):
            if extra_args[i].startswith("--"):
                key = extra_args[i][2:]
                # handle flags without values (bools)
                if i + 1 >= len(extra_args) or extra_args[i + 1].startswith("--"):
                    overrides[key] = True
                    i += 1
                else:
                    overrides[key] = extra_args[i + 1]
                    i += 2
            else:
                i += 1

        # Override YAML args with CLI args
        merged = {**args_dict, **overrides}

        # Rebuild CLI args
        self.cli_args = []
        for k, v in merged.items():
            if isinstance(v, (dict, list, bool, type(None))):
                v_str = json.dumps(v)
            else:
                v_str = str(v)
            self.cli_args.extend([f"--{k}", v_str])

    def run(self) -> None:
        api = HfApi(token=self.token)
        job = api.run_job(
            image="huggingface/trl",
            command=["trl", "sft", *self.cli_args],
            secrets={"HF_TOKEN": get_token_to_send(self.token)},
            flavor=self.flavor,
            timeout=self.timeout,
            namespace=self.namespace,
        )
        # Always print the job ID to the user
        print(f"Job started with ID: {job.id}")
        print(f"View at: {job.url}")

        if self.detach:
            return

        # Now let's stream the logs
        for log in api.fetch_job_logs(job_id=job.id):
            print(log)


def main():
    parser = ArgumentParser(prog="trl-jobs")
    commands_parser = parser.add_subparsers(dest="command", help="trl-jobs commands")
    SFTCommand.register_subcommand(commands_parser)

    args, extra_args = parser.parse_known_args()
    if not hasattr(args, "func"):
        parser.print_help()
        exit(1)

    service = args.func(args, extra_args)
    if service is not None:
        service.run()


if __name__ == "__main__":
    main()
