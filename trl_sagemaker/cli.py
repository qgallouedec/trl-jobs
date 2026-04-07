"""Main CLI entry point for trl-sagemaker."""

from argparse import ArgumentParser

from trl_sagemaker.commands.sft import SFTCommand


def main():
    parser = ArgumentParser(
        prog="trl-sagemaker",
        description="Launch TRL training jobs on AWS SageMaker",
    )

    # Global AWS options
    parser.add_argument(
        "--profile",
        type=str,
        help="AWS CLI profile name (from ~/.aws/config)",
    )
    parser.add_argument(
        "--region",
        type=str,
        help="AWS region (defaults to profile/env default)",
    )

    commands_parser = parser.add_subparsers(dest="command", help="Training commands")
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
