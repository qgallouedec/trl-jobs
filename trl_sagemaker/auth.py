"""AWS authentication utilities for trl-sagemaker."""

import boto3
from botocore.exceptions import NoCredentialsError, ClientError
from sagemaker.core.helper.session_helper import Session as SageMakerSession


def get_session(profile: str | None = None, region: str | None = None) -> SageMakerSession:
    """
    Create a SageMaker session using standard AWS credential chain.

    Credential resolution order (handled by boto3):
    1. Environment variables (AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_SESSION_TOKEN)
    2. Shared credentials file (~/.aws/credentials)
    3. AWS config file (~/.aws/config) - including SSO
    4. IAM role (EC2/ECS/Lambda)

    Args:
        profile: Optional AWS CLI profile name (from ~/.aws/config)
        region: Optional AWS region (defaults to profile/env default)

    Returns:
        SageMakerSession: Configured SageMaker session
    """
    boto_session = boto3.Session(profile_name=profile, region_name=region)
    return SageMakerSession(boto_session=boto_session)


def validate_credentials(session: SageMakerSession) -> str:
    """
    Validate AWS credentials and return the caller identity ARN.

    Args:
        session: SageMaker session to validate

    Returns:
        str: The ARN of the authenticated identity

    Raises:
        RuntimeError: If credentials are invalid or missing
    """
    try:
        sts = session.boto_session.client("sts")
        identity = sts.get_caller_identity()
        return identity["Arn"]
    except NoCredentialsError:
        raise RuntimeError(
            "No AWS credentials found. Configure using one of:\n"
            "  - AWS SSO: aws configure sso\n"
            "  - Environment: export AWS_ACCESS_KEY_ID=... AWS_SECRET_ACCESS_KEY=...\n"
            "  - Credentials file: aws configure\n"
            "  - CLI flag: --profile <profile_name>"
        )
    except ClientError as e:
        raise RuntimeError(f"AWS authentication failed: {e}")
