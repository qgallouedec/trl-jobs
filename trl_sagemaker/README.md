# 🚀 TRL SageMaker

**TRL SageMaker** is a CLI tool for running [TRL](https://huggingface.co/docs/trl/) (Transformer Reinforcement Learning) training jobs on **Amazon SageMaker**. It provides optimized configurations for popular models and handles all the SageMaker boilerplate for you.

## 📦 Installation

### Install with SageMaker support only

```bash
pip install trl-jobs[sagemaker]
```

This installs only the dependencies needed for `trl-sagemaker` (boto3, sagemaker SDK, pyyaml).

### Install everything

```bash
pip install trl-jobs[all]
```

This installs both `trl-jobs` (for Hugging Face Jobs) and `trl-sagemaker` (for AWS SageMaker) with all dependencies.

## 🔐 AWS Authentication

Before running jobs, you need valid AWS credentials. TRL SageMaker uses the standard AWS credential chain:

### Option 1: AWS SSO (Recommended)

```bash
# Configure SSO
aws configure sso

# Login
aws sso login --profile <your-profile>

# Run with profile
trl-sagemaker sft --profile <your-profile> --model_name Qwen/Qwen3-8B --dataset_name trl-lib/Capybara
```

### Option 2: Environment Variables

```bash
export AWS_ACCESS_KEY_ID=AKIA...
export AWS_SECRET_ACCESS_KEY=...
export AWS_DEFAULT_REGION=us-east-1

trl-sagemaker sft --model_name Qwen/Qwen3-8B --dataset_name trl-lib/Capybara
```

### Option 3: AWS Credentials File

```bash
# Configure credentials
aws configure

# This creates ~/.aws/credentials with your keys
```

### Option 4: IAM Role (EC2/ECS/Lambda)

If running on AWS infrastructure, credentials are automatically available via instance metadata.

### SageMaker Execution Role

You'll also need a **SageMaker execution role** with permissions to:
- Access S3 buckets for data and model artifacts
- Create and manage SageMaker training jobs
- Pull Docker images from ECR

```bash
# Pass role ARN explicitly
trl-sagemaker sft --role arn:aws:iam::123456789012:role/SageMakerExecutionRole ...
```

## ⚡ Quick Start

Run your first SFT job on SageMaker:

```bash
trl-sagemaker sft \
  --model_name Qwen/Qwen3-8B \
  --dataset_name trl-lib/Capybara \
  --role arn:aws:iam::123456789012:role/SageMakerExecutionRole
```

With LoRA/PEFT (recommended for larger models):

```bash
trl-sagemaker sft \
  --model_name meta-llama/Meta-Llama-3-8B \
  --dataset_name trl-lib/Capybara \
  --peft \
  --role arn:aws:iam::123456789012:role/SageMakerExecutionRole
```

Detached mode (returns immediately):

```bash
trl-sagemaker sft \
  --model_name Qwen/Qwen3-8B \
  --dataset_name trl-lib/Capybara \
  --role arn:aws:iam::123456789012:role/SageMakerExecutionRole \
  --detach
```

## 🛠 Available Commands

### 🔹 SFT (Supervised Fine-Tuning)

```bash
trl-sagemaker sft --model_name <model> --dataset_name <dataset> [options]
```

#### Required Arguments

| Argument | Description |
|----------|-------------|
| `--model_name` | HuggingFace model name (e.g., `Qwen/Qwen3-8B`) |
| `--dataset_name` | HuggingFace dataset name or S3 path (`s3://...`) |

#### Optional Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--peft` | `False` | Use PEFT/LoRA instead of full fine-tuning |
| `--instance_type` | `ml.p4d.24xlarge` | SageMaker instance type |
| `--instance_count` | `1` | Number of training instances |
| `--role` | Auto-detected | SageMaker execution role ARN |
| `--output_path` | Auto-generated | S3 path for model artifacts |
| `--image` | HF TRL image | Custom training Docker image |
| `--max_runtime` | `86400` (24h) | Maximum training time in seconds |
| `--job_name` | Auto-generated | Custom job name |
| `-d, --detach` | `False` | Don't wait for completion |

#### AWS Arguments

| Argument | Description |
|----------|-------------|
| `--profile` | AWS CLI profile name |
| `--region` | AWS region (e.g., `us-east-1`) |

#### Extra TRL Arguments

You can pass any argument supported by `trl sft`:

```bash
trl-sagemaker sft \
  --model_name Qwen/Qwen3-8B \
  --dataset_name trl-lib/Capybara \
  --role arn:aws:iam::123456789012:role/SageMakerExecutionRole \
  --learning_rate 3e-5 \
  --num_train_epochs 3 \
  --per_device_train_batch_size 8
```

For the full list, see the [TRL CLI docs](https://huggingface.co/docs/trl/en/clis).

## 📊 Supported Configurations

Pre-optimized configs are available for these model/instance combinations:

### Full Fine-Tuning

| Model | Instance | Config |
|-------|----------|--------|
| [Qwen/Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B) | `ml.p4d.24xlarge` | ✅ |
| [HuggingFaceTB/SmolLM3-3B](https://huggingface.co/HuggingFaceTB/SmolLM3-3B) | `ml.g5.12xlarge` | ✅ |

### LoRA/PEFT

| Model | Instance | Config |
|-------|----------|--------|
| [Qwen/Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B) | `ml.p4d.24xlarge` | ✅ |
| [Qwen/Qwen3-14B](https://huggingface.co/Qwen/Qwen3-14B) | `ml.p4d.24xlarge` | ✅ |
| [meta-llama/Meta-Llama-3-8B](https://huggingface.co/meta-llama/Meta-Llama-3-8B) | `ml.p4d.24xlarge` | ✅ |

### Example Commands

```bash
# Qwen3-8B full fine-tuning on p4d
trl-sagemaker sft --model_name Qwen/Qwen3-8B --dataset_name trl-lib/Capybara --role <role-arn>

# Qwen3-8B with LoRA on p4d
trl-sagemaker sft --model_name Qwen/Qwen3-8B --dataset_name trl-lib/Capybara --peft --role <role-arn>

# Llama-3-8B with LoRA on p4d
trl-sagemaker sft --model_name meta-llama/Meta-Llama-3-8B --dataset_name trl-lib/Capybara --peft --role <role-arn>

# SmolLM3-3B on g5 (cheaper instance)
trl-sagemaker sft --model_name HuggingFaceTB/SmolLM3-3B --dataset_name trl-lib/Capybara --instance_type ml.g5.12xlarge --role <role-arn>
```

## 📈 Experiment Tracking

Training runs are tracked with [TrackIO](https://huggingface.co/docs/trackio/index). Make sure you're logged in to Hugging Face:

```bash
huggingface-cli login
```

Runs appear in your HuggingFace profile under the TrackIO tab.

To specify a custom TrackIO space, pass `--hub_model_id`:

```bash
trl-sagemaker sft \
  --model_name Qwen/Qwen3-8B \
  --dataset_name trl-lib/Capybara \
  --role <role-arn> \
  -- --hub_model_id your-username/your-trackio-space
```

## 🏗 SageMaker Instance Types

Common instance types for training:

| Instance | GPUs | GPU Memory | Use Case |
|----------|------|------------|----------|
| `ml.g5.xlarge` | 1x A10G | 24 GB | Small models, testing |
| `ml.g5.12xlarge` | 4x A10G | 96 GB | Medium models (3-7B) |
| `ml.p4d.24xlarge` | 8x A100 | 320 GB | Large models (7-14B+) |
| `ml.p5.48xlarge` | 8x H100 | 640 GB | Very large models |

## 🔧 Troubleshooting

### Invalid Role ARN

```
ParamValidationError: Invalid length for parameter RoleArn
```

**Solution:** Use the full ARN format:
```bash
--role arn:aws:iam::123456789012:role/YourRoleName
```

### Invalid Job Name

```
ValidationException: trainingJobName failed to satisfy constraint
```

**Solution:** Job names can only contain alphanumeric characters and hyphens. Use `--job_name` to specify a valid name:
```bash
--job_name my-training-job-001
```

### No Credentials Found

```
RuntimeError: No AWS credentials found
```

**Solution:** Configure AWS credentials using one of the methods above.

## 📜 License

MIT License. See [LICENSE](../LICENSE) for details.
