# FastViT-HD Extraction Script Usage

## Overview
The `extract_fastvithd_complete.py` script extracts the FastViT-HD vision encoder from Apple's FastVLM checkpoints and creates standalone vision encoder files.

## Requirements
- Python 3.8+
- PyTorch
- SafeTensors
- Transformers
- llava package (from this repository)

## Usage

```bash
python extract_fastvithd_complete.py --checkpoint-path <path_to_checkpoint> --output-dir <output_directory> [--model-size <size>]
```

### Arguments
- `--checkpoint-path` (required): Path to a FastVLM checkpoint directory (e.g., `checkpoints/llava-fastvithd_7b_stage3`)
- `--output-dir` (required): Output directory to write extracted files
- `--model-size` (optional): Model size (`0.5b`, `1.5b`, `7b`). If omitted, auto-detected from checkpoint path.

## Output Files
The script creates exactly 4 files in the output directory:

1. **config.json**: Vision encoder configuration with detected parameters
2. **model.safetensors**: Vision encoder weights only (no language model artifacts)
3. **preprocessor_config.json**: Image processing configuration
4. **README.md**: Usage instructions for the extracted model

## Example

```bash
# Extract from a 7B model checkpoint
python extract_fastvithd_complete.py \
  --checkpoint-path checkpoints/llava-fastvithd_7b_stage3 \
  --output-dir extracted_vision_encoder \
  --model-size 7b

# Auto-detect model size
python extract_fastvithd_complete.py \
  --checkpoint-path checkpoints/llava-fastvithd_7b_stage3 \
  --output-dir extracted_vision_encoder
```

## Key Features
- Automatically detects multi-scale settings (s2_scales) from the vision tower
- Derives image_size from max(s2_scales) when available
- Extracts hidden_size and patch_size from actual model config
- Falls back to sensible defaults from mobileclip_l.json when needed
- Cleans state dict keys by removing common prefixes
- Creates standalone vision encoder with all necessary configuration files
- Compact implementation (≤150 lines of code)

## Validation
The script automatically validates:
- Multi-scale detection: Looks for `s2_scales` attribute on vision tower
- Configuration extraction: Attempts to get config from vision model
- Fallback handling: Uses mobileclip_l.json defaults when needed
- State dict cleaning: Removes prefixes like `vision_model.`, `model.`, `vision_tower.`

## Output Summary
After extraction, the script prints:
- Output directory location
- Number of extracted tensors
- Detected hidden size
- Detected image size
- Detected scales for multi-scale processing