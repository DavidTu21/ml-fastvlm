#!/usr/bin/env python3
"""Extract FastViT-HD vision encoder from Apple's FastVLM checkpoints."""
import json, argparse, torch
from pathlib import Path
from safetensors.torch import save_file
from llava.model.builder import load_pretrained_model

def detect_model_size(path):
    """Auto-detect model size from checkpoint path."""
    path_lower = path.lower()
    for size in ['0.5b', '1.5b', '7b']:
        if size in path_lower: return size
    return 'unknown'

def clean_state_dict_keys(state_dict):
    """Remove common prefixes from state dict keys."""
    cleaned = {}
    prefixes = ['vision_model.', 'model.', 'vision_tower.']
    for key, value in state_dict.items():
        clean_key = key
        for prefix in prefixes:
            if clean_key.startswith(prefix):
                clean_key = clean_key[len(prefix):]
                break
        cleaned[clean_key] = value
    return cleaned

def extract_vision_config(vision_tower):
    """Extract vision configuration from the loaded vision tower."""
    # Detect scales and image size
    scales = getattr(vision_tower, 's2_scales', None)
    if scales and isinstance(scales, list) and len(scales) > 0:
        scales = [int(s) for s in scales]
        image_size = max(scales)
        print(f"Detected s2_scales: {scales}, setting image_size to {image_size}")
    else:
        scales = [336, 672, 1008]
        image_size = max(scales)
        print(f"No s2_scales detected, using fallback: {scales}, image_size: {image_size}")
    
    # Extract config from vision model
    vision_config = getattr(vision_tower, 'config', None)
    if hasattr(vision_tower, 'vision_tower') and hasattr(vision_tower.vision_tower, 'config'):
        vision_config = vision_tower.vision_tower.config
    
    # Extract hidden size and patch size with fallbacks
    hidden_size = 3072 if not (vision_config and hasattr(vision_config, 'hidden_size')) and not hasattr(vision_tower, 'hidden_size') else (vision_config.hidden_size if vision_config and hasattr(vision_config, 'hidden_size') else vision_tower.hidden_size)
    patch_size = 64 if not (vision_config and hasattr(vision_config, 'patch_size')) else vision_config.patch_size
    
    return {'scales': scales, 'image_size': image_size, 'hidden_size': hidden_size, 
            'patch_size': patch_size, 'use_multiscale': len(scales) > 1}

def main():
    parser = argparse.ArgumentParser(description="Extract FastViT-HD vision encoder")
    parser.add_argument("--checkpoint-path", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--model-size", type=str, choices=['0.5b', '1.5b', '7b'])
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_size = args.model_size or detect_model_size(args.checkpoint_path)
    
    print(f"Loading model from: {args.checkpoint_path}")
    print(f"Model size: {model_size}")
    
    # Load model
    tokenizer, model, image_processor, _ = load_pretrained_model(args.checkpoint_path, None, model_size, device_map="cpu")
    
    # Get vision tower
    vision_tower = model.get_vision_tower()
    if not vision_tower.is_loaded: vision_tower.load_model(device_map="cpu")
    
    print("Extracting vision configuration...")
    vision_config = extract_vision_config(vision_tower)
    
    # Extract and clean state dict
    state_dict = vision_tower.vision_tower.state_dict() if hasattr(vision_tower, 'vision_tower') else vision_tower.state_dict()
    cleaned_state_dict = clean_state_dict_keys(state_dict)
    
    # Create config.json
    config = {
        "model_type": "fastvithd",
        "vision_config": {
            "model_type": "fastvit_hd_vision_model", "hidden_size": vision_config['hidden_size'],
            "image_size": vision_config['image_size'], "patch_size": vision_config['patch_size'],
            "num_hidden_layers": 16, "num_attention_heads": 16, "intermediate_size": 4096,
            "layer_norm_eps": 1e-06, "attention_dropout": 0.0, "projection_dim": 1024,
            "architecture": "fastvit_hd", "scales": vision_config['scales'],
            "use_multiscale": vision_config['use_multiscale']
        },
        "architectures": ["FastViTHDModel"], "transformers_version": "4.49.0"
    }
    
    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    # Save model weights
    save_file(cleaned_state_dict, output_dir / "model.safetensors")
    
    # Create preprocessor_config.json
    processor_config = {
        "do_center_crop": getattr(image_processor, 'do_center_crop', True),
        "do_convert_rgb": getattr(image_processor, 'do_convert_rgb', True),
        "do_normalize": getattr(image_processor, 'do_normalize', True),
        "do_rescale": getattr(image_processor, 'do_rescale', True),
        "do_resize": getattr(image_processor, 'do_resize', True),
        "image_mean": getattr(image_processor, 'image_mean', [0.48145466, 0.4578275, 0.40821073]),
        "image_std": getattr(image_processor, 'image_std', [0.26862954, 0.26130258, 0.27577711]),
        "resample": 3, "rescale_factor": 0.00392156862745098,
        "size": {"shortest_edge": vision_config['image_size'], "longest_edge": vision_config['image_size']},
        "crop_size": {"height": vision_config['image_size'], "width": vision_config['image_size']},
        "multi_scale": {"enabled": vision_config['use_multiscale'], "scales": vision_config['scales'], "base_image_size": 1024}
    }
    
    with open(output_dir / "preprocessor_config.json", "w") as f:
        json.dump(processor_config, f, indent=2)
    
    # Create README.md
    readme = f"""# FastViT-HD Vision Encoder
Extracted from FastVLM checkpoint: `{args.checkpoint_path}`

## Model Details
- **Hidden Size**: {vision_config['hidden_size']}
- **Image Size**: {vision_config['image_size']}
- **Scales**: {vision_config['scales']}
- **Total Parameters**: {len(cleaned_state_dict)} tensors

## Usage
```python
import json
from safetensors.torch import load_file
with open("config.json", "r") as f: config = json.load(f)
state_dict = load_file("model.safetensors")
print(f"Hidden size: {{config['vision_config']['hidden_size']}}")
```"""
    
    with open(output_dir / "README.md", "w") as f:
        f.write(readme)
    
    # Print summary
    print(f"\\nExtraction complete!")
    print(f"Output directory: {output_dir}")
    print(f"Number of tensors: {len(cleaned_state_dict)}")
    print(f"Hidden size: {vision_config['hidden_size']}")
    print(f"Image size: {vision_config['image_size']}")
    print(f"Scales: {vision_config['scales']}")

if __name__ == "__main__":
    main()