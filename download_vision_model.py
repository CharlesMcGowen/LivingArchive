#!/usr/bin/env python3
"""
Download and set up vision model for EgoLlama
Uses HuggingFace transformers to download LLaVA or similar vision models
"""

import os
import sys
from pathlib import Path

def download_llava_model():
    """Download LLaVA vision model from HuggingFace"""
    print("=" * 70)
    print("EgoLlama Vision Model Downloader")
    print("=" * 70)
    print()
    
    try:
        from transformers import AutoProcessor, AutoModelForVision2Seq
        print("✅ Transformers library available")
    except ImportError as e:
        print(f"❌ Error: {e}")
        print("\nPlease install transformers:")
        print("  pip install transformers")
        return False
    
    # Model options
    models = {
        "1": {
            "name": "llava-hf/llava-1.5-7b-hf",
            "size": "~14GB",
            "description": "LLaVA 1.5 7B - Best balance of performance and size"
        },
        "2": {
            "name": "llava-hf/llava-1.6-mistral-7b-hf",
            "size": "~14GB",
            "description": "LLaVA 1.6 Mistral 7B - Improved reasoning"
        },
        "3": {
            "name": "liuhaotian/llava-v1.6-34b",
            "size": "~68GB",
            "description": "LLaVA 1.6 34B - Highest accuracy (requires significant VRAM)"
        }
    }
    
    print("Available vision models:")
    print()
    for key, model in models.items():
        print(f"{key}. {model['name']}")
        print(f"   Size: {model['size']}")
        print(f"   {model['description']}")
        print()
    
    choice = input("Select model (1-3) [default: 1]: ").strip() or "1"
    
    if choice not in models:
        print("❌ Invalid choice")
        return False
    
    selected_model = models[choice]
    model_name = selected_model["name"]
    
    print()
    print(f"📥 Downloading {model_name}...")
    print(f"   Size: {selected_model['size']}")
    print("   This may take a while...")
    print()
    
    try:
        # Create models directory
        models_dir = Path("/mnt/webapps-nvme/ego/models/vision")
        models_dir.mkdir(parents=True, exist_ok=True)
        
        cache_dir = models_dir / model_name.replace("/", "_")
        
        print(f"📁 Downloading to: {cache_dir}")
        print()
        
        # Download model
        print("Downloading model...")
        model = AutoModelForVision2Seq.from_pretrained(
            model_name,
            cache_dir=str(cache_dir),
            low_cpu_mem_usage=True
        )
        print("✅ Model downloaded")
        
        # Download processor
        print("Downloading processor...")
        processor = AutoProcessor.from_pretrained(
            model_name,
            cache_dir=str(cache_dir)
        )
        print("✅ Processor downloaded")
        
        # Save configuration
        config_file = models_dir / "vision_model_config.txt"
        with open(config_file, "w") as f:
            f.write(f"model_name={model_name}\n")
            f.write(f"cache_dir={cache_dir}\n")
            f.write(f"size={selected_model['size']}\n")
        
        print()
        print("=" * 70)
        print("✅ Vision model downloaded successfully!")
        print("=" * 70)
        print()
        print(f"Model: {model_name}")
        print(f"Location: {cache_dir}")
        print()
        print("Next steps:")
        print("1. Restart EgoLlama gateway: docker restart ego-llama-gateway")
        print("2. Test in Cursor with an image!")
        print()
        
        return True
        
    except Exception as e:
        print(f"❌ Error downloading model: {e}")
        import traceback
        traceback.print_exc()
        return False

def check_disk_space():
    """Check if there's enough disk space"""
    import shutil
    stat = shutil.disk_usage("/mnt/webapps-nvme")
    free_gb = stat.free / (1024**3)
    
    print(f"💾 Available disk space: {free_gb:.1f} GB")
    
    if free_gb < 20:
        print("⚠️  Warning: Less than 20GB free. Vision models are large!")
        response = input("Continue anyway? (y/n): ").strip().lower()
        return response == 'y'
    
    return True

if __name__ == "__main__":
    print()
    
    if not check_disk_space():
        print("Exiting...")
        sys.exit(1)
    
    print()
    success = download_llava_model()
    
    sys.exit(0 if success else 1)

