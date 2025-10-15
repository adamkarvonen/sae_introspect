
import os
import tempfile
from huggingface_hub import HfApi, hf_hub_download, snapshot_download
from pathlib import Path

def download_and_reupload():
    # Source repository and subdirectory
    source_repo = "adamkarvonen/loras"
    # subdir = "model_lora_Qwen_Qwen3-8B_evil_claude37/misaligned_2_to_normal_12"
    # target_repo = "thejaminator/misaligned_2_to_normal_12"  # Will be uploaded as a new repository
    subdir = "model_lora_Qwen_Qwen3-8B_sycophancy_mixed_claude37/misaligned_2"
    target_repo = "thejaminator/syco_misaligned_2"  # Will be uploaded as a new repository
    
    # Initialize HF API
    api = HfApi()
    
    # Create temporary directory for download
    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"Downloading files to temporary directory: {temp_dir}")
        
        # Try different repo types
        repo_types = ["model", "dataset", "space"]
        local_path = None
        
        for repo_type in repo_types:
            try:
                print(f"Trying repo_type: {repo_type}")
                local_path = snapshot_download(
                    repo_id=source_repo,
                    repo_type=repo_type,
                    local_dir=temp_dir,
                    allow_patterns=f"{subdir}/*"
                )
                print(f"Successfully downloaded with repo_type: {repo_type}")
                break
            except Exception as e:
                print(f"Failed with repo_type {repo_type}: {e}")
                continue
        
        if local_path is None:
            print("Failed to download with any repo_type. Please check:")
            print("1. Repository exists and is accessible")
            print("2. You are authenticated with Hugging Face (run 'huggingface-cli login')")
            print("3. Repository is not private or you have access to it")
            return
        
        # Find the actual downloaded subdirectory
        subdir_path = os.path.join(temp_dir, subdir)
        if not os.path.exists(subdir_path):
            # Try alternative path structure
            subdir_path = os.path.join(temp_dir, "model_lora_Qwen_Qwen3-8B_evil_claude37", "misaligned_2")
        
        if not os.path.exists(subdir_path):
            print(f"Error: Could not find subdirectory {subdir} in downloaded files")
            print(f"Available files in {temp_dir}:")
            for root, dirs, files in os.walk(temp_dir):
                level = root.replace(temp_dir, '').count(os.sep)
                indent = ' ' * 2 * level
                print(f"{indent}{os.path.basename(root)}/")
                subindent = ' ' * 2 * (level + 1)
                for file in files:
                    print(f"{subindent}{file}")
            return
        
        print(f"Found files in: {subdir_path}")
        
        # List all files to be uploaded
        files_to_upload = []
        for root, dirs, files in os.walk(subdir_path):
            for file in files:
                file_path = os.path.join(root, file)
                # Calculate relative path from subdir_path
                rel_path = os.path.relpath(file_path, subdir_path)
                files_to_upload.append((file_path, rel_path))
        
        print(f"Files to upload: {[f[1] for f in files_to_upload]}")
        
        # Create the target repository
        try:
            api.create_repo(
                repo_id=target_repo,
                repo_type="model",  # Change to "dataset" if needed
                exist_ok=True
            )
            print(f"Created/verified repository: {target_repo}")
        except Exception as e:
            print(f"Error creating repository: {e}")
            return
        
        # Upload files
        print("Uploading files...")
        for local_file_path, repo_file_path in files_to_upload:
            try:
                api.upload_file(
                    path_or_fileobj=local_file_path,
                    path_in_repo=repo_file_path,
                    repo_id=target_repo,
                    repo_type="model"  # Change to "dataset" if needed
                )
                print(f"✓ Uploaded: {repo_file_path}")
            except Exception as e:
                print(f"✗ Failed to upload {repo_file_path}: {e}")
        
        print(f"Upload complete! Repository: {target_repo}")

if __name__ == "__main__":
    download_and_reupload()

