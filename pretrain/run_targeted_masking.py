#!/usr/bin/env python3
"""
python run_targeted_masking.py --data_path /path/to/your/dataset --use_targeted_masking --high_mask_ratio 0.8 --low_mask_ratio 0.4
"""

import os
import sys
import subprocess

def main():
    # Basic parameters
    base_cmd = [
        "main.py",
        "--data_path", "/root/lanyun-tmp/spark_data/CT_Pretrain", 
        "--model", "resnet50",
        "--input_size", "224",
        "--bs", "128", 
        "--ep", "500",
        "--lr", "2e-4",
        "--mask", "0.8",  # This parameter is ignored when using targeted masking but should be retained
    ]
    
    targeted_masking_cmd = [
        "--use_targeted_masking", "True",
    ]
    
    exp_configs = [
        {
            "name": "random_masking",
            "desc": "Random masking strategy",
            "cmd": base_cmd + ["--exp_name", "random_masking_baseline"],
            "high_ratio": None,
            "low_ratio": None,
        },
        {
            "name": "targeted_masking_high",
            "desc": "High masking in salient regions",
            "cmd": base_cmd + ["--use_targeted_masking", "True", "--exp_name", "targeted_masking_high", "--high_mask_ratio", "0.8", "--low_mask_ratio", "0.4"],
            "high_ratio": "0.8",
            "low_ratio": "0.4",
        },
        {
            "name": "targeted_masking_medium",
            "desc": "Medium masking in salient regions", 
            "cmd": base_cmd + ["--use_targeted_masking", "True", "--exp_name", "targeted_masking_medium", "--high_mask_ratio", "0.7", "--low_mask_ratio", "0.3"],
            "high_ratio": "0.7",
            "low_ratio": "0.3",
        },
        {
            "name": "targeted_masking_low",
            "desc": "Low masking in salient regions",
            "cmd": base_cmd + ["--use_targeted_masking", "True", "--exp_name", "targeted_masking_low", "--high_mask_ratio", "0.6", "--low_mask_ratio", "0.2"],
            "high_ratio": "0.6",
            "low_ratio": "0.2",
        },
    ]
    
    print("=== Targeted Masking Training Demo ===")
    print("Make sure your dataset is structured as follows:")
    print("dataset_root/")
    print("  ├── images/           # raw images")
    print("  └── masks_expanded/   # expanded masks")
    print()
    
    # Check dataset path
    data_path = "/root/lanyun-tmp/spark_data/new_combined_data"
    print(f"Using dataset path: {data_path}")
    
    if not os.path.exists(data_path):
        print(f"Error: dataset path {data_path} does not exist")
        print("Please check the path or manually input the correct path:")
        data_path = input("Enter your dataset path: ").strip()
        if not os.path.exists(data_path):
            print(f"Error: dataset path {data_path} does not exist")
            return
    
    if not os.path.exists(os.path.join(data_path, "images")):
        print(f"Error: 'images' folder not found under {data_path}")
        return
    
    if not os.path.exists(os.path.join(data_path, "masks_expanded")):
        print(f"Error: 'masks_expanded' folder not found under {data_path}")
        return
    
    print(f"Dataset path check passed: {data_path}")
    print()
    
    # Show available experiment configs
    print("Available experiment configurations:")
    for i, config in enumerate(exp_configs):
        print(f"{i+1}. {config['name']} - {config['desc']}")
        if config['high_ratio'] is not None:
            print(f"   Salient region masking ratio: {config['high_ratio']}")
            print(f"   Non-salient region masking ratio: {config['low_ratio']}")
        else:
            print("   Random masking (60% uniform masking)")
        print()
    
    # Choose experiment config
    choice = input("Select an experiment configuration to run (1-4): ").strip()
    try:
        choice_idx = int(choice) - 1
        if choice_idx < 0 or choice_idx >= len(exp_configs):
            print("Invalid selection")
            return
    except ValueError:
        print("Invalid selection")
        return
    
    selected_config = exp_configs[choice_idx]
    
    # Command already includes correct dataset path
    cmd = selected_config['cmd'].copy()
    
    # Customize experiment name and output directory
    print(f"\nSelected configuration: {selected_config['name']} - {selected_config['desc']}")
    
    # Custom experiment name
    custom_exp_name = input("Customize experiment name? (y/N): ").strip().lower()
    if custom_exp_name == 'y':
        exp_name = input("Enter experiment name: ").strip()
        if exp_name:
            for i, arg in enumerate(cmd):
                if arg == "--exp_name":
                    cmd[i + 1] = exp_name
                    break
    
    # Custom output directory
    custom_exp_dir = input("Customize output directory? (y/N): ").strip().lower()
    if custom_exp_dir == 'y':
        exp_dir = input("Enter output directory path: ").strip()
        if exp_dir:
            cmd.extend(["--exp_dir", exp_dir])
    
    # Load pretrained weights
    use_init_weight = input("Load pretrained weights (init_weight)? (y/N): ").strip().lower()
    if use_init_weight == 'y':
        init_weight = input("Enter init_weight file path: ").strip()
        if init_weight:
            cmd.extend(["--init_weight", init_weight])
    
    # Distributed training option
    use_ddp = input("Use distributed training (torchrun)? (y/N): ").strip().lower()
    if use_ddp == 'y':
        nproc = input("Enter number of GPUs per node (e.g., 2/4/8): ").strip()
        if not nproc.isdigit():
            print("Invalid input, using default: 1")
            nproc = "1"
        torchrun_prefix = ["torchrun", f"--nproc_per_node={nproc}"]
    else:
        torchrun_prefix = [sys.executable]
    
    final_cmd = torchrun_prefix + cmd
    
    print(f"\nRunning experiment: {selected_config['name']}")
    print(f"Command: {' '.join(final_cmd)}")
    print()
    
    # Confirm execution
    confirm = input("Proceed with training? (y/N): ").strip().lower()
    if confirm != 'y':
        print("Cancelled")
        return
    
    # Run command
    try:
        print("Starting training...")
        subprocess.run(final_cmd, check=True)
        print("Training completed!")
    except subprocess.CalledProcessError as e:
        print(f"Training failed with exit code: {e.returncode}")
    except KeyboardInterrupt:
        print("Training interrupted by user")

if __name__ == "__main__":
    main()
