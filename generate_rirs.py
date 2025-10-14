"""
Generate room impulse responses using pygsound with specified room dimensions
and random source/receiver positions, compatible with FastRirDataset.
"""

import os
import numpy as np
import argparse
import soundfile as sf
import json
import pickle
import csv
import random
from tqdm import tqdm
import pygsound as ps
import rt60 as rt60
import matplotlib.pyplot as plt


eps = 0.000000001
def generate_linearly_spaced_room_configs(num_configs, room_min_dims, room_max_dims, pairs_per_room=3, csv_output_path=None, randomize=False, random_seed=None):
    """Generate room configurations with dimensions linearly spaced from min to max.
    
    Args:
        num_configs: Number of configurations to generate
        room_min_dims: Minimum room dimensions [length, width, height]
        room_max_dims: Maximum room dimensions [length, width, height]
        pairs_per_room: Number of source/receiver pairs per unique room dimension
        csv_output_path: Optional path to save configurations as CSV
        randomize: Whether to randomize the order of configurations
        random_seed: Optional seed for reproducible randomization
        
    Returns:
        List of room configuration dictionaries
    """
    configs = []

    # Calculate how many unique room dimensions we need
    num_rooms = num_configs // pairs_per_room
    if num_configs % pairs_per_room != 0:
        num_rooms += 1  # Round up to ensure we have enough configurations

    # Number of steps for each dimension
    dim_steps = int(np.ceil(num_rooms ** (1/3)))  # Cube root to get steps per dimension

    # Create linearly spaced values for each dimension
    length_values = np.linspace(room_min_dims[0], room_max_dims[0], dim_steps)
    width_values = np.linspace(room_min_dims[1], room_max_dims[1], dim_steps)
    height_values = np.linspace(room_min_dims[2], room_max_dims[2], dim_steps)

    # For source and receiver positions, we'll create a grid within each room
    pos_steps = max(2, int(np.ceil(pairs_per_room ** (1/3))))
    max_positions = pos_steps**3  # Total possible positions in the grid

    room_count = 0
    config_count = 0
    
    # Generate configurations by taking combinations of room dimensions and positions
    for length in length_values:
        for width in width_values:
            for height in height_values:
                if config_count >= num_configs:
                    break

                room_dims = [length, width, height]

                # Create source positions grid (excluding areas too close to walls)
                min_distance = 0.5
                source_x_values = np.linspace(min_distance, length - min_distance, pos_steps)
                source_y_values = np.linspace(min_distance, width - min_distance, pos_steps)
                source_z_values = np.linspace(min_distance, height - min_distance, pos_steps)

                # Create receiver positions grid (excluding areas too close to walls)
                receiver_x_values = np.linspace(min_distance, length - min_distance, pos_steps)
                receiver_y_values = np.linspace(min_distance, width - min_distance, pos_steps)
                receiver_z_values = np.linspace(min_distance, height - min_distance, pos_steps)

                # Create multiple source/receiver pairs for this room
                for pair_idx in range(pairs_per_room):
                    if config_count >= num_configs:
                        break

                    # Use a different room index offset for each pair to ensure diversity
                    base_idx = (room_count * pairs_per_room + pair_idx) % max_positions
                    source_idx = base_idx
                    receiver_idx = (source_idx + max_positions // 2) % max_positions  # Maximize distance

                    # Convert indices to 3D coordinates
                    sx = source_idx // (pos_steps**2)
                    sy = (source_idx % (pos_steps**2)) // pos_steps
                    sz = source_idx % pos_steps

                    rx = receiver_idx // (pos_steps**2)
                    ry = (receiver_idx % (pos_steps**2)) // pos_steps
                    rz = receiver_idx % pos_steps

                    source_pos = [source_x_values[sx], source_y_values[sy], source_z_values[sz]]
                    receiver_pos = [receiver_x_values[rx], receiver_y_values[ry], receiver_z_values[rz]]

                    # Make sure source and receiver are not too close
                    if np.linalg.norm(np.array(source_pos) - np.array(receiver_pos)) < 1.0:
                        # Adjust receiver position if too close
                        receiver_pos[0] = min(length - min_distance, source_pos[0] + 1.5)

                    absorption_coeff = np.random.uniform(0.3, 0.9)

                    config = {
                        "room_dims": room_dims,
                        "source_pos": source_pos,
                        "receiver_pos": receiver_pos,
                        "absorption_coeff": absorption_coeff,
                        "room_index": room_count,
                        "pair_index": pair_idx
                    }

                    configs.append(config)
                    config_count += 1

                room_count += 1
                if config_count >= num_configs:
                    break

    configs = configs[:num_configs]
    
    # Randomize configurations if requested
    if randomize:
        if random_seed is not None:
            print(f"Randomizing room configurations with seed {random_seed}...")
            random.seed(random_seed)
        else:
            print("Randomizing room configurations with random seed...")
        random.shuffle(configs)
    
    # If CSV output path is provided, write configurations to CSV
    if csv_output_path:
        try:
            print(f"DEBUG: Attempting to write to CSV file: {os.path.abspath(csv_output_path)}")
            with open(csv_output_path, 'w', newline='') as csvfile:
                fieldnames = ['config_id', 'room_length', 'room_width', 'room_height', 
                             'source_x', 'source_y', 'source_z', 
                             'receiver_x', 'receiver_y', 'receiver_z', 
                             'absorption_coeff', 'room_index', 'pair_index']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                
                writer.writeheader()
                for i, config in enumerate(configs):
                    writer.writerow({
                        'config_id': i,
                        'room_length': config['room_dims'][0],
                        'room_width': config['room_dims'][1],
                        'room_height': config['room_dims'][2],
                        'source_x': config['source_pos'][0],
                        'source_y': config['source_pos'][1],
                        'source_z': config['source_pos'][2],
                        'receiver_x': config['receiver_pos'][0],
                        'receiver_y': config['receiver_pos'][1],
                        'receiver_z': config['receiver_pos'][2],
                        'absorption_coeff': config['absorption_coeff'],
                        'room_index': config['room_index'],
                        'pair_index': config['pair_index']
                    })
            print(f"DEBUG: Successfully wrote CSV file")
            print(f"DEBUG: File exists: {os.path.exists(csv_output_path)}")
            if os.path.exists(csv_output_path):
                print(f"DEBUG: File size: {os.path.getsize(csv_output_path)} bytes")
        except Exception as e:
            print(f"ERROR writing to CSV: {str(e)}")
            import traceback
            traceback.print_exc()

    return configs

def parse_room_configs_from_csv(csv_path):
    """Parse room configurations from a CSV file.
    
    Args:
        csv_path: Path to the CSV file containing room configurations
        
    Returns:
        List of room configuration dictionaries
    """
    configs = []
    
    with open(csv_path, 'r', newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            config = {
                'room_dims': [
                    float(row['room_length']),
                    float(row['room_width']),
                    float(row['room_height'])
                ],
                'source_pos': [
                    float(row['source_x']),
                    float(row['source_y']),
                    float(row['source_z'])
                ],
                'receiver_pos': [
                    float(row['receiver_x']),
                    float(row['receiver_y']),
                    float(row['receiver_z'])
                ],
                'absorption_coeff': float(row['absorption_coeff']),
                'room_index': int(row['room_index']),
                'pair_index': int(row['pair_index'])
            }
            configs.append(config)
    
    return configs

def generate_rir(config, sample_rate=16000, max_order=50):
    """Generate RIR using pygsound with the given configuration."""

    # Set up the context
    context = ps.Context()
    context.diffuse_count = 20000
    context.specular_count = 2000
    context.channel_type = ps.ChannelLayoutType.mono
    context.sample_rate = sample_rate

    # Set up the scene
    scene = ps.Scene()

    # Create the room mesh
    room_dims = config["room_dims"]
    #absorb = random.uniform(0.5, 0.9)
    absorption_coeff = config["absorption_coeff"]
    room = ps.createbox(room_dims[0], room_dims[1], room_dims[2], absorption_coeff, 0.1)
    #room = ps.createbox(8, 6, 2.5, 0.41, 0.1)
    scene.setMesh(room)

    # Set up source and receiver
    source_pos = config["source_pos"]
    receiver_pos = config["receiver_pos"]

    results = scene.computeIR([source_pos], [receiver_pos], context)
    #results = scene.computeIR([[2.96, 1.5, 1.75]], [[5.52, 4.44, 1.6]], context)

    # Extract impulse response
    ir = np.array(results['samples'][0][0][0])

    # Calculate RT60 (approximate)
    t60_fast = rt60.t60_impulse_fastRIR(ir,sample_rate)
    t60_overall = rt60.t60_impulse_overall(ir,sample_rate)
    t60_bands = rt60.t60_impulse_bands(ir,sample_rate)

    config["estimated_rt60"] = t60_fast
    config["t60_overall"] = t60_overall
    config["t60_bands"] = t60_bands

    return ir, config

def format_fast_rir_embedding(config):
    """
    Format the configuration into a FastRIR embedding format.
    According to FastRirDataset.decode_conditioner:
    Embedding = ([LP_X,LP_Y,LP_Z,SP_X,SP_Y,SP_Z,RD_X,RD_Y,RD_Z,(T60+CRR)] /5) - 1
    
    Extended to include additional RT60 information (overall and frequency bands).
    """
    mic_pos = config["receiver_pos"]
    source_pos = config["source_pos"]
    room_dims = config["room_dims"]
    rt60 = config["estimated_rt60"]
    t60_overall = config["t60_overall"]
    t60_bands = config["t60_bands"]

    # Apply correction as defined in FAST-RIR
    crr = 0
    
    # Create embedding array with extended RT60 information
    embedding = np.array([
        mic_pos[0], mic_pos[1], mic_pos[2],
        source_pos[0], source_pos[1], source_pos[2],
        room_dims[0], room_dims[1], room_dims[2],
        rt60 + crr,
        t60_overall,
        *t60_bands  # Unpack the frequency band RT60 values
    ])

    # Normalize: (embedding / 5) - 1
    normalized_embedding = (embedding / 5) - 1

    return normalized_embedding

def generate_histograms_from_csv(csv_path, output_dir="histograms"):
    """Generate histograms of room volumes and RT60 values from a configuration CSV file.
    
    Args:
        csv_path: Path to the CSV file containing room configurations
        output_dir: Directory to save the histograms
    """
    print(f"Generating histograms from {csv_path}...")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load configurations
    configs = parse_room_configs_from_csv(csv_path)
    print(f"Loaded {len(configs)} configurations")
    
    # Extract room volumes
    room_volumes = []
    for config in configs:
        room_dims = config['room_dims']
        volume = room_dims[0] * room_dims[1] * room_dims[2]
        room_volumes.append(volume)
    
    # Generate room volume histogram
    plt.figure(figsize=(10, 6))
    plt.hist(room_volumes, bins=30, alpha=0.7, color='blue')
    plt.xlabel('Room Volume (m³)')
    plt.ylabel('Count')
    plt.title('Distribution of Room Volumes')
    plt.grid(True, alpha=0.3)
    
    # Add statistics to the plot
    mean_volume = np.mean(room_volumes)
    median_volume = np.median(room_volumes)
    min_volume = np.min(room_volumes)
    max_volume = np.max(room_volumes)
    
    stats_text = f"Mean: {mean_volume:.2f} m³\nMedian: {median_volume:.2f} m³\n"
    stats_text += f"Min: {min_volume:.2f} m³\nMax: {max_volume:.2f} m³"
    
    plt.annotate(stats_text, xy=(0.95, 0.95), xycoords='axes fraction', 
                 horizontalalignment='right', verticalalignment='top',
                 bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))
    
    plt.savefig(f"{output_dir}/room_volume_histogram.png")
    plt.close()
    
    print(f"Room volume histogram saved to {output_dir}")
    print(f"Room volume statistics: Mean={mean_volume:.2f}, Median={median_volume:.2f}, "
          f"Min={min_volume:.2f}, Max={max_volume:.2f}")
    
    # Generate a few sample RIRs to get RT60 values if they're not in the config
    if "estimated_rt60" not in configs[0]:
        print("Generating sample RIRs to estimate RT60 values...")
        
        rt60_values = []
        # Sample a subset of configs to avoid long processing time
        sample_size = min(100, len(configs))
        sample_configs = random.sample(configs, sample_size)
        
        for config in tqdm(sample_configs):
            rir, updated_config = generate_rir(config)
            rt60_values.append(updated_config["estimated_rt60"])
    else:
        # Extract RT60 values if they're already in the configs
        rt60_values = [config.get("estimated_rt60") for config in configs if "estimated_rt60" in config]
    
    if rt60_values:
        # Generate RT60 histogram
        plt.figure(figsize=(10, 6))
        plt.hist(rt60_values, bins=30, alpha=0.7, color='green')
        plt.xlabel('RT60 (seconds)')
        plt.ylabel('Count')
        plt.title('Distribution of RT60 Values')
        plt.grid(True, alpha=0.3)
        
        # Add statistics to the plot
        mean_rt60 = np.mean(rt60_values)
        median_rt60 = np.median(rt60_values)
        min_rt60 = np.min(rt60_values)
        max_rt60 = np.max(rt60_values)
        
        stats_text = f"Mean: {mean_rt60:.3f} s\nMedian: {median_rt60:.3f} s\n"
        stats_text += f"Min: {min_rt60:.3f} s\nMax: {max_rt60:.3f} s"
        
        plt.annotate(stats_text, xy=(0.95, 0.95), xycoords='axes fraction', 
                     horizontalalignment='right', verticalalignment='top',
                     bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))
        
        plt.savefig(f"{output_dir}/rt60_histogram.png")
        plt.close()
        
        print(f"RT60 histogram saved to {output_dir}")
        print(f"RT60 statistics: Mean={mean_rt60:.3f}, Median={median_rt60:.3f}, "
              f"Min={min_rt60:.3f}, Max={max_rt60:.3f}")
    else:
        print("No RT60 data available for histogram")

def generate_histograms_from_csv2(output_dir, metadata_dir, output_hist_dir="histograms"):
    """Generate histograms of room volumes and RT60 values from existing RIR metadata files.
    
    Args:
        output_dir: Directory containing the generated RIR dataset
        metadata_dir: Directory containing the metadata JSON files (usually output_dir/metadata)
        output_hist_dir: Directory to save the histograms
    """
    print(f"Generating histograms from existing metadata in {metadata_dir}...")
    
    # Create output directory
    os.makedirs(output_hist_dir, exist_ok=True)
    
    # Find all metadata JSON files
    if not os.path.exists(metadata_dir):
        print(f"Error: Metadata directory {metadata_dir} does not exist")
        return
    
    metadata_files = [f for f in os.listdir(metadata_dir) if f.endswith('.json')]
    if not metadata_files:
        print(f"Error: No metadata JSON files found in {metadata_dir}")
        return
    
    print(f"Found {len(metadata_files)} metadata files")
    
    # Load data from metadata files
    room_volumes = []
    rt60_values = []
    room_dims_data = []
    
    for metadata_file in tqdm(metadata_files, desc="Loading metadata"):
        metadata_path = os.path.join(metadata_dir, metadata_file)
        try:
            with open(metadata_path, 'r') as f:
                config = json.load(f)
            
            # Extract room dimensions and calculate volume
            room_dims = config['room_dims']
            volume = room_dims[0] * room_dims[1] * room_dims[2]
            room_volumes.append(volume)
            room_dims_data.append(room_dims)
            
            # Extract RT60 value
            if 'estimated_rt60' in config:
                rt60_values.append(config['estimated_rt60'])
            
        except Exception as e:
            print(f"Warning: Could not read metadata file {metadata_file}: {e}")
            continue
    
    if not room_volumes:
        print("Error: No valid metadata found")
        return
    
    # Generate room volume histogram
    plt.figure(figsize=(10, 6))
    plt.hist(room_volumes, bins=30, alpha=0.7, color='blue')
    plt.xlabel('Room Volume (m³)')
    plt.ylabel('Count')
    plt.title('Distribution of Room Volumes')
    plt.grid(True, alpha=0.3)
    
    # Add statistics to the plot
    mean_volume = np.mean(room_volumes)
    median_volume = np.median(room_volumes)
    min_volume = np.min(room_volumes)
    max_volume = np.max(room_volumes)
    
    stats_text = f"Mean: {mean_volume:.2f} m³\nMedian: {median_volume:.2f} m³\n"
    stats_text += f"Min: {min_volume:.2f} m³\nMax: {max_volume:.2f} m³"
    
    plt.annotate(stats_text, xy=(0.95, 0.95), xycoords='axes fraction', 
                 horizontalalignment='right', verticalalignment='top',
                 bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))
    
    plt.savefig(f"{output_hist_dir}/room_volume_histogram.png")
    plt.close()
    
    print(f"Room volume histogram saved to {output_hist_dir}")
    print(f"Room volume statistics: Mean={mean_volume:.2f}, Median={median_volume:.2f}, "
          f"Min={min_volume:.2f}, Max={max_volume:.2f}")
    
    # Generate individual room dimension histograms
    if room_dims_data:
        room_dims_array = np.array(room_dims_data)
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        dim_names = ['Length', 'Width', 'Height']
        dim_units = ['m', 'm', 'm']
        
        for i, (ax, name, unit) in enumerate(zip(axes, dim_names, dim_units)):
            ax.hist(room_dims_array[:, i], bins=20, alpha=0.7, color=['red', 'green', 'orange'][i])
            ax.set_xlabel(f'Room {name} ({unit})')
            ax.set_ylabel('Count')
            ax.set_title(f'Distribution of Room {name}')
            ax.grid(True, alpha=0.3)
            
            # Add statistics
            mean_dim = np.mean(room_dims_array[:, i])
            median_dim = np.median(room_dims_array[:, i])
            min_dim = np.min(room_dims_array[:, i])
            max_dim = np.max(room_dims_array[:, i])
            
            stats_text = f"Mean: {mean_dim:.2f}\nMedian: {median_dim:.2f}\nMin: {min_dim:.2f}\nMax: {max_dim:.2f}"
            ax.annotate(stats_text, xy=(0.95, 0.95), xycoords='axes fraction', 
                       horizontalalignment='right', verticalalignment='top',
                       bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(f"{output_hist_dir}/room_dimensions_histogram.png")
        plt.close()
        
        print(f"Room dimensions histogram saved to {output_hist_dir}")
    
    # Generate RT60 histogram if data is available
    if rt60_values:
        plt.figure(figsize=(10, 6))
        plt.hist(rt60_values, bins=30, alpha=0.7, color='green')
        plt.xlabel('RT60 (seconds)')
        plt.ylabel('Count')
        plt.title('Distribution of RT60 Values')
        plt.grid(True, alpha=0.3)
        
        # Add statistics to the plot
        mean_rt60 = np.mean(rt60_values)
        median_rt60 = np.median(rt60_values)
        min_rt60 = np.min(rt60_values)
        max_rt60 = np.max(rt60_values)
        
        stats_text = f"Mean: {mean_rt60:.3f} s\nMedian: {median_rt60:.3f} s\n"
        stats_text += f"Min: {min_rt60:.3f} s\nMax: {max_rt60:.3f} s"
        
        plt.annotate(stats_text, xy=(0.95, 0.95), xycoords='axes fraction', 
                     horizontalalignment='right', verticalalignment='top',
                     bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))
        
        plt.savefig(f"{output_hist_dir}/rt60_histogram.png")
        plt.close()
        
        print(f"RT60 histogram saved to {output_hist_dir}")
        print(f"RT60 statistics: Mean={mean_rt60:.3f}, Median={median_rt60:.3f}, "
              f"Min={min_rt60:.3f}, Max={max_rt60:.3f}")
        
        # Additional RT60 analysis
        rt60_filtered = [rt60 for rt60 in rt60_values if rt60 <= 0.75]
        if len(rt60_filtered) != len(rt60_values):
            print(f"Note: {len(rt60_values) - len(rt60_filtered)} RIRs had RT60 > 0.75s")
            print(f"Filtered dataset contains {len(rt60_filtered)} RIRs with RT60 ≤ 0.75s")
    else:
        print("No RT60 data found in metadata files")
    
    print(f"Total processed: {len(room_volumes)} RIR configurations")

def main():
    parser = argparse.ArgumentParser(description="Generate room impulse responses using pygsound.")
    parser.add_argument("--num_rirs", type=int, default=100, help="Number of RIRs to generate")
    parser.add_argument("--output_dir", type=str, default="rirs", help="Output directory for RIRs")
    parser.add_argument("--sample_rate", type=int, default=16000, help="Sample rate for RIRs")
    parser.add_argument("--max_order", type=int, default=50, help="Maximum reflection order")
    parser.add_argument("--pairs_per_room", type=int, default=3,
                      help="Number of source/receiver pairs per unique room dimension")
    parser.add_argument("--config_csv", type=str, default="",
                      help="Path to CSV file with room configurations (if empty, configs will be generated)")
    parser.add_argument("--save_config_csv", type=str, default="",
                      help="Path to save generated room configurations as CSV (if empty, no CSV will be saved)")
    parser.add_argument("--randomize", action="store_true",
                      help="Randomize the order of room configurations")
    parser.add_argument("--random_seed", type=int, default=None,
                      help="Random seed for reproducible randomization")
    parser.add_argument("--config_only", action="store_true",
                      help="Generate only room configurations without RIRs")
    parser.add_argument("--start_index", type=int, default=0,
                      help="Starting index for processing configurations from CSV (for batch processing)")
    parser.add_argument("--batch_size", type=int, default=0,
                      help="Number of configurations to process in this batch (0 = all remaining)")
    parser.add_argument("--generate_histograms", action="store_true",
                      help="Generate histograms of room volumes and RT60 values from a configuration CSV file")
    parser.add_argument("--histograms_dir", type=str, default="histograms",
                      help="Directory to save the histograms")
    args = parser.parse_args()

    print("TEST")

    # Set global random seed if provided
    if args.random_seed is not None:
        np.random.seed(args.random_seed)
        random.seed(args.random_seed)
        print(f"Using random seed: {args.random_seed}")

    # Generate histograms if requested
    if args.generate_histograms:
        if not args.config_csv or not os.path.exists(args.config_csv):
            print("Error: --config_csv must be specified and must exist to generate histograms")
            return
        
        generate_histograms_from_csv2(args.output_dir, args.config_csv, args.histograms_dir)
        if not args.config_only and args.batch_size == 0 and args.start_index == 0:
            # If we're only generating histograms and not processing configurations
            return

    # Create directory for configs if in config-only mode
    if args.config_only:
        if not args.save_config_csv:
            print("Error: --save_config_csv must be specified in --config_only mode")
            return
        
        print(f"DEBUG: Will save config to: {args.save_config_csv}")
        print(f"DEBUG: Absolute path: {os.path.abspath(args.save_config_csv)}")
        
        # Create the directory for the CSV if it doesn't exist
        csv_dir = os.path.dirname(args.save_config_csv)
        if csv_dir:
            try:
                os.makedirs(csv_dir, exist_ok=True)
                print(f"DEBUG: Created directory: {csv_dir}")
            except Exception as e:
                print(f"ERROR creating directory: {str(e)}")
        else:
            print(f"DEBUG: No directory specified, will write to current directory")
            print(f"DEBUG: Current directory: {os.getcwd()}")
            print(f"DEBUG: Have write permission: {os.access('.', os.W_OK)}")
    
    # Create output directory structure compatible with FastRirDataset
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "RIR"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "train"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "metadata"), exist_ok=True)  # For additional info

    # Room dimension constraints (m)
    room_min_dims = np.array([8.0, 6.0, 2.5])  # Length, Width, Height
    room_max_dims = np.array([11.0, 8.0, 3.5])  # Length, Width, Height

    # Either load configurations from CSV or generate them
    if args.config_csv and os.path.exists(args.config_csv):
        print(f"Loading room configurations from {args.config_csv}...")
        all_configs = parse_room_configs_from_csv(args.config_csv)
        
        # Apply batch processing if using a config file
        if args.start_index >= len(all_configs):
            print(f"Error: start_index {args.start_index} is beyond the number of configurations ({len(all_configs)})")
            return
        
        end_index = len(all_configs)
        if args.batch_size > 0:
            end_index = min(args.start_index + args.batch_size, len(all_configs))
        
        configs = all_configs[args.start_index:end_index]
        print(f"Processing configurations {args.start_index} to {end_index-1} (total: {len(configs)} of {len(all_configs)})")
    else:
        print("Generating linearly spaced room configurations...")
        try:
            configs = generate_linearly_spaced_room_configs(
                args.num_rirs,
                room_min_dims,
                room_max_dims,
                pairs_per_room=args.pairs_per_room,
                csv_output_path=args.save_config_csv if args.save_config_csv else None,
                randomize=args.randomize,
                random_seed=args.random_seed
            )
            print(f"DEBUG: Generated {len(configs)} configurations")
        except Exception as e:
            print(f"ERROR during configuration generation: {str(e)}")
            import traceback
            traceback.print_exc()
            return
    
    # If config_only mode, we're done here
    if args.config_only:
        if args.save_config_csv:
            if os.path.exists(args.save_config_csv):
                print(f"DEBUG: Confirmed CSV file exists: {args.save_config_csv}")
                print(f"DEBUG: File size: {os.path.getsize(args.save_config_csv)} bytes")
            else:
                print(f"DEBUG: WARNING - CSV file does not exist: {args.save_config_csv}")
                print(f"DEBUG: Current directory contents: {os.listdir('.')}")
        print(f"Configuration generation complete. {len(configs)} configurations saved to {args.save_config_csv}")
        return
    
    # Store file names and embeddings for FastRirDataset format
    file_names = []
    embeddings = {}

    # Generate RIRs
    print(f"Generating {len(configs)} RIRs...")
    valid_count = 0
    skipped_count = 0
    
    for i, config in enumerate(tqdm(configs)):
        # Generate RIR
        rir, updated_config = generate_rir(
            config,
            sample_rate=args.sample_rate,
            max_order=args.max_order
        )
        
        # Skip RIRs with RT60 > 0.75s
        if updated_config["estimated_rt60"] > 0.75:
            skipped_count += 1
            continue
            
        # Generate file ID in the format used by FastRIR
        # Use valid_count instead of i to ensure sequential numbering for included RIRs
        file_id = f"rir_{(args.start_index + valid_count):05d}"
        file_names.append(file_id)
        valid_count += 1

        # Save RIR as WAV file (using the FastRIR directory structure)
        audio_path = os.path.join(args.output_dir, "RIR", f"{file_id}.wav")
        sf.write(audio_path, rir, args.sample_rate)

        # Format the config into a FAST-RIR compatible embedding
        embedding = format_fast_rir_embedding(updated_config)
        embeddings[file_id] = embedding

        # Convert NumPy arrays to lists for JSON serialization
        json_safe_config = {}
        for key, value in updated_config.items():
            if isinstance(value, np.ndarray):
                json_safe_config[key] = value.tolist()
            elif isinstance(value, np.float32) or isinstance(value, np.float64):
                json_safe_config[key] = float(value)
            elif isinstance(value, np.int32) or isinstance(value, np.int64):
                json_safe_config[key] = int(value)
            else:
                json_safe_config[key] = value

        # Save metadata as JSON (for reference)
        metadata_path = os.path.join(args.output_dir, "metadata", f"{file_id}.json")
        with open(metadata_path, 'w') as f:
            json.dump(json_safe_config, f, indent=2)
    
    # Print statistics about filtered RIRs
    if skipped_count > 0:
        print(f"Skipped {skipped_count} RIRs with RT60 > 0.75s ({skipped_count/len(configs):.1%} of total)")
        print(f"Included {valid_count} RIRs with RT60 ≤ 0.75s")
            
    # Save filenames.pickle and embeddings.pickle for FastRirDataset
    # If we're doing batch processing, we need to load existing pickle files and update them
    if args.config_csv and args.start_index > 0 and not args.config_only:
        filenames_path = os.path.join(args.output_dir, "train", "filenames.pickle")
        embeddings_path = os.path.join(args.output_dir, "train", "embeddings.pickle")
        
        # Load existing data if files exist
        existing_file_names = []
        existing_embeddings = {}
        
        if os.path.exists(filenames_path):
            with open(filenames_path, 'rb') as f:
                existing_file_names = pickle.load(f)
                
        if os.path.exists(embeddings_path):
            with open(embeddings_path, 'rb') as f:
                existing_embeddings = pickle.load(f)
        
        # Merge existing and new data
        all_file_names = existing_file_names + file_names
        all_embeddings = {**existing_embeddings, **embeddings}
        
        # Save the merged data
        with open(filenames_path, 'wb') as f:
            pickle.dump(all_file_names, f)
            
        with open(embeddings_path, 'wb') as f:
            pickle.dump(all_embeddings, f)
            
        print(f"Updated dataset files with {len(file_names)} new RIRs (total: {len(all_file_names)})")
    else:
        # Save new files without merging (initial batch or full dataset)
        with open(os.path.join(args.output_dir, "train", "filenames.pickle"), 'wb') as f:
            pickle.dump(file_names, f)

        with open(os.path.join(args.output_dir, "train", "embeddings.pickle"), 'wb') as f:
            pickle.dump(embeddings, f)
            
        print(f"Generated {len(configs)} RIRs in {args.output_dir}")
        
    print(f"Data is formatted for use with FastRirDataset")
    
    # If batch processing, print information about remaining configurations
    if args.config_csv and args.batch_size > 0 and end_index < len(all_configs):
        remaining = len(all_configs) - end_index
        next_start = end_index
        print(f"\nRemaining configurations: {remaining}")
        print(f"To process the next batch, run with: --start_index {next_start} --batch_size {args.batch_size}")

if __name__ == "__main__":
    main()
