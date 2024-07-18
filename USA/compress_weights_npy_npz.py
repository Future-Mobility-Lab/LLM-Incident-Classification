import numpy as np
import os
import glob
from tqdm import tqdm
import time

def compress_npy_to_npz(npy_file_path, npz_file_path=None):
    if npz_file_path is None:
        npz_file_path = npy_file_path.replace('.npy', '.npz')
    data = np.load(npy_file_path)
    np.savez_compressed(npz_file_path, data=data)
    print(f"Compressed and saved '{os.path.basename(npy_file_path)}' as '{os.path.basename(npz_file_path)}'")

def process_directory(directory):
    # Get all .npy files in the directory
    npy_files = glob.glob(os.path.join(directory, '*.npy'))
    
    if not npy_files:
        print("No .npy files found in the directory.")
        return
    
    # Initialize the progress bar
    pbar = tqdm(total=len(npy_files), unit='file', desc="Compressing files")

    # Start time
    start_time = time.time()

    # Compress each .npy file to .npz format
    for npy_file in npy_files:
        compress_npy_to_npz(npy_file)
        pbar.update(1)  # Update the progress bar by one

    # Finish progress bar
    pbar.close()

    # End time
    end_time = time.time()
    time_taken = end_time - start_time
    print(f"Processed {len(npy_files)} files in {time_taken:.2f} seconds.")

# Example usage:
# Process all .npy files in the current directory
current_directory = '.'  # Indicates the current directory
process_directory(current_directory)
