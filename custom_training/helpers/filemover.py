import os
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed

def move_files(src_dir, dest_dir):
    count = 0
    for root, _, files in os.walk(src_dir, topdown=False):
        for file in files:
            src_file_path = os.path.join(root, file)
            dest_file_path = os.path.join(dest_dir, file)
            shutil.move(src_file_path, dest_file_path)
            count += 1
            print(f"Moved: {src_file_path} -> {dest_file_path}")
        # After moving all files, remove the directory
        if root != src_dir:
            os.rmdir(root)
            print(f"Deleted directory: {root}")
    print(f"moved {count} files")


def main(main_dir):
    # Create a ThreadPoolExecutor with a reasonable number of threads
    with ThreadPoolExecutor(max_workers=10) as executor:
        # Walk through the main directory to find sub-directories
        for root, dirs, _ in os.walk(main_dir):
            for dir in dirs:
                src_dir = os.path.join(root, dir)
                # Submit the move_files function to the executor
                executor.submit(move_files, src_dir, main_dir)

if __name__ == "__main__":
    main_directory = "C:\\AI\\datasets\\nate"  # Replace with your main directory path
    main(main_directory)