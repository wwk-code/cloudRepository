import os
import sys

def find_large_files(directory, size_limit_mb=100):
    size_limit_bytes = size_limit_mb * 1024 * 1024  # 转换为字节
    for root, dirs, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            try:
                file_size = os.path.getsize(file_path)
                if file_size > size_limit_bytes:
                    print(f"Large file found: {file_path} ({file_size / (1024 * 1024):.2f} MB)")
            except OSError as e:
                print(f"Error accessing {file_path}: {e}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python find_large_files.py <directory_path>")
        sys.exit(1)

    directory_path = sys.argv[1]
    
    if os.path.isdir(directory_path):
        find_large_files(directory_path)
    else:
        print(f"The path '{directory_path}' is not a valid directory.")