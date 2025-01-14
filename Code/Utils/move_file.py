import shutil
import os

def move_file(source_path, destination_folder):
    """
    Move a file from source path to destination folder.
    
    Args:
        source_path: Path to the source file
        destination_folder: Path to the destination folder
    
    Returns:
        str: Path to the moved file
    """
    # Check if source file exists
    if not os.path.exists(source_path):
        raise FileNotFoundError(f"Source file not found: {source_path}")
    
    # Check if destination folder exists, if not create it
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)
    
    # Get the filename from the source path
    file_name = os.path.basename(source_path)
    
    # Create the destination path
    destination_path = os.path.join(destination_folder, file_name)
    
    # Move the file
    shutil.move(source_path, destination_path)