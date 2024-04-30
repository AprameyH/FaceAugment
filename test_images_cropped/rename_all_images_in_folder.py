import os

# Function to rename all images in a specified folder
def rename_images_in_folder(folder_path, prefix):
    # Get a list of all files in the specified folder
    files = os.listdir(folder_path)
    
    # Filter the list to include only files with common image extensions
    image_extensions = ('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff')  # Add more if needed
    images = [file for file in files if file.lower().endswith(image_extensions)]
    
    # Sort images to maintain consistent renaming order
    images.sort()
    
    # Rename each image in the folder
    for index, image in enumerate(images):
        # Determine the new file name based on the index
        new_file_name = f"{prefix}{index}" + os.path.splitext(image)[1]  # Preserve the original file extension
        
        # Get the full paths for the old and new file names
        old_path = os.path.join(folder_path, image)
        new_path = os.path.join(folder_path, new_file_name)
        
        # Rename the file
        os.rename(old_path, new_path)
    
    print("All images have been renamed.")

# Example usage
folder_path = 'Ayumi_Hamasaki'  # Update with the correct path to your folder
rename_images_in_folder(folder_path, "ah")
