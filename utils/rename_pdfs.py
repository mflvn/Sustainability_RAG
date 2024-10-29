import os

# Specify the directory containing the PDF files
directory = "./modified_reports"

# Define the prefix to remove
prefix_to_remove = "modified_"

# Loop through all the files in the directory
for filename in os.listdir(directory):
    if filename.startswith(prefix_to_remove):
        # Generate the new filename by removing the prefix
        new_filename = filename[len(prefix_to_remove):]
        
        # Generate the full old and new file paths
        old_file_path = os.path.join(directory, filename)
        new_file_path = os.path.join(directory, new_filename)
        
        # Rename the file
        os.rename(old_file_path, new_file_path)
        print(f"Renamed '{filename}' to '{new_filename}'")
