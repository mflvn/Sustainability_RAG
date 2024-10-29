import os

def get_folder_names(directory):
    folder_names = []
    items = os.listdir(directory)
    
    for item in items:
        full_path = os.path.join(directory, item)
        if os.path.isdir(full_path):
            folder_names.append(item)
    
    return folder_names

def write_list_to_file(folder_list, output_file):
    with open(output_file, 'w') as file:
        for folder in folder_list:
            file.write(folder + '\n')

folder_list = get_folder_names('pdf_images')
output_file = 'industries.txt'
write_list_to_file(folder_list, output_file)

print(f"Folder names have been written to {output_file}")