import os
from PyPDF2 import PdfReader, PdfWriter

# Specify the directory containing the PDF files
directory = "./standards_reports"
out_directory = "./standards_modified_reports"

# Loop through all the files in the directory that end with .pdf
for filename in os.listdir(directory):
    if filename.endswith(".pdf"):
        file_path = os.path.join(directory, filename)
        
        # Create a PDF reader object
        reader = PdfReader(file_path)
        
        # Create a PDF writer object for the output
        writer = PdfWriter()
        
        # Add all pages except the first three to the writer object
        for page in range(1, len(reader.pages)):
            writer.add_page(reader.pages[page])
        
        # Output file path
        output_file_path = os.path.join(out_directory, "modified_" + filename)
        
        # Write out the modified PDF
        with open(output_file_path, "wb") as f:
            writer.write(f)
        
        print(f"Removed first four pages from '{filename}' and saved as '{output_file_path}'")
