import os
import re

import fitz  # PyMuPDF

directory = "./standards_modified_reports"
out_directory = "./standards_pdf_images"


# Function to check if a page contains a table
def page_contains_table(page):
    text = page.get_text()

    # # Check for "Table 1" or "Table 2"
    # if re.search(r'Table 1\.\s*Sustainability Disclosure Topics & Metrics', text, re.IGNORECASE) or \
    #    re.search(r'Table 2\.\s*Activity Metrics', text, re.IGNORECASE):
    #     return True

    # # Check for "...continued" or "continued..." anywhere in the text
    # if "...continued" in text or "continued..." in text:
    #     return True

    # Check for typical table content
    # This looks for lines with multiple pipe characters (|) or multiple tab characters
    lines = text.split("\n")
    for line in lines:
        if line.count("|") > 2 or line.count("\t") > 2:
            return True

    return False


for filename in os.listdir(directory):
    if filename.endswith(".pdf")    :
        file_path = os.path.join(directory, filename)

        doc = fitz.open(file_path)

        image_folder = os.path.join(out_directory, filename[:-4])
        if not os.path.exists(image_folder):
            os.makedirs(image_folder)

        zoom_x = 2.5
        zoom_y = 2.5
        mat = fitz.Matrix(zoom_x, zoom_y)

        for page_number in range(len(doc)):
            page = doc.load_page(page_number)

            # Check if page contains a table
            has_table = page_contains_table(page)

            # Create filename
            if has_table:
                image_filename = os.path.join(
                    image_folder, f"page_{page_number + 1}_TABLE.png"
                )
            else:
                image_filename = os.path.join(
                    image_folder, f"page_{page_number + 1}.png"
                )

            # Render and save the page as an image
            pix = page.get_pixmap(matrix=mat)
            pix.save(image_filename)
            print(f"Saved '{image_filename}'")

        doc.close()
