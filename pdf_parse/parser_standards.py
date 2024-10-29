import base64
import json
import os
import re
from typing import Dict, List

import anthropic

# Initialize Anthropic client
client = anthropic.Anthropic(
    api_key=os.getenv("CLAUDE_API_KEY")
)


def natural_sort_key(s):
    return [int(c) if c.isdigit() else c.lower() for c in re.split(r"(\d+)", s)]


def read_images_from_folder(folder_path: str) -> List[Dict]:
    all_images = []

    # Get all image files and sort them
    image_files = [
        f
        for f in os.listdir(folder_path)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ]
    image_files.sort(key=natural_sort_key)

    for filename in image_files:
        file_path = os.path.join(folder_path, filename)
        with open(file_path, "rb") as image_file:
            image_data = image_file.read()
            base64_image = base64.b64encode(image_data).decode("utf-8")
            image_dict = {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": f"image/{filename.split('.')[-1]}",
                    "data": base64_image,
                },
            }
            all_images.append(image_dict)

    return all_images


# Tool for processing individual pages
individual_page_processor = {
    "name": "process_individual_page",
    "description": "Process a single page of a PDF document and extract specified information",
    "input_schema": {
        "type": "object",
        "properties": {
            "text_content": {
                "type": "string",
                "description": "All text content, specifically in markdown format." ,
            },
            "page_number": {
                "type": "integer",
                "description": "Page number of the processed page in the footer",
            },
        },
        "required": ["text_content", "page_number"],
    },
}


def process_individual_page(image: Dict, page_number: int) -> Dict:
    response = client.messages.create(
        model="claude-3-5-sonnet-20240620",
        max_tokens=2048,
        tools=[individual_page_processor],
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"""Process this PDF page and extract all text information, maintaining format of tables where terms are defined.\n
                        Exclude any headers (IFRS S2 CLIMATE-RELATED DISCLOSURES—JUNE 2023) or (IFRS SUSTAINABILITY DISCLOSURE STANDARDS) and footers (© IFRS Foundation) of the pages, but include footnotes. Exclude any text that has been crossed out, but include any underlined text. Provide the output in the specified schema.""",
                    },
                    image,
                ],
            }
        ],
    )
    return response.content[1].input


def process_single_pdf(pdf_folder: str, output_folder: str):
    images = read_images_from_folder(pdf_folder)

    # Create PDF-specific output folder
    pdf_output_folder = os.path.join(output_folder, os.path.basename(pdf_folder))
    os.makedirs(pdf_output_folder, exist_ok=True)

    markdown_content = "# PDF Content\n\n"

    # Process individual pages
    for i, image in enumerate(images, start=1):
        page_result = process_individual_page(image, i)
        page_number = page_result["page_number"]
        # Save JSON for each page
        with open(os.path.join(pdf_output_folder, f"page{page_number}.json"), "w") as f:
            json.dump(page_result, f)

        # Add page content to markdown
        markdown_content += f"## Page {page_number}\n\n"
        markdown_content += page_result["text_content"] + "\n\n"

    # Save all content as a single markdown file
    with open(
        os.path.join(pdf_output_folder, "full_content.md"), "w", encoding="utf-8"
    ) as f:
        f.write(markdown_content)

    print(f"All content saved for {pdf_folder}")


def process_all_pdfs(main_folder: str, output_folder: str):
    pdf_folders = sorted(os.listdir(main_folder), key=natural_sort_key)
    for pdf_folder in pdf_folders:
        pdf_path = os.path.join(main_folder, pdf_folder)
        if os.path.isdir(pdf_path):
            print(f"Processing PDF folder: {pdf_folder}")
            process_single_pdf(pdf_path, output_folder)
            print(f"Finished processing {pdf_folder}")


# Usage
main_folder = "./standards_pdf_images_test"
output_folder = "./standards_markdown_output"
process_all_pdfs(main_folder, output_folder)