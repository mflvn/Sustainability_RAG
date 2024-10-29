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


def read_images_from_folder(folder_path: str) -> Dict[str, List[Dict]]:
    table_images = []
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
            if "TABLE" in filename:
                table_images.append(image_dict)

            all_images.append(image_dict)

    return {"table_images": table_images, "other_images": all_images}


# Tool for processing table pages
table_pages_processor = {
    "name": "process_table_pages",
    "description": "Process the pages containing tables and extract specified information",
    "input_schema": {
        "type": "object",
        "properties": {
            "sustainability_metrics_table": {
                "type": "string",
                "description": "Markdown of sustainability metrics table",
            },
            "activity_metrics_table": {
                "type": "string",
                "description": "Markdown of activity metrics table, if present",
            },
            "report_title": {"type": "string", "description": "Title of the report"},
            "industry": {"type": "string", "description": "Industry of the report"},
        },
        "required": ["sustainability_metrics_table", "report_title", "industry"],
    },
}

# Tool for processing individual pages
individual_page_processor = {
    "name": "process_individual_page",
    "description": "Process a single page of a PDF document and extract specified information",
    "input_schema": {
        "type": "object",
        "properties": {
            "text_content": {
                "type": "string",
                "description": "All text content, specifically in markdown format, excluding the tables Table 1. Sustainability Disclosure Topics & Metrics and Table 2. Activity Metrics." ,
            },
            "page_number": {
                "type": "integer",
                "description": "Page number of the processed page in the footer",
            },
        },
        "required": ["text_content", "page_number"],
    },
}


def process_table_pages(images: List[Dict]) -> Dict:
    response = client.messages.create(
        model="claude-3-5-sonnet-20240620",
        max_tokens=2048,
        tools=[table_pages_processor],
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Process these PDF pages and extract the following information:\n"
                        "1. Markdown of the entire sustainability metrics table, which has these five columns: TOPIC, METRIC, CATEGORY, UNIT OF MEASURE, CODE (make sure to exclude any text that has been crossed out, but include any underlined text!)\n"
                        "2. Markdown of the activity metrics table, if present, which has these four columns: ACTIVITY METRIC, CATEGORY, UNIT OF MEASURE, CODE (make sure to exclude any text that has been crossed out, but include any underlined text!). If there is no activity metrics table, omit this field.\n"
                        "3. Report title\n"
                        "4. Industry\n\n"
                        "Exclude any headers (APPENDIX B OF [DRAFT] IFRS S2 CLIMATE-RELATED DISCLOSURES) or (EXPOSURE DRAFT—MARCH 2022) and footers (© 2022 SASB, part of Value Reporting Foundation. All rights reserved.) of the pages, but include footnotes. Exclude any instances of 'continued...' or '...continued'. Exclude any text that has been crossed out, but include any underlined text. Note that tables may continue across pages. Provide the output in the specified schema.",
                    },
                ]
                + images,
            }
        ],
    )
    return response.content[1].input


def process_individual_page(
    image: Dict, report_title: str, industry: str, page_number: int
) -> Dict:
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
                        "text": f"""Process this PDF page and extract all text content, excluding the tables Table 1. Sustainability Disclosure Topics & Metrics and Table 2. Activity Metrics, but including any other tables that appear. Exclude any text that has been crossed out, but include any underlined text. Exclude any headers (APPENDIX B OF [DRAFT] IFRS S2 CLIMATE-RELATED DISCLOSURES) or (EXPOSURE DRAFT—MARCH 2022) and footers (© 2022 SASB, part of Value Reporting Foundation. All rights reserved.) of the pages, but include footnotes. Exclude the section header titled 'Sustainability Disclosure Topics & Metrics'. Exclude subsections 'Table 1. Sustainability Disclosure Topics & Metrics' and 'Table 2. Activity Metrics'. Exclude any instances of 'continued...' or '...continued'. The report title is '{report_title}', the industry is '{industry}' (but DO NOT include the report title and industry as sections in the markdown). Provide the output in the specified schema.""",
                    },
                    image,
                ],
            }
        ],
    )
    return response.content[1].input


def process_single_pdf(pdf_folder: str, output_folder: str):
    images = read_images_from_folder(pdf_folder)

    # Process table pages
    table_pages_result = process_table_pages(images["table_images"])
    print(f"Table pages processed for {pdf_folder}")

    # Create PDF-specific output folder
    pdf_output_folder = os.path.join(output_folder, os.path.basename(pdf_folder))
    os.makedirs(pdf_output_folder, exist_ok=True)

    # Save JSON for tables
    with open(os.path.join(pdf_output_folder, "tables.json"), "w") as f:
        json.dump(table_pages_result, f)

    report_title = table_pages_result["report_title"]
    industry = table_pages_result["industry"]

    # Prepare markdown content
    markdown_content = f"# IFRS S2 Climate-related Disclosures Appendix B Industry-based disclosure requirements: {report_title}\n\n"
    markdown_content += f"Industry: {industry}\n\n"
    markdown_content += "## Table 1. Sustainability Disclosure Topics & Metrics\n\n"
    markdown_content += table_pages_result["sustainability_metrics_table"] + "\n\n"
    # save individual tables
    with open(
        os.path.join(pdf_output_folder, "sustainability_metrics_table.md"), "w"
    ) as f:
        f.write(table_pages_result["sustainability_metrics_table"])

    if "activity_metrics_table" in table_pages_result:
        markdown_content += "## Table 2. Activity Metrics\n\n"
        markdown_content += table_pages_result["activity_metrics_table"] + "\n\n"
        with open(
            os.path.join(pdf_output_folder, "activity_metrics_table.md"), "w"
        ) as f:
            f.write(table_pages_result["activity_metrics_table"])

    # markdown_content += "## Page Content\n\n"

    # Process individual pages
    for i, image in enumerate(images["other_images"], start=1):
        page_result = process_individual_page(image, report_title, industry, i)
        page_number = page_result["page_number"]
        # Save JSON for each page
        with open(os.path.join(pdf_output_folder, f"page{page_number}.json"), "w") as f:
            json.dump(page_result, f)

        # Add page content to markdown
        markdown_content += f"#### Page {page_number}\n\n"
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
main_folder = "./pdf_images"
output_folder = "./markdown_output"
process_all_pdfs(main_folder, output_folder)
