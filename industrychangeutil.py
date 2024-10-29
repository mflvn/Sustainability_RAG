import json
import os


def process_json_file(file_path):
    with open(file_path, "r") as f:
        data = json.load(f)

    modified = False
    for item in data:
        if "industry" in item and isinstance(item["industry"], str):
            item["industries"] = [item["industry"]]
            del item["industry"]
            modified = True

    if modified:
        with open(file_path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"Modified: {file_path}")
    else:
        print(f"No changes needed: {file_path}")


def process_directory(directory):
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".json"):
                file_path = os.path.join(root, file)
                try:
                    process_json_file(file_path)
                except json.JSONDecodeError:
                    print(f"Error: Invalid JSON in {file_path}")
                except Exception as e:
                    print(f"Error processing {file_path}: {str(e)}")


if __name__ == "__main__":
    #  process current directory
    process_directory(".")
    print("Processing complete.")
