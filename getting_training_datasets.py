import pandas as pd
import os
from PIL import Image
import requests
from io import BytesIO
import json
import argparse

def main(args):
    # Read in the dataset
    dataset = pd.read_csv(args.input_csv)

    training_data = dataset[['image_link', 'mass_prompt', 'ratio_prompt', 'size_text_prompt', 'shorter_prompt', '75_tokens']]

    data_folder = args.data_folder
    os.makedirs(data_folder, exist_ok=True)

    for index, data in training_data.iterrows():
        image_url = data['image_link']

        # Getting the Image and opening it using PIL
        response = requests.get(image_url)
        img = Image.open(BytesIO(response.content))

        # Resize the image to 512x512
        img_resized = img.resize((512, 512))

        # Save the resized image to the 'data' folder
        image_path = os.path.join(data_folder, f'image_{index + 1}.jpg')
        img_resized.save(image_path)

        # Update the dataset with the image path
        training_data.at[index, 'image_path'] = image_path

    # Save the updated DataFrame with image paths
    training_data.to_csv(args.output_csv, index=False)

    updated_training_data = pd.read_csv(args.output_csv)

    metadata_dict = {}

    for index, data in updated_training_data.iterrows():
        image_path = data['image_path'].split("/")[1].split(".")[0]
        metadata = {"tags": "solo, no humans, space, starry night",
                    "caption": data["75_tokens"]}

        metadata_dict[image_path] = metadata

    with open(args.metadata_json, "w") as json_file:
        json.dump(metadata_dict, json_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare Datasets for Training")
    parser.add_argument("--input-csv", type=str, default="training_data_prompts.csv", help="Input CSV file")
    parser.add_argument("--output-csv", type=str, default="updated_training_data_prompts.csv", help="Output CSV file")
    parser.add_argument("--data-folder", type=str, default="data_huggingface", help="Folder for resized images")
    parser.add_argument("--metadata-json", type=str, default="metadata.json", help="Metadata JSON file")
    args = parser.parse_args()
    main(args)
