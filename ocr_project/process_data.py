import argparse
import os
import json

from PIL import Image


def collect_tokens(annotation_file, output_dir, skip_images=None):
    if skip_images is None:
        skip_images = []

    tokens = set()
    with open(annotation_file, 'r') as file:
        with open(os.path.join(output_dir, 'annotations.txt'), 'w') as processed_annotations_file:
            for line in file:
                # skip over description lines
                if line.startswith('#'):
                    continue
                parts = line.strip().split(' ')
                if parts[0] not in skip_images:
                    label = parts[-1]
                    tokens.update(label)
                    processed_annotations_file.write(line)

    tokens = sorted(tokens)
    token_to_index = {token: idx + 1 for idx, token in enumerate(tokens)}

    with open(os.path.join(output_dir, 'tokens.json'), 'w') as json_file:
        json.dump(token_to_index, json_file, indent=4)


def convert_and_save_images(root_dir, new_root_dir):
    skip_images = []
    for root, _, files in os.walk(root_dir):
        for filename in files:
            if filename.endswith('.png'):
                img_path = os.path.join(root, filename)
                try:
                    with Image.open(img_path) as img:
                        gray_img = img.convert('L')
                        new_img_path = os.path.join(new_root_dir, os.path.relpath(img_path, root_dir)).replace('.png', '.jpg')
                        os.makedirs(os.path.dirname(new_img_path), exist_ok=True)
                        gray_img.save(new_img_path, 'JPEG')
                except Exception as e:
                    print("Warning: image has some problems and will be skipped", e)
                    skip_images.append(filename)
                    
    return skip_images


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Download and extract dataset.')

    parser.add_argument('--annotation_file', type=str, required=True, help='Path to annotations file')
    parser.add_argument('--root_dir', type=str, required=True, help='Root directory for dataset')
    parser.add_argument('--processed_data_path', type=str, default='./processed_data', help='Root directory for processed data')

    args = parser.parse_args()

    skip_images = convert_and_save_images(args.root_dir, args.processed_data_path)
    print(f'Successfully processed dataset and saved to {args.processed_data_path}')

    collect_tokens(args.annotation_file, args.processed_data_path, skip_images)
    print(f'Successfully processed annotations and saved tokens.json to {args.processed_data_path}')
