import argparse
import gdown
import os
import zipfile


def download_dataset(file_id, output='dataset.zip'):
    download_url = f'https://drive.google.com/uc?id={file_id}'
    gdown.download(download_url, output, quiet=False)
    return output


def extract_and_cleanup(zip_file, extract_path="./data"):
    """
    Extracts the compressed file and then deletes the compressed file.

    Args:
    - zip_file (str): The path to the compressed file.
    - extract_path (str): The path to the directory where the files will be extracted.

    Returns:
    - extract_path (str): The path to the directory where the files were extracted.
    """
    os.makedirs(extract_path, exist_ok=True)
    
    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        zip_ref.extractall(extract_path)
    
    os.remove(zip_file)
    
    return extract_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Download and extract dataset.')

    parser.add_argument('--file_id', type=str, required=True, help='Google Drive file ID')
    parser.add_argument('--output', type=str, default='dataset.zip', help='Output zip file name')

    parser.add_argument('--extract_path', type=str, default='./data', help='Path to extract the dataset')

    args = parser.parse_args()

    dataset_path = download_dataset(args.file_id, args.output)

    extracted_path = extract_and_cleanup(dataset_path, args.extract_path)

    print(f'Successfully downloaded dataset to {extracted_path}')
