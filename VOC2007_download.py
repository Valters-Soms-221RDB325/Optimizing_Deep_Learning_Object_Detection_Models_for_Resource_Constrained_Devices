import urllib.request
import tarfile
import os
import shutil

VOC2007_URL = "http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar"
DOWNLOAD_DIR = "."
TAR_FILENAME = "VOCtrainval_06-Nov-2007.tar"
EXTRACT_DIR = "VOC2007_data"

def download_and_extract_voc2007():
    tar_filepath = os.path.join(DOWNLOAD_DIR, TAR_FILENAME)
    extract_path = os.path.join(DOWNLOAD_DIR, EXTRACT_DIR)

    if not os.path.exists(tar_filepath):
        print(f"Downloading {TAR_FILENAME}...")
        try:
            urllib.request.urlretrieve(VOC2007_URL, tar_filepath)
            print("Download complete.")
        except Exception as e:
            print(f"Error downloading: {e}")
            if os.path.exists(tar_filepath):
                os.remove(tar_filepath)
            return
    else:
        print(f"{TAR_FILENAME} already exists. Skipping download.")

    if os.path.exists(tar_filepath):
        print(f"Extracting {TAR_FILENAME}...")
        try:
            if not os.path.exists(extract_path):
                os.makedirs(extract_path)
            
            with tarfile.open(tar_filepath, "r") as tar:
                tar.extractall(path=extract_path)
            print("Extraction complete.")
            print(f"Data extracted to: {os.path.abspath(extract_path)}")
        except Exception as e:
            print(f"Error extracting: {e}")
    else:
        print("Tar file not found. Cannot extract.")

if __name__ == "__main__":
    download_and_extract_voc2007()