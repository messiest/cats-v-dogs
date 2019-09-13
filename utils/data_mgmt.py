import os
import shutil
import subprocess
import zipfile

import pandas as pd
from sklearn.model_selection import train_test_split


def download_data():
    # requires that the kaggle cli is installed and configured
    if not os.path.exists(r"data/pixabay.zip"):
        subprocess.check_call(
            [
                'kaggle',
                'datasets',
                'download',
                'ppleskov/cute-cats-and-dogs-from-pixabaycom',
                '--path',
                'data/',
                '--unzip',  # output is still a zipfile, thus the extraction
            ]
        )
    else:
        print("Data files already downloaded.")

    if not os.path.exists(r"data/pixabay/"):
        print("Extracting Data")
        zip_file = zipfile.ZipFile(r'data/pixabay.zip')
        zip_file.extractall(path='data/pixabay/')
        print("Done.")
    else:
        print("Data files already extracted.")


def build_data_dirs():
    df = pd.read_csv(r'data/labels.csv')
    df.drop_duplicates(subset=['url'], inplace=True)

    training, validation = train_test_split(df, test_size=.1)

    training.insert(0, 'split', 'training')
    validation.insert(0, 'split', 'validation')

    df = pd.concat([training, validation])

    for i, row in df.iterrows():
        src = os.path.join('data', 'pixabay', row['type'], str(row['cute']), f"{row['id']}.jpg")
        if row['cute']:
            label = f"cute_{row['type']}"
        else:
            label = row['type']
        dst = os.path.join('data', 'cats-v-dogs', row['split'], label, f"{row['id']}.jpg")

        os.makedirs(os.path.join('data', 'cats-v-dogs', row['split'], label), exist_ok=True)
        print(f"{row['split'].title()}:", src, dst, end="\r")
        shutil.copyfile(src, dst)

    print(f"\n{len(os.listdir(os.path.join('data', 'cats-v-dogs', 'training', 'cats')))} cat training photos")
    print(f"{len(os.listdir(os.path.join('data', 'cats-v-dogs', 'training', 'cute_cats')))} cute cat training photos")
    print(f"{len(os.listdir(os.path.join('data', 'cats-v-dogs', 'training', 'dogs')))} dog training photos")
    print(f"{len(os.listdir(os.path.join('data', 'cats-v-dogs', 'training', 'cute_dogs')))} cute dog training photos")
    print(f"{len(os.listdir(os.path.join('data', 'cats-v-dogs', 'validation', 'cats')))} cat validation photos")
    print(f"{len(os.listdir(os.path.join('data', 'cats-v-dogs', 'validation', 'cute_cats')))} cute cat validation photos")
    print(f"{len(os.listdir(os.path.join('data', 'cats-v-dogs', 'validation', 'dogs')))} dog validation photos")
    print(f"{len(os.listdir(os.path.join('data', 'cats-v-dogs', 'validation', 'cute_dogs')))} cute dog validation photos")

if __name__ == "__main__":
    # download_data()
    build_data_dirs()
