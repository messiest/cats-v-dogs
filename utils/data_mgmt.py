import os
import shutil
import subprocess
import zipfile

import pandas as pd
from sklearn.model_selection import train_test_split


def white_space(msg):
    columns , _ = shutil.get_terminal_size()
    length = columns - len(msg)

    return " " * length


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


def build_cute_cats_n_dogs():
    df = pd.read_csv(r'data/labels.csv')
    df.drop_duplicates(keep=False, subset=['url'], inplace=True)

    training, validation = train_test_split(df, test_size=.1, shuffle=True)

    training.insert(0, 'split', 'training')
    validation.insert(0, 'split', 'validation')

    df = pd.concat([training, validation])
    df = df.sample(frac=1)  # shuffle dataframe

    for i, row in df.iterrows():
        src = os.path.join('data', 'pixabay', row['type'], str(row['cute']), f"{row['id']}.jpg")
        if row['cute']:
            label = f"cute_{row['type']}"
        else:
            label = row['type']
        dst = os.path.join('data', 'cute-cats-n-dogs', row['split'], label, f"{row['id']}.jpg")

        os.makedirs(os.path.join('data', 'cute-cats-n-dogs', row['split'], label), exist_ok=True)
        msg = f"{row['split'].title()}: {src}, {dst}"
        print(msg + white_space(msg), end="\r")
        shutil.copyfile(src, dst)

    print()
    for path in os.walk('data/cute-cats-n-dogs/'):
        if not path[1]:
            msg = f"{len(path[2])} {path[0].split('/')[-1]} {path[0].split('/')[-2]} photos"
            print(msg)


def build_cats_v_dogs():
    df = pd.read_csv(r'data/labels.csv')
    df.drop_duplicates(keep=False, subset=['url'], inplace=True)

    training_dir = os.path.join('data', 'cats-v-dogs', 'training')
    cats_training_dir = os.path.join('data', 'cats-v-dogs', 'training', 'cats')
    dogs_training_dir = os.path.join('data', 'cats-v-dogs', 'training', 'dogs')

    validation_dir = os.path.join('data', 'cats-v-dogs', 'validation')
    cats_validation_dir = os.path.join('data', 'cats-v-dogs', 'validation', 'cats')
    dogs_validation_dir = os.path.join('data', 'cats-v-dogs', 'validation', 'dogs')

    training, validation = train_test_split(df, test_size=.1, shuffle=True)

    training.insert(0, 'split', 'training')
    validation.insert(0, 'split', 'validation')

    df = pd.concat([training, validation])
    df = df.sample(frac=1)  # shuffle dataframe

    for i, row in df.iterrows():
        src = os.path.join('data', 'pixabay', row['type'], str(row['cute']), f"{row['id']}.jpg")
        dst = os.path.join('data', 'cats-v-dogs', row['split'], row['type'], f"{row['id']}.jpg")
        os.makedirs(os.path.join('data', 'cats-v-dogs', row['split'], row['type']), exist_ok=True)

        msg = f"{row['split'].title()}: {src}, {dst}"
        print(msg + white_space(msg), end="\r")
        shutil.copyfile(src, dst)

    print()
    for path in os.walk('data/cats-v-dogs/'):
        if not path[1]:
            msg = f"{len(path[2])} {path[0].split('/')[-1]} {path[0].split('/')[-2]} photos"
            print(msg)

if __name__ == "__main__":
    # download_data()
    build_data_dirs()
