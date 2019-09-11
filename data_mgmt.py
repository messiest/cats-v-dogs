import os
import shutil

import pandas as pd
from sklearn.model_selection import train_test_split


def main():
    data_dir = os.path.join('data', 'pixabay')

    cat_dir = os.path.join(data_dir, 'cats')
    dog_dir = os.path.join(data_dir, 'dogs')

    print(f"{len(os.listdir(cat_dir))} cat photos")
    print(f"{len(os.listdir(dog_dir))} dog photos")

    df = pd.concat(
        [
            pd.DataFrame(
                {
                    'file': [os.path.join(cat_dir, cat) for cat in os.listdir(cat_dir)],
                    'label': ['cats' for _ in os.listdir(cat_dir)]
                },
            ),
            pd.DataFrame(
                {
                    'file': [os.path.join(dog_dir, dog) for dog in os.listdir(dog_dir)],
                    'label': ['dogs' for _ in os.listdir(dog_dir)]
                },
            ),
        ]
    )

    train, validation = train_test_split(df, train_size=.9, stratify=df['label'])

    os.makedirs(os.path.join('data', 'cats-v-dogs', 'training', 'cats'), exist_ok=True)
    os.makedirs(os.path.join('data', 'cats-v-dogs', 'training', 'dogs'), exist_ok=True)
    os.makedirs(os.path.join('data', 'cats-v-dogs', 'validation', 'cats'), exist_ok=True)
    os.makedirs(os.path.join('data', 'cats-v-dogs', 'validation', 'dogs'), exist_ok=True)

    for i, row in train.iterrows():
        file = row['file']
        label = row['label']

        file_name = os.path.basename(file)

        print(f"Training: {label}, {file_name}", end='\r')

        shutil.copyfile(file, os.path.join('data', 'cats-v-dogs', 'training', label, file_name))

    for i, row in validation.iterrows():
        file = row['file']
        label = row['label']

        file_name = os.path.basename(file)

        print(f"Validation: {label}, {file_name}", end='\r')

        shutil.copyfile(file, os.path.join('data', 'cats-v-dogs', 'validation', label, file_name))

    print(f"\n{len(os.listdir(os.path.join('data', 'cats-v-dogs', 'training', 'cats')))} cat training photos")
    print(f"{len(os.listdir(os.path.join('data', 'cats-v-dogs', 'training', 'dogs')))} dog training photos")
    print(f"{len(os.listdir(os.path.join('data', 'cats-v-dogs', 'validation', 'cats')))} cat validation photos")
    print(f"{len(os.listdir(os.path.join('data', 'cats-v-dogs', 'validation', 'dogs')))} dog validation photos")

if __name__ == "__main__":
    main()
