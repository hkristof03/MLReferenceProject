import time
from pathlib import Path
from typing import Tuple

import hydra
from omegaconf.dictconfig import DictConfig
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from sklearn.preprocessing import LabelEncoder

from utils import get_logger

logger = get_logger()


def preprocess(config: DictConfig) -> None:
    """

    :param config:
    :return:
    """
    log = get_logger()
    seed = config.seed
    text_feature = config.preprocess.text_feature
    text_feature_hash = f'hash_{text_feature}'
    target_feature = config.preprocess.target_feature
    split_ratio = config.preprocess.split_ratio
    path_data = Path().resolve().joinpath(*config.preprocess.path_data)

    df = pd.read_csv(path_data.joinpath(config.preprocess.raw_data))
    df[text_feature_hash] = df[text_feature].apply(lambda x: hash(x))

    duplications_ratio = len(
        df.loc[df[text_feature_hash].duplicated()]
    ) / len(df)

    log.info(f"Duplications ratio: {duplications_ratio}")
    df.drop_duplicates(subset=text_feature_hash, keep=False).reset_index(
        drop=True, inplace=True
    )
    df.drop(columns=text_feature_hash, inplace=True)
    log.info(
        f"Dropping {len(df.loc[(df[text_feature].isnull())])} rows with null "
        "values!"
    )
    df.dropna(subset=[text_feature], inplace=True)
    log.info(f"There are {df[target_feature].nunique()} unique classes!")
    log.info(
        "Distribution of classes:\n"
        f"{df[target_feature].value_counts() / len(df)}"
    )
    # shuffle
    df = df.sample(frac=1, random_state=seed)
    log.info(
        "Creating train and validation parquet files with a split ratio of " 
        f"{split_ratio}"
    )
    df_train = df.iloc[:int(len(df) * (1 - split_ratio))]
    df_valid = df.iloc[len(df_train):]
    df_test = df_valid.iloc[len(df_valid) // 2:]
    df_valid = df_valid.iloc[:len(df_valid) // 2]

    log.info("Encoding target feature...")
    le = LabelEncoder()
    le = le.fit(df_train[target_feature])
    df_train.loc[:, target_feature] = le.transform(df_train[target_feature])
    df_valid.loc[:, target_feature] = le.transform(df_valid[target_feature])

    log.info("Saving LabelEncoder's state for inference...")
    np.save(
        str(path_data.joinpath(config.preprocess.save_classes)), le.classes_
    )
    log.info(f"Train samples: {len(df_train):,}")
    log.info(f"Validation samples: {len(df_valid):,}")
    log.info(f"Test samples: {len(df_test):,}")
    df_train.to_parquet(
        path_data.joinpath(config.preprocess.train_data), index=False
    )
    df_valid.to_parquet(
        path_data.joinpath(config.preprocess.validation_data), index=False
    )
    df_test.to_parquet(
        path_data.joinpath(config.preprocess.test_data), index=False
    )


def prepare_datasets(
    config: DictConfig
) -> Tuple[Dataset, Dataset]:
    """

    :param config:
    :return:
    """
    seed = config.seed
    sample_frac = config.preprocess.sample_frac
    text_feature = config.preprocess.text_feature
    target_feature = config.preprocess.target_feature
    path_data = Path().resolve().joinpath(*config.preprocess.path_data)

    df_train = pd.read_parquet(path_data.joinpath(
        config.preprocess.train_data
    ))
    df_valid = pd.read_parquet(path_data.joinpath(
        config.preprocess.validation_data
    ))
    if sample_frac:
        df_train = df_train.sample(frac=sample_frac, random_state=seed)
        df_valid = df_valid.sample(frac=sample_frac, random_state=seed)

    train_texts = df_train[text_feature].tolist()
    valid_texts = df_valid[text_feature].tolist()
    train_labels = df_train[target_feature].tolist()
    valid_labels = df_valid[target_feature].tolist()

    tokenizer = AutoTokenizer.from_pretrained(config.model.model_name)
    start = time.time()
    train_encodings = tokenizer(train_texts, truncation=True, padding=True)
    valid_encodings = tokenizer(valid_texts, truncation=True, padding=True)
    logger.info(f"Tokenization took {round(time.time() - start)} seconds...")
    train_dataset = Dataset(train_encodings, train_labels)
    valid_dataset = Dataset(valid_encodings, valid_labels)

    return train_dataset, valid_dataset


class Dataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in
                self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])

        return item

    def __len__(self):
        return len(self.labels)


@hydra.main(version_base=None, config_path='config',
            config_name='config_text_classification.yaml')
def preprocess_wrapper(config: DictConfig) -> None:
    preprocess(config)


if __name__ == '__main__':
    preprocess_wrapper()
