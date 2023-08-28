from pathlib import Path

from evidently import ColumnMapping
from evidently.metric_preset import (
    DataDriftPreset,
    DataQualityPreset,
    TextOverviewPreset,
    ClassificationPreset
)
from evidently.report import Report
from omegaconf.dictconfig import DictConfig
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import nltk

from utils.utils import read_config


conf: DictConfig = read_config()
nltk.download('words')
nltk.download('wordnet')
nltk.download('omw-1.4')


def compute_data_drift_report(
    df_ref: pd.DataFrame,
    df_curr: pd.DataFrame
) -> Report:
    """"""
    text_feature = conf.reports.text_feature
    column_mapping = ColumnMapping(
        target=conf.reports.target,
        text_features=[text_feature],
    )
    text_report = Report(metrics=[
        TextOverviewPreset(column_name=text_feature),
        DataQualityPreset(),
        DataDriftPreset()
    ])
    text_report.run(
        reference_data=df_ref[[text_feature]].sample(frac=0.05, replace=False),
        current_data=df_curr[[text_feature]],
        column_mapping=column_mapping
    )
    return text_report


def compute_classification_report(df_curr: pd.DataFrame) -> Report:
    """"""
    text_feature = conf.reports.text_feature
    column_mapping = ColumnMapping(
        target=conf.reports.target,
        prediction=conf.reports.prediction,
        text_features=[text_feature],
    )
    classification_report = Report(metrics=[
        ClassificationPreset()
    ])
    classification_report.run(
        reference_data=None,
        current_data=df_curr,
        column_mapping=column_mapping
    )
    return classification_report


def read_reference_data(label_encoder: LabelEncoder) -> pd.DataFrame:

    target = conf.reports.target
    path_ref_data = Path().resolve().joinpath(*conf.reports.reference_data)
    df_ref = pd.read_parquet(path_ref_data)
    df_ref[target] = label_encoder.inverse_transform(df_ref[target])

    return df_ref
