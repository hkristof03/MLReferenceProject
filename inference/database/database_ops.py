from typing import List

from omegaconf.dictconfig import DictConfig
from sqlalchemy import create_engine, Engine, select
from sqlalchemy.orm import sessionmaker, Session
import pandas as pd

from data_models.database_models import Base, PredictionTable
from data_models.ml_models import Prediction
from utils.utils import read_config


conf: DictConfig = read_config()


def create_database() -> None:
    engine = create_engine(conf.database_uri)
    Base.metadata.create_all(engine)


def save_predictions(predictions: List[Prediction]) -> None:

    engine = create_engine(conf.database_uri)
    session = open_sqa_session(engine)
    session.add_all([PredictionTable(**pred.dict()) for pred in predictions])
    session.commit()


def get_data_for_data_drift_report(last_n_predictions: int) -> pd.DataFrame:
    engine = create_engine(conf.database_uri)
    with engine.connect() as conn:
        order = PredictionTable.created_date.desc()
        query = select(
            PredictionTable.narrative,
            PredictionTable.product
        ).order_by(order).limit(last_n_predictions)

        df_curr = pd.read_sql_query(sql=query, con=conn)

    return df_curr


def get_predictions_for_classification_report(
    last_n_predictions: int
) -> pd.DataFrame:

    engine = create_engine(conf.database_uri)
    with engine.connect() as conn:
        order = PredictionTable.created_date.desc()
        query = select(
            PredictionTable.narrative,
            PredictionTable.product_prediction,
            PredictionTable.product
        ).order_by(order).limit(last_n_predictions)

        df_curr = pd.read_sql_query(sql=query, con=conn)

    return df_curr


def open_sqa_session(engine: Engine) -> Session:

    session = sessionmaker(bind=engine)

    return session()
