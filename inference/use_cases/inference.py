from pathlib import Path
from typing import Tuple
from uuid import uuid4


import numpy as np
from sklearn.preprocessing import LabelEncoder
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    PreTrainedTokenizer,
    PreTrainedModel
)

from data_models.ml_models import Prediction
from utils.utils import read_config


conf = read_config()


def load_tokenizer_and_model(
    model_name: str = conf.model_name,
    experiment: str = conf.experiment,
    checkpoint: str = conf.checkpoint,
) -> Tuple[PreTrainedTokenizer, PreTrainedModel]:
    """"""
    path_artifacts = Path().resolve().joinpath(*conf.trained_models)
    path_checkpoint = path_artifacts.joinpath(experiment, checkpoint)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(path_checkpoint)

    return tokenizer, model


def load_label_encoder() -> LabelEncoder:
    le = LabelEncoder()
    path_saved_classes = Path().resolve().joinpath(*conf.saved_classes)
    le.classes_ = np.load(str(path_saved_classes), allow_pickle=True)

    return le


def predict_product_for_narrative(
    tokenizer: PreTrainedTokenizer,
    model: PreTrainedModel,
    label_encoder: LabelEncoder,
    user_id: uuid4,
    narrative: str,
    product: str
) -> Prediction:
    """"""
    data = tokenizer([narrative], return_tensors='pt')['input_ids']
    logits = model(data).logits
    prediction = label_encoder.inverse_transform([logits.argmax(-1).item()])[0]

    return Prediction(
        user_id=user_id,
        narrative=narrative,
        product=product,
        product_prediction=prediction
    )
