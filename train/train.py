import os
from datetime import datetime
from pathlib import Path

import hydra
from omegaconf.dictconfig import DictConfig
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)
import torch
import transformers
from transformers import (
    AutoModelForSequenceClassification,
    optimization,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)
from transformers import PreTrainedModel
from transformers.optimization import Optimizer, SchedulerType

from preprocess import prepare_datasets
from utils import get_logger

logger = get_logger()


@hydra.main(version_base=None, config_path='config',
            config_name='config_text_classification.yaml')
def train_model(config: DictConfig):

    seed = config.seed
    torch.manual_seed(seed)
    transformers.set_seed(seed)

    experiment_name = str(
        datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    ) if not config.experiment_name else config.experiment_name

    path_artifacts = Path().resolve().joinpath(*config.artifacts)
    os.environ['MLFLOW_TRACKING_URI'] = str(path_artifacts.joinpath(
        *config.model_lifecycle
    ))
    os.environ['MLFLOW_EXPERIMENT_NAME'] = experiment_name
    os.environ['MLFLOW_FLATTEN_PARAMS'] = str(True)
    os.environ['TOKENIZERS_PARALLELISM'] = str(True)

    path_logs = str(path_artifacts.joinpath(
        *config.log_dir, experiment_name
    ))
    path_results = str(path_artifacts.joinpath(
        *config.results, experiment_name
    ))
    model_name = config.model.model_name
    save_model = config.train.save_model
    batch_size = config.train.training_arguments.per_device_train_batch_size
    num_epochs = config.train.training_arguments.num_train_epochs

    train_dataset, valid_dataset = prepare_datasets(config)
    logger.info(f"Training on {'GPU' if torch.cuda.is_available() else 'CPU'}")

    num_training_steps = len(train_dataset) * num_epochs // batch_size

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=config.model.num_labels,
        return_dict=True
    )
    if config.model.freeze_encoder:
        for param in model.base_model.parameters():
            param.requires_grad = False

    logger.info(
        "Total number of trainable parameters: "
        f"{sum(p.numel() for p in model.parameters() if p.requires_grad):,}"
    )
    optimizer = get_optimizer(config, model)
    schedule = get_schedule(config, optimizer, num_training_steps)
    callbacks = []
    training_args = TrainingArguments(
        output_dir=path_results,
        logging_dir=path_logs,
        seed=seed,
        **config.train.training_arguments
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        optimizers=(optimizer, schedule),
        compute_metrics=compute_metrics,
        callbacks=callbacks
    )
    trainer.add_callback(ComputeTrainMetricsCallback(trainer))
    trainer.train()

    if save_model:
        trainer.save_model(str(Path(path_results).joinpath(model_name)))


# https://discuss.huggingface.co/t/metrics-for-training-set-in-trainer/2461/4
class ComputeTrainMetricsCallback(TrainerCallback):
    def __init__(self, trainer: Trainer):
        self.trainer = trainer

    def on_epoch_end(self, args, state, control, **kwargs):
        train_metrics = self.trainer.evaluate(
            self.trainer.train_dataset, metric_key_prefix='train'
        )
        logger.info(f"Train metrics: {train_metrics}")
        valid_metrics = self.trainer.evaluate(
            self.trainer.eval_dataset, metric_key_prefix='valid'
        )
        logger.info(f"Valid metrics: {valid_metrics}")


def get_optimizer(config: DictConfig, model: PreTrainedModel) -> Optimizer:
    name = config.train.optimizer.name
    params = config.train.optimizer.params

    return getattr(optimization, name)(model.parameters(), **params)


def get_schedule(
    config: DictConfig,
    optimizer: Optimizer,
    num_training_steps: int
) -> SchedulerType:
    name = config.train.schedule.name
    params = config.train.schedule.params

    return getattr(optimization, name)(
        optimizer=optimizer,
        num_training_steps=num_training_steps,
        **params
    )


def compute_metrics(eval_preds):
    logits, labels = eval_preds
    preds = logits.argmax(-1)

    accuracy = accuracy_score(labels, preds)
    precision = precision_score(labels, preds, average='weighted')
    recall = recall_score(labels, preds, average='weighted')
    f1 = f1_score(labels, preds, average='weighted')

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
    }


if __name__ == '__main__':

    train_model()
