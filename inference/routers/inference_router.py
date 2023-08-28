from fastapi import APIRouter, BackgroundTasks, Body, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse

from database.database_ops import (
    create_database,
    save_predictions,
    get_data_for_data_drift_report,
    get_predictions_for_classification_report
)
from data_models.request_models import (
    PredictionRequest,
    ReplaceModelRequest
)
from data_models.response_models import PredictionResponse
from use_cases.inference import (
    load_tokenizer_and_model,
    load_label_encoder,
    predict_product_for_narrative
)
from monitoring.reports import (
    compute_data_drift_report,
    compute_classification_report,
    read_reference_data
)
from routers import routes as r
from utils.utils import get_logger


router = APIRouter()

log = get_logger()
create_database()
tokenizer, model = load_tokenizer_and_model()
log.info(f"Tokenizer: {type(tokenizer).__name__}")
log.info(f"Model: {type(model).__name__}")
label_encoder = load_label_encoder()
log.info("Reading reference data for reports...")
df_ref = read_reference_data(label_encoder)


@router.post(r.INFERENCE, response_model=PredictionResponse)
def predict_product(
    background_tasks: BackgroundTasks,
    request: PredictionRequest = Body(...)
):
    log.info(f"Received request: {request}")
    pred = predict_product_for_narrative(
        tokenizer=tokenizer,
        model=model,
        label_encoder=label_encoder,
        **request.dict()
    )
    background_tasks.add_task(save_predictions, [pred])

    pr = PredictionResponse(
        user_id=pred.user_id,
        product_prediction=pred.product_prediction,
        product=pred.product,
        created_date=pred.created_date
    )
    log.info(f"Prediction: {pr}")

    return pr


@router.post(r.REPLACE_MODEL)
def replace_model(request: ReplaceModelRequest = Body(...)):
    log.info(f"Received request to replace the model: {request}")
    try:
        tokenizer, model = load_tokenizer_and_model(**request.dict())
        log.info(
            "Successfully replaced the previous model for "
            f"{type(model).__name__}!"
        )
        return JSONResponse(
            status_code=200, content={"message": "Replaced model!"}
        )
    except Exception as e:
        log.error(f"Failed to replace the model: {e}")
        raise HTTPException(status_code=409, detail=str(e))


@router.get(r.DATA_DRIFT_REPORT, response_class=HTMLResponse)
def get_data_drift_report(last_n: int):

    df_curr = get_data_for_data_drift_report(last_n)
    report = compute_data_drift_report(df_ref, df_curr)

    return HTMLResponse(report.get_html())


@router.get(r.CLASSIFICATION_REPORT, response_class=HTMLResponse)
def get_classification_report(last_n: int):

    df_curr = get_predictions_for_classification_report(last_n)
    report = compute_classification_report(df_curr)

    return HTMLResponse(report.get_html())
