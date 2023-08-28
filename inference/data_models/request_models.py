from pydantic import BaseModel, constr, UUID4


class PredictionRequest(BaseModel):
    user_id: UUID4
    narrative: constr(min_length=10, max_length=5000)
    product: constr(min_length=0, max_length=100)


class ReplaceModelRequest(BaseModel):
    model_name: str
    experiment: constr(min_length=19, max_length=19)
    checkpoint: constr(min_length=11, max_length=20)
