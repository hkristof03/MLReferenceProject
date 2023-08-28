from datetime import datetime
from uuid import uuid4

from pydantic import BaseModel, constr, UUID4


class PredictionResponse(BaseModel):
    user_id: UUID4 = uuid4()
    product: constr(min_length=1, max_length=100)
    product_prediction: constr(min_length=1, max_length=100)
    created_date: datetime
