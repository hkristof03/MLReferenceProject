from datetime import datetime
from typing import Optional

from pydantic import BaseModel, constr, Field, UUID4


class Prediction(BaseModel):
    user_id: UUID4
    created_date: datetime = Field(default_factory=datetime.now)
    narrative: constr(min_length=10, max_length=5000)
    product_prediction: str
    product: Optional[str]
