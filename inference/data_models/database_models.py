from sqlalchemy import Column, DateTime, Float, Integer, String
from sqlalchemy.orm import declarative_base

Base = declarative_base()


class PredictionTable(Base):

    __tablename__ = 'predictions'
    id = Column(Integer, primary_key=True)
    user_id = Column(String, nullable=False)
    created_date = Column(DateTime, nullable=False)
    narrative = Column(String, nullable=False)
    product_prediction = Column(String, nullable=False)
    product = Column(String, nullable=True)
