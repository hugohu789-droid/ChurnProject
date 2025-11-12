from sqlalchemy import create_engine, Column, Integer, String, DateTime, Numeric
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime

Base = declarative_base()

class FileUpload(Base):
    __tablename__ = "file_uploads"
    
    id = Column(Integer, primary_key=True, index=True)
    original_filename = Column(String)
    saved_filename = Column(String)
    file_path = Column(String)
    upload_time = Column(DateTime, default=datetime.now)
    file_size = Column(Integer)
    status = Column(String)

class TrainModel(Base):
    __tablename__ = "train_models"
    
    id = Column(Integer, primary_key=True, index=True)
    file_id = Column(Integer)
    record_number = Column(Integer)
    accuracy = Column(Numeric)
    recall_rate = Column(Numeric)
    precision = Column(Numeric)
    model_name = Column(String)
    model1_path = Column(String)
    model2_path = Column(String)
    train_date = Column(DateTime, default=datetime.now)

# class PredictFileUpload(Base):
#     __tablename__ = "prdict_file_uploads"
    
#     id = Column(Integer, primary_key=True, index=True)
#     original_filename = Column(String)
#     saved_filename = Column(String)
#     file_path = Column(String)
#     upload_time = Column(DateTime, default=datetime.now)
#     file_size = Column(Integer)
#     model_id = Column(String)
#     results_path = Column(String)

class PredictionHistory(Base):
    __tablename__ = "prdiction_histories"
    
    id = Column(Integer, primary_key=True, index=True)
    original_filename = Column(String)
    saved_filename = Column(String)
    train_file_path = Column(String)
    train_model_id = Column(Integer)
    train_model_name = Column(String)
    result1_path = Column(String)
    result2_path = Column(String)
    predict_date = Column(DateTime, default=datetime.now)
    status = Column(String)

engine = create_engine('sqlite:///./fileupload.db')
Base.metadata.create_all(bind=engine)

if __name__ == "__main__":
    Base.metadata.create_all(bind=engine)