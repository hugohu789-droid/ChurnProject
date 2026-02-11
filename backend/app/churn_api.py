from fastapi import FastAPI, Query, UploadFile, HTTPException, File, Form, BackgroundTasks, Depends
from fastapi.responses import FileResponse
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy import func
from datetime import datetime
import os
from pydantic import BaseModel
from pathlib import Path

from models import PredictionHistory, TrainModel, engine, FileUpload
import shutil
import modeltrain

app = FastAPI()
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
SAFE_DOWNLOAD_DIR = Path(os.getcwd()).resolve()  # Define a safe directory for downloads

@app.get("/")
async def root():
    return {"message": "File Upload API is running"}

# Dependency to get DB session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

class TrainRequest(BaseModel):
    id: int
    model_name: str

# 新增：分页请求模型
class PageRequest(BaseModel):
    page: int = 1
    page_size: int = 10

@app.get("/api/dashboard/stats")
def dashboard_stats(db: Session = Depends(get_db)):
    # Basic Statistics
    total_files = db.query(FileUpload).count()
    total_models = db.query(TrainModel).count()
    total_predictions = db.query(PredictionHistory).count()
    
    # Calculate the average accuracy
    avg_accuracy = db.query(func.avg(TrainModel.accuracy)).scalar()
    avg_accuracy = avg_accuracy if avg_accuracy is not None else 0.0
    
    # Get the 5 most recently trained models
    recent_models = db.query(TrainModel).order_by(TrainModel.train_date.desc()).limit(5).all()
    
    recent_models_list = []
    for m in recent_models:
        recent_models_list.append({
            "id": m.id,
            "model_name": m.model_name,
            "accuracy": m.accuracy,
            "precision": m.precision,
            "train_date": m.train_date
        })

    return {
        "total_files": total_files,
        "total_models": total_models,
        "total_predictions": total_predictions,
        "average_accuracy": avg_accuracy,
        "recent_models": recent_models_list
    }

def train_model_background(id_value: int, model_name: str):
    # Background tasks create their own session
    db = SessionLocal()
    try:
        file_record = db.query(FileUpload).filter(FileUpload.id == id_value).first()
        if not file_record:
            print(f"Error: File record with id {id_value} not found")
            return
        file_path = file_record.file_path

        # Create date-based directory
        today = datetime.now()
        date_dir = today.strftime("%Y%m%d")
        models_dir = os.path.join("trained_models", date_dir)
        os.makedirs(models_dir, exist_ok=True)

        # Handle filename conflicts
        base_name = os.path.splitext(file_record.saved_filename)[0]
        final_model_name = base_name + '_' + 'churn_model.joblib'
        model_path = os.path.join(models_dir, final_model_name)


        # Train
        metrics = modeltrain.train_model(file_path, model_save_path = model_path)

        train_model = TrainModel(
            file_id = id_value,
            record_number= 0,
            accuracy= metrics['accuracy'],
            recall_rate= metrics['recall'],
            precision= metrics['precision'],
            model1_path= model_path,
            model_name = model_name # Use the user provided name or generated one
        )

        db.add(train_model)
        
        # Update status
        file_record.status = "trained"
        db.add(file_record)
        
        db.commit()
    except Exception as e:
        print(f"Error during background training: {e}")
        db.rollback()
    finally:
        db.close()

@app.post("/api/modeltraining/train")
def train(
    background_tasks: BackgroundTasks,
    body: TrainRequest,
    db: Session = Depends(get_db)
):
    id_value = body.id
    model_name = body.model_name

    background_tasks.add_task(train_model_background, id_value, model_name)
    
    file_record = db.query(FileUpload).filter(FileUpload.id == id_value).first()
    if not file_record:
        raise HTTPException(status_code=404, detail="File not found")
    
    file_record.status = "training"
    db.commit()

    return {"id": id_value, "model_name": model_name}

@app.post("/api/modeltraining/list")
def list_upload_files(body: PageRequest, db: Session = Depends(get_db)):
    page = max(1, body.page)
    page_size = max(1, body.page_size)

    total = db.query(FileUpload).count()
    offset = (page - 1) * page_size
    records = db.query(FileUpload).order_by(FileUpload.upload_time.desc()).offset(offset).limit(page_size).all()

    items = []
    for r in records:
        items.append({
            "id": r.id,
            "original_filename": r.original_filename,
            "saved_filename": r.saved_filename,
            "file_path": r.file_path,
            "file_size": r.file_size,
            "status": r.status,
            "upload_time": r.upload_time
        })

    return {
        "page": page,
        "page_size": page_size,
        "total": total,
        "records": items
    }

@app.get("/api/models/{id}")
def model_detail(id: int, db: Session = Depends(get_db)):
    model = db.query(TrainModel).filter(TrainModel.id == id).first()
    if not model:
        raise HTTPException(status_code=404, detail=f"Model with id {id} not found")
    
    return {
        "id": model.id,
        "file_id": model.file_id,
        "record_number": model.record_number,
        "accuracy": model.accuracy,
        "recall_rate": model.recall_rate,
        "precision": model.precision,
        "model_name": model.model_name,
        "train_date": model.train_date
    }

@app.post("/api/trained/models")
def list_trained_models(db: Session = Depends(get_db)):
    records = db.query(TrainModel).order_by(TrainModel.train_date.desc()).all()

    items = []
    for r in records:
        items.append({
            "id": r.id,
            "file_id": r.file_id,
            "record_number": r.record_number,
            "accuracy": r.accuracy,   
            "recall_rate": r.recall_rate,
            "model_name": r.model_name,
            "train_date": r.train_date
        })

    return {
        "records": items
    }

@app.post("/api/upload")
def upload_file(file: UploadFile = File(...), db: Session = Depends(get_db)):
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Only CSV files are allowed")
    
    try:
        # Create date-based directory
        today = datetime.now()
        date_dir = today.strftime("%Y%m%d")
        upload_dir = os.path.join("uploadfiles", date_dir)
        os.makedirs(upload_dir, exist_ok=True)

        # Handle filename conflicts
        base_name = os.path.splitext(file.filename)[0]
        extension = os.path.splitext(file.filename)[1]
        counter = 1
        new_filename = file.filename
        while os.path.exists(os.path.join(upload_dir, new_filename)):
            new_filename = f"{base_name}_{counter}{extension}"
            counter += 1

        # Save file
        file_path = os.path.join(upload_dir, new_filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Save to database
        db = SessionLocal()
        file_size = os.path.getsize(file_path)
        db_file = FileUpload(
            original_filename=file.filename,
            saved_filename=new_filename,
            file_path=file_path,
            file_size=file_size,
            status="uploaded"
        )
        db.add(db_file)
        db.commit()

        return {
            "original_filename": file.filename,
            "saved_filename": new_filename,
            "file_path": file_path,
            "file_size": file_size
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
def predict_model_background(predict_id: int, file_path: str, model_id: str):
    # predict model name from TrainModel table
    db = SessionLocal()
    try:
        predict_record = db.query(PredictionHistory).filter(PredictionHistory.id == predict_id).first()
        if not predict_record:
            return

        today = datetime.now()
        date_dir = today.strftime("%Y%m%d")
        upload_dir = os.path.join("predictresults", date_dir)
        os.makedirs(upload_dir, exist_ok=True)

        # Handle filename conflicts
        base_name = 'predict_result_' + str(predict_id)
        extension = '.csv'
        counter = 1
        new_filename = base_name + extension
        while os.path.exists(os.path.join(upload_dir, new_filename)):
            new_filename = f"{base_name}_{counter}{extension}"
            counter += 1
        predict_record.result1_path = os.path.join(upload_dir, new_filename)

        model = db.query(TrainModel).filter(TrainModel.id == model_id).first()
        if model:
            modeltrain.predict(data_path = file_path, model_path = model.model1_path, output_path = predict_record.result1_path)

        predict_record.status = "completed"
        db.add(predict_record)
        db.commit()
    except Exception as e:
        print(f"Prediction error: {e}")
    finally:
        db.close()

@app.post("/api/predict")
def predict(
    background_tasks: BackgroundTasks, 
    file: UploadFile = File(...), 
    model_id: str = Form(..., alias="modelId"), # Alias allows frontend to send modelId
    db: Session = Depends(get_db)
):
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Only CSV files are allowed")

    ret = {}

    db_file_id = 0

    file_path = ""
   
    try:
        # Create date-based directory
        today = datetime.now()
        date_dir = today.strftime("%Y%m%d")
        upload_dir = os.path.join("predictfiles", date_dir)
        os.makedirs(upload_dir, exist_ok=True)

        # Handle filename conflicts
        base_name = os.path.splitext(file.filename)[0]
        extension = os.path.splitext(file.filename)[1]
        counter = 1
        new_filename = file.filename
        while os.path.exists(os.path.join(upload_dir, new_filename)):
            new_filename = f"{base_name}_{counter}{extension}"
            counter += 1

        # Save file
        file_path = os.path.join(upload_dir, new_filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Save to database
        db = SessionLocal()

        model = db.query(TrainModel).filter(TrainModel.id == model_id).first()

        file_size = os.path.getsize(file_path)
        db_file = PredictionHistory(
            original_filename=file.filename,
            saved_filename=new_filename,
            train_file_path = file_path,
            train_model_id = model_id,
            train_model_name = model.model_name if model else "",
            result1_path = "",
            result2_path = "",
            predict_date = datetime.now(),
            status = "predicting"
        )
        db.add(db_file)
        db.commit()

        db_file_id = db_file.id
        
        ret = {
            "original_filename": file.filename,
            "saved_filename": new_filename,
            "file_path": file_path,
            "file_size": file_size
        } 
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    background_tasks.add_task(predict_model_background, db_file_id, file_path, model_id)
    return ret
    
@app.post("/api/predictions")
def list_predictions(body: PageRequest, db: Session = Depends(get_db)): 
    page = max(1, body.page)
    page_size = max(1, body.page_size)

    total = db.query(PredictionHistory).count()
    offset = (page - 1) * page_size
    records = db.query(PredictionHistory).order_by(PredictionHistory.predict_date.desc()).offset(offset).limit(page_size).all()

    items = []
    for r in records:
        items.append({
            "id": r.id,
            "train_model_id": r.train_model_id,
            "train_model_name": r.train_model_name,
            "result1_path": r.result1_path,   
            "result2_path": r.result2_path,
            "predict_date": r.predict_date,
            "status": r.status
        }) 

    return {
        "page": page,
        "page_size": page_size,
        "total": total,
        "records": items
    }

@app.post("/api/models/list")
def list_models(body: PageRequest, db: Session = Depends(get_db)):
    page = max(1, body.page)
    page_size = max(1, body.page_size)

    total = db.query(TrainModel).count()
    offset = (page - 1) * page_size
    records = db.query(TrainModel).order_by(TrainModel.train_date.desc()).offset(offset).limit(page_size).all()

    items = []
    for r in records:
        items.append({
            "id": r.id,
            "file_id": r.file_id,
            "record_number": r.record_number,
            "accuracy": r.accuracy,   
            "recall_rate": r.recall_rate,
            "precision": r.precision,
            "model_name": r.model_name,
            "train_date": r.train_date
        })

    return {
        "page": page,
        "page_size": page_size,
        "total": total,
        "records": items
    }



@app.get("/api/download")
async def download_file(
    # We expect the path sent by the frontend to be *relative* to SAFE_DOWNLOAD_DIR
    # e.g., "user_reports/2025/report_A.csv"
    file_path: str = Query(..., description="The relative path to the file")
):
    """
    Securely validates the path and returns the file.
    """
    try:
        
        # 1. Combine the secure root directory with the user-requested relative path
        # lstrip('.\\/') cleans up any attempts like ../ or /..
        relative_path = file_path.lstrip('.\\/')
        full_path = SAFE_DOWNLOAD_DIR.joinpath(relative_path).resolve()

        # 2. Core Security Check:
        # Verify that the resolved full_path is *still* inside SAFE_DOWNLOAD_DIR
        if not full_path.is_relative_to(SAFE_DOWNLOAD_DIR):
            print(f"[AUTH_FAIL] Path Traversal Attempt: {file_path}")
            raise HTTPException(status_code=403, detail="Access Forbidden")

        # 3. Check if the file exists
        if not full_path.is_file():
            print(f"[NOT_FOUND] File not found: {full_path}")
            raise HTTPException(status_code=404, detail="File not found")

        # 4. All safe, prepare the FileResponse
        # Extract the filename from the path to be displayed in the browser's download prompt
        filename = os.path.basename(full_path)

        return FileResponse(
            path=full_path,
            media_type='text/csv',  # Tell the browser it's a CSV
            filename=filename     # Suggest the default "Save As" filename to the browser
        )

    except Exception as e:
        print(f"[ERROR] An unexpected error occurred: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")

@app.delete("/api/modeltraining/{file_id}")
def delete_file(file_id: int, db: Session = Depends(get_db)):
    # search records
    file_record = db.query(FileUpload).filter(FileUpload.id == file_id).first()
    if not file_record:
        raise HTTPException(status_code=404, detail=f"File with id {file_id} not found")
    
    # If the file exists, delete the physical file.
    if os.path.exists(file_record.file_path):
        os.remove(file_record.file_path)
    
    # Delete database record
    db.delete(file_record)
    db.commit()
    
    return {"message": f"File with id {file_id} has been deleted"}

if __name__ == "__main__":
    from pydantic import BaseModel
    try:
        import uvicorn
    except ImportError:
        print("Error: uvicorn package is not installed.")
        print("Please install required packages using:")
        print("pip install fastapi uvicorn sqlalchemy")
        exit(1)
    uvicorn.run(app="churn_api:app", 
                host="0.0.0.0", 
                port=8000, 
                log_level="info",
                reload=True)