from fastapi import FastAPI, Query, UploadFile, HTTPException, File, Form, BackgroundTasks
from fastapi.responses import FileResponse
from sqlalchemy.orm import sessionmaker
from datetime import datetime
import os
from pydantic import BaseModel
from typing import List
from pathlib import Path

from models import PredictionHistory, TrainModel, engine, FileUpload
import shutil
import modeltrain

app = FastAPI()
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
SAFE_DOWNLOAD_DIR = Path('/Users/hugo/Dev/1.Python/py313').resolve()  # Define a safe directory for downloads

@app.get("/")
async def root():
    return {"message": "File Upload API is running"}

class TrainRequest(BaseModel):
    id: int
    modelName: str

# 新增：分页请求模型
class PageRequest(BaseModel):
    page: int = 1
    page_size: int = 10

class PredictForm(BaseModel):
    file: UploadFile
    modelId: str

def train_model_background(id_value: int, model_name: str):
    db = SessionLocal()
    try:
        fileRecord = db.query(FileUpload).filter(FileUpload.id == id_value).first()
        if not fileRecord:
            raise HTTPException(status_code=404, detail=f"File record with id {id} not found")
        file_path = fileRecord.file_path
        # Load the dataset
        df = modeltrain.loadData(file_path)

        # Create date-based directory
        today = datetime.now()
        date_dir = today.strftime("%Y%m%d")
        models_dir = os.path.join("trained_models", date_dir)
        os.makedirs(models_dir, exist_ok=True)

        # Handle filename conflicts
        base_name = os.path.splitext(fileRecord.saved_filename)[0]
        model1_name = base_name + '_' + 'lgbm_churn_model.joblib'
        model2_name = base_name + '_' + 'lgbm_churn_model_optuna-tuned.joblib'

        cols_to_drop = ['SNAPSHOT_DATE', 'CUSTOMER_NUMBER', 'PRODUCT_TYPE']
        (train_df, test_df, X_train, y_train, X_test, y_test) = modeltrain.cleanData(df, cols_to_drop = cols_to_drop)

        (precision, recall, auc) = modeltrain.final_model_training(X_train, y_train, X_test, y_test,
                                         dir = models_dir + '/', 
                                         model1_name = model1_name, 
                                         model2_name = model2_name)

        train_model = TrainModel(
            file_id = id_value,
            record_number= len(df),
            accuracy= auc,
            recall_rate= recall,
            precision= precision,
            model1_path= os.path.join(models_dir, model1_name),
            model2_path= os.path.join(models_dir, model2_name),
            model_name = model_name
        )

        db.add(train_model)
        db.commit()
    finally:
        db.close()

    db = SessionLocal()
    try:
        # db.add(train_model)

        # update upload_files id equal id_value records, update status to "trained"
        file_record = db.query(FileUpload).filter(FileUpload.id == id_value).first()
        if file_record:
            file_record.status = "trained"
            db.add(file_record)

        db.commit()
    finally:
        db.close()

@app.post("/api/modeltraining/train")
async def train(background_tasks: BackgroundTasks,
                body: TrainRequest):
    # get two parameters from body
    id_value = body.id
    model_name = body.modelName

    background_tasks.add_task(train_model_background, id_value, model_name)
    
    db = SessionLocal()
    try:
        # db.add(train_model)

        # update upload_files id equal id_value records, update status to "trained"
        file_record = db.query(FileUpload).filter(FileUpload.id == id_value).first()
        if file_record:
            file_record.status = "training"
            db.add(file_record)

        db.commit()
    finally:
        db.close()

    return {"id": id_value, "modelName": model_name}

@app.post("/api/modeltraining/list")
async def list_upload_files(body: PageRequest):
    page = max(1, body.page)
    page_size = max(1, body.page_size)

    db = SessionLocal()
    try:
        total = db.query(FileUpload).count()
        offset = (page - 1) * page_size
        records = db.query(FileUpload).order_by(FileUpload.upload_time.desc()).offset(offset).limit(page_size).all()

        items = []
        for r in records:
            items.append({
                "id": r.id,
                "original_filename": getattr(r, "original_filename", None),
                "saved_filename": getattr(r, "saved_filename", None),
                "file_path": getattr(r, "file_path", None),
                "file_size": getattr(r, "file_size", None),
                "status": getattr(r, "status", None),
                "upload_time": getattr(r, "upload_time", None)
            })

        return {
            "page": page,
            "page_size": page_size,
            "total": total,
            "records": items
        }
    finally:
        db.close()

@app.get("/api/models/{id}")
async def model_detail(id: int):
    db = SessionLocal()
    try:
        model = db.query(TrainModel).filter(TrainModel.id == id).first()
        if not model:
            raise HTTPException(status_code=404, detail=f"Model with id {id} not found")
        
        return {
            "id": model.id,
            "file_id": model.file_id,
            "record_number": model.record_number,
            "accuracy": getattr(model, "accuracy", None),
            "recall_rate": getattr(model, "recall_rate", None),
            "precision": getattr(model, "precision", None),
            "model_name": getattr(model, "model_name", None),
            "train_date": getattr(model, "train_date", None)
        }
    finally:
        db.close()

@app.post("/api/trained/models")
async def list_trained_models():

    db = SessionLocal()
    try:
        records = db.query(TrainModel).order_by(TrainModel.train_date.desc()).all()

        items = []
        for r in records:
            items.append({
                "id": r.id,
                "file_id": r.file_id,
                "record_number": r.record_number,
                "accuracy": getattr(r, "accuracy", None),   
                "recall_rate": getattr(r, "recall_rate", None),
                "model_name": getattr(r, "model_name", None),
                "train_date": getattr(r, "train_date", None)
            })

        return {
            "records": items
        }
    finally:
        db.close()

@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...)):  # 明确指定为 File 类型
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
        db.close()

        return {
            "original_filename": file.filename,
            "saved_filename": new_filename,
            "file_path": file_path,
            "file_size": file_size
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
def predict_model_background(predictId: int, file_path: str, model_id: str):
    # predict model name from TrainModel table
    
    try:
        db = SessionLocal()

        predict_record = db.query(PredictionHistory).filter(PredictionHistory.id == predictId).first()

        model = db.query(TrainModel).filter(TrainModel.id == model_id).first()
        df1, df2 = modeltrain.predict(file_path, model_path=model.model1_path, model_path_optuna=model.model2_path)
        
        today = datetime.now()
        date_dir = today.strftime("%Y%m%d")
        upload_dir = os.path.join("predictresults", date_dir)
        os.makedirs(upload_dir, exist_ok=True)

        # Handle filename conflicts
        base_name = 'lightgbm_predict_result_' + str(predictId)
        extension = '.csv'
        counter = 1
        new_filename = base_name + extension
        while os.path.exists(os.path.join(upload_dir, new_filename)):
            new_filename = f"{base_name}_{counter}{extension}"
            counter += 1
        predict_record.result1_path = os.path.join(upload_dir, new_filename)
        df1.to_csv(predict_record.result1_path, index=False)

        base_name = 'lightgbm_optuna_predict_result_' + str(predictId)
        extension = '.csv'
        counter = 1
        new_filename = base_name + extension
        while os.path.exists(os.path.join(upload_dir, new_filename)):
            new_filename = f"{base_name}_{counter}{extension}"
            counter += 1
        predict_record.result2_path = os.path.join(upload_dir, new_filename)
        df2.to_csv(predict_record.result2_path, index=False)

        predict_record.status = "completed"
        db.add(predict_record)
        db.commit()
        db.close()
    except Exception as e:
        return str(e)

@app.post("/api/predict")
async def upload_predict_file(background_tasks: BackgroundTasks, file: UploadFile = File(...), modelId: str = Form(...)):
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Only CSV files are allowed")
    
    file = file
    model_id = modelId

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

        db.close()

        
        ret = {
            "original_filename": file.filename,
            "saved_filename": new_filename,
            "file_path": file_path,
            "file_size": file_size
        } 
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        db.close()
    
    background_tasks.add_task(predict_model_background, db_file_id, file_path, model_id)
    return ret
    
@app.post("/api/predictions")
async def list_predictions(body: PageRequest): 
    page = max(1, body.page)
    page_size = max(1, body.page_size)

    db = SessionLocal()
    try:
        total = db.query(PredictionHistory).count()
        offset = (page - 1) * page_size
        records = db.query(PredictionHistory).order_by(PredictionHistory.predict_date.desc()).offset(offset).limit(page_size).all()

        items = []
        for r in records:
            items.append({
                "id": r.id,
                "train_model_id": r.train_model_id,
                "train_model_name": getattr(r, "train_model_name", None),
                "result1_path": getattr(r, "result1_path", None),   
                "result2_path": getattr(r, "result2_path", None),
                "predict_date": getattr(r, "predict_date", None),
                "status": getattr(r, "status", None)
            }) 

        return {
            "page": page,
            "page_size": page_size,
            "total": total,
            "records": items
        }
    finally:
        db.close()

@app.post("/api/models/list")
async def list_models(body: PageRequest):
    page = max(1, body.page)
    page_size = max(1, body.page_size)

    db = SessionLocal()
    try:
        total = db.query(TrainModel).count()
        offset = (page - 1) * page_size
        records = db.query(TrainModel).order_by(TrainModel.train_date.desc()).offset(offset).limit(page_size).all()

        items = []
        for r in records:
            items.append({
                "id": r.id,
                "file_id": r.file_id,
                "record_number": r.record_number,
                "accuracy": getattr(r, "accuracy", None),   
                "recall_rate": getattr(r, "recall_rate", None),
                "precision": getattr(r, "precision", None),
                "model_name": getattr(r, "model_name", None),
                "train_date": getattr(r, "train_date", None)
            })

        return {
            "page": page,
            "page_size": page_size,
            "total": total,
            "records": items
        }
    finally:
        db.close()



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
async def delete_file(file_id: int):
    db = SessionLocal()
    try:
        # 查找记录
        file_record = db.query(FileUpload).filter(FileUpload.id == file_id).first()
        if not file_record:
            raise HTTPException(status_code=404, detail=f"File with id {file_id} not found")
        
        # 如果文件存在，删除物理文件
        if os.path.exists(file_record.file_path):
            os.remove(file_record.file_path)
        
        # 删除数据库记录
        db.delete(file_record)
        db.commit()
        
        return {"message": f"File with id {file_id} has been deleted"}
    finally:
        db.close()

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
                host="127.0.0.1", 
                port=8000, 
                log_level="info",
                reload=True)