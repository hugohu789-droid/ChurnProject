from datetime import datetime
from pydantic import BaseModel


class FileUploadResponse(BaseModel):
    id: int
    original_filename: str
    saved_filename: str
    file_path: str
    file_size: int
    status: str
    upload_time: datetime

    model_config = {"from_attributes": True}
