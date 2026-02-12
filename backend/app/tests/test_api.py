# ./ChurnProject\backend\app\tests\test_api.py
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool
import sys
import os

# Ensure that the app module can be imported
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from churn_api import app, get_db
from models import Base

# 1. Configure a test in-memory database (SQLite)
SQLALCHEMY_DATABASE_URL = "sqlite:///:memory:"

engine = create_engine(
    SQLALCHEMY_DATABASE_URL,
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# 2. Override the get_db dependency and use the test database.
def override_get_db():
    try:
        db = TestingSessionLocal()
        yield db
    finally:
        db.close()

app.dependency_overrides[get_db] = override_get_db

client = TestClient(app)

# 3. Initialize database table structure
def setup_module(module):
    Base.metadata.create_all(bind=engine)

# --- Test cases ---

def test_read_root():
    """Test if the root path is normal"""
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "File Upload API is running"}

def test_dashboard_stats_empty():
    """Dashboard statistics interface when testing an empty database"""
    response = client.get("/api/dashboard/stats")
    assert response.status_code == 200
    data = response.json()
    
    # The validation returned data structure and initial values
    assert "total_files" in data
    assert data["total_files"] == 0
    assert data["total_models"] == 0
    assert data["average_accuracy"] == 0.0
    assert isinstance(data["recent_models"], list)

def test_upload_file_invalid_extension():
    """Test whether uploading non-CSV files is rejected"""
    response = client.post(
        "/api/upload",
        files={"file": ("test.txt", b"some content", "text/plain")}
    )
    assert response.status_code == 400
    assert response.json()["detail"] == "Only CSV files are allowed"
