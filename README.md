# ML Platform — Self-Service Model Registry & Inference

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
![Vue.js](https://img.shields.io/badge/frontend-Vue.js%203-42b883)
![TypeScript](https://img.shields.io/badge/language-TypeScript-3178c6)
![FastAPI](https://img.shields.io/badge/backend-FastAPI-009688)
![Python](https://img.shields.io/badge/language-Python%203.9+-3776ab)
![AWS](https://img.shields.io/badge/deployment-AWS%20EC2-FF9900)
![Docker](https://img.shields.io/badge/container-Docker-2496ED)

## 📖 About The Project

A full-stack platform for registering trained ML models, browsing the registry, running predictions on new datasets, and downloading results.

Unlike simple demo scripts, this project demonstrates an **end-to-end Data Science Lifecycle implementation**—from data ingestion and asynchronous model training to RESTful API deployment and interactive visualization. It is engineered with a focus on **Type Safety**, **Scalability**, and **DevOps Automation**.

### ✨ Key Features

* **📊 Interactive Dashboard**: Real-time visualization of system metrics and model performance (Accuracy/Precision) built with **Element Plus**.
* **🧠 Asynchronous Training**: Leverages **FastAPI BackgroundTasks** to handle resource-intensive ML training jobs without blocking the main thread.
* **🔄 End-to-End Pipeline**: Seamless flow from raw CSV upload -> Data Cleaning -> Training -> Inference -> Reporting.
* **🛡️ Type-Safe Architecture**: Full TypeScript implementation on the frontend synced with Pydantic models on the backend.
* **🚀 Automated DevOps**: A custom **GitHub Actions CI/CD pipeline** that automates testing and deployment to AWS EC2 using an efficient on-premise build strategy.

---

## 🛠️ Technical Stack

- **Backend**: Python 3.11, FastAPI, SQLAlchemy, Pydantic
- **Frontend**: Vue 3 + TypeScript, Vite, Pinia, Element Plus
- **ML**: scikit-learn, pandas, numpy
- **Infra**: Docker Compose, AWS EC2, Nginx
- **CI/CD**: GitHub Actions (Pytest + Flake8)

---

## 🏗️ System Architecture & Deployment

The application follows a **Microservices-ready** architecture deployed on a single AWS EC2 instance. **Nginx** acts as the reverse proxy and static file server, ensuring secure and efficient traffic routing.

### 1. Runtime Architecture
The application follows a **Microservices-ready** architecture deployed on a single AWS EC2 instance using Docker Compose. **Nginx** acts as the reverse proxy and static file server, ensuring secure and efficient traffic routing.

```mermaid
graph TD
    User([User / Client]) -->|HTTPS / Port 443| Nginx[Nginx Reverse Proxy]
    
    subgraph AWS_EC2 [AWS EC2 Instance]
        style AWS_EC2 fill:#f9f9f9,stroke:#333,stroke-width:2px
        
        Nginx -->|Serve Static Assets| Vue[Frontend Container<br />Vue 3 / Vite]
        Nginx -->|Proxy /api| FastAPI[Backend Container<br/>FastAPI / Uvicorn]
        
        FastAPI <-->|Read/Write| DB[(Database<br/>SQLite/PostgreSQL)]
        FastAPI -->|Async Processing| BG[Background Tasks<br/>Model Training]
        
        BG -.->|Load/Save| ModelStore[Model Artifacts<br/>Disk Storage]
    end
```

### 2. CI/CD Pipeline (Automated Deployment)
To optimize costs for this project, I implemented an "On-Premise Build" strategy instead of using an external container registry. The pipeline ensures that only verified code reaches the production server.

Workflow Steps:

**Validation**: GitHub Actions runs parallel tests for Frontend (Jest) and Backend (Pytest).

**Transfer**: Verified artifacts are securely transferred to AWS EC2 via SCP.

**Live Build**: Docker images are built directly on the server to ensure environment consistency.

```mermaid
sequenceDiagram
    participant Dev as Developer
    participant GH as GitHub Actions
    participant EC2 as AWS Production Server
    
    Dev->>GH: Push Code to 'main'
    
    Note over GH: 1. Parallel Validation
    par Testing
        GH->>GH: Frontend Unit Tests
    and Linting
        GH->>GH: Backend Pytest & Flake8
    end
    
    Note over GH, EC2: 2. Secure Transfer (SCP)
    GH->>EC2: Copy Backend Code + /dist
    
    Note over EC2: 3. Remote Build & Deploy
    GH->>EC2: SSH Trigger
    activate EC2
    EC2->>EC2: docker compose up -d --build
    deactivate EC2
    
    EC2-->>Dev: Deployment Successful
```
---

## 🚀 Getting Started

Follow these steps to set up the project locally for development.

### Prerequisites

*   **Node.js** (v18+) & npm
*   **Python** (v3.9+)
*   **Git**

### 1. Clone the Repository

```bash
git clone [https://github.com/hugohu789-droid/ml-platform.git](https://github.com/hugohu789-droid/ml-platform.git)
cd ml-platform
```

### 2. Backend Setup

Navigate to the backend directory, set up a virtual environment, and install dependencies.

```bash
cd backend

# Create & activate virtual environment
python -m venv venv
source venv/bin/activate  # Windows: .\venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the server (Hot Reload enabled)
uvicorn app.churn_api:app --reload --host 0.0.0.0 --port 8000
```
*The API will be available at `http://localhost:8000`*
*API Docs: `http://localhost:8000/docs`*

### 3. Frontend Setup

```bash
cd frontend

# Install dependencies
npm install

# Start development server
npm run dev
```
*The application will be available at `http://localhost:5173`*

---

## 📂 Project Structure

```text
ml-platform/
├── .github/workflows/  # CI/CD Pipeline definitions
├── backend/
│   ├── app/
│   │   ├── tests/   # tests/ directory
│   │       ├── test_api.py    #test code for the API
│   │   ├── churn_api.py    # FastAPI entry point & Routes
│   │   ├── models.py       # Pydantic & SQLAlchemy Models
│   │   └── modeltrain.py   # Core ML Logic (Scikit-Learn)
│   ├── requirements.txt
│   └── docker-compose.yml      # Container Orchestration
├── frontend/
│   ├── src/
│   │   ├── api/            # TypeScript API Service layer
│   │   ├── views/          # Vue Components (Dashboard, Prediction)
│   │   ├── stores/         # Pinia State Management (Optional)
│   │   └── types/          # TypeScript Interfaces
│   ├── package.json
└── README.md
```

---

## 🔮 Future Improvements

**Authentication**: Implement JWT (JSON Web Token) based login system for multi-user support.

**Advanced Monitoring**: Integrate Prometheus and Grafana for real-time server metrics.

**Model Versioning**: Implement MLflow to track model experiments and versions.

**Caching**: Introduce Redis to cache prediction results and reduce latency.

---

## 👤 Author

**Hugo**
* **Role**: Senior Software Developer (C++ / Java / C# / Python / Full Stack)
* **GitHub**: [@hugohu789-droid]()

---

*This project is for educational and demonstration purposes.*