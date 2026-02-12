# Customer Churn Prediction System

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Vue.js](https://img.shields.io/badge/frontend-Vue.js%203-42b883)
![TypeScript](https://img.shields.io/badge/language-TypeScript-3178c6)
![FastAPI](https://img.shields.io/badge/backend-FastAPI-009688)
![Python](https://img.shields.io/badge/language-Python%203.9+-3776ab)

## 📖 About The Project

The **Customer Churn Prediction System** is a full-stack machine learning application designed to help businesses identify customers likely to cancel their subscriptions. By leveraging historical data, the system trains predictive models and provides actionable insights through an interactive dashboard.

This project demonstrates an end-to-end implementation of a Data Science lifecycle, from data ingestion and model training to deployment and inference, wrapped in a modern, type-safe web interface.

### Key Features

*   **📊 Interactive Dashboard**: Real-time overview of system metrics, model performance (Accuracy/Precision), and recent activities.
*   **📁 Data Management**: Secure CSV file upload handling with validation and history tracking.
*   **🤖 Automated Model Training**: Asynchronous background tasks (using FastAPI `BackgroundTasks`) to train models without blocking the UI.
*   **🔮 Batch Prediction**: Select trained models to run predictions on new datasets and generate downloadable reports.
*   **🎨 Modern UI/UX**: A responsive interface built with **Vue 3**, **TypeScript**, and **Element Plus**, featuring a "Tech Theme" design.

---

## 🛠️ Technical Architecture

### Frontend (Client)
Built with a focus on component reusability and type safety.
*   **Framework**: Vue.js 3 (Composition API)
*   **Language**: TypeScript
*   **Build Tool**: Vite
*   **UI Library**: Element Plus
*   **State/Routing**: Vue Router
*   **HTTP Client**: Axios (configured with interceptors)

### Backend (Server)
Engineered for performance and scalability.
*   **Framework**: FastAPI (Python)
*   **Database ORM**: SQLAlchemy
*   **Data Processing**: Pandas, Scikit-learn
*   **Task Management**: Asynchronous background processing for long-running training jobs.
*   **API Documentation**: Auto-generated Swagger/OpenAPI.

---

## 🚀 Getting Started

Follow these steps to set up the project locally for development and testing.

### Prerequisites

*   **Node.js** (v16+) and npm
*   **Python** (v3.9+)
*   **Git**

### 1. Clone the Repository

```bash
git clone https://github.com/hugohu789-droid/ChurnProject.git
cd ChurnProject
```

### 2. Backend Setup

Navigate to the backend directory, set up a virtual environment, and install dependencies.

```bash
cd backend

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
.\venv\Scripts\activate
# On macOS/Linux:
# source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run the server
uvicorn churn_api:app --reload --host 0.0.0.0 --port 8000
```
*The API will be available at `http://localhost:8000`*
*API Docs: `http://localhost:8000/docs`*

### 3. Frontend Setup

Open a new terminal, navigate to the frontend directory, and start the development server.

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
ChurnProject/
├── backend/
│   ├── app/
│   │   ├── churn_api.py    # Main FastAPI application entry point
│   │   ├── models.py       # SQLAlchemy database models
│   │   ├── modeltrain.py   # ML logic for training and prediction
│   │   └── ...
│   └── requirements.txt
├── frontend/
│   ├── src/
│   │   ├── api/            # TypeScript API definitions
│   │   ├── views/          # Vue Page components (Dashboard, Training, etc.)
│   │   ├── App.vue         # Root component with global styles
│   │   └── ...
│   └── package.json
└── README.md
```

---

## 🔮 Future Improvements

*   **Containerization**: Full Docker support is currently in the `deploy` folder (using Docker Compose).
*   **Authentication**: Implement JWT (JSON Web Token) based login system.
*   **Visualization**: Add ECharts to the dashboard for visualizing churn trends over time.
*   **CI/CD**: GitHub Actions workflow for automated testing and linting.

---

## 👤 Author

**Hugo**
*   GitHub: @hugohu789-droid

---

*This project is for educational and demonstration purposes.*