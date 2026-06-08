import sys
import os

# Add backend directory to sys.path so we can import app
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fastapi.testclient import TestClient
from app.main import app
from app.database import Base, engine

client = TestClient(app)

def test_api_endpoints():
    print("="*60)
    print("STARTING FASTAPI BACKEND & ML MODEL INTEGRATION TESTS")
    print("="*60)
    
    # 1. Health check test
    print("\n---> Testing GET /health...")
    response = client.get("/health")
    assert response.status_code == 200
    res_data = response.json()
    print("Response:", res_data)
    assert res_data["status"] == "healthy"
    assert res_data["ml_engine_ready"] is True
    print("[OK] Health check passed.")
    
    # 2. Prediction request test
    print("\n---> Testing POST /predict (First Assessment)...")
    payload = {
        "patient_id": "PAT-999",
        "patient_name": "Test Patient",
        "age": 42,
        "gender": "Male",
        "rr_interval": 1084.0,
        "pp_interval": 1090.0,
        "qt_interval": 448.0
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    res_data = response.json()
    print("Response:", res_data)
    assert "record" in res_data
    assert "suggestions" in res_data
    assert len(res_data["suggestions"]) > 0
    assert res_data["record"]["patient_id"] == "PAT-999"
    print(f"[OK] Prediction passed. Risk level: {res_data['record']['risk_level']} ({res_data['record']['risk_score']}%)")
    
    # 3. Second prediction for same patient (tests sliding window features)
    print("\n---> Testing POST /predict (Second Assessment for same patient)...")
    payload_2 = {
        "patient_id": "PAT-999",
        "patient_name": "Test Patient",
        "age": 42,
        "gender": "Male",
        "rr_interval": 960.0,
        "pp_interval": 965.0,
        "qt_interval": 414.0
    }
    response = client.post("/predict", json=payload_2)
    assert response.status_code == 200
    res_data_2 = response.json()
    print("Response:", res_data_2)
    assert res_data_2["record"]["rr_interval"] == 960.0
    print("[OK] Rolling feature engineering prediction passed.")
    
    # 4. History endpoint test
    print("\n---> Testing GET /history...")
    response = client.get("/history?search=Test")
    assert response.status_code == 200
    history_list = response.json()
    print(f"Found {len(history_list)} records matching 'Test'.")
    assert len(history_list) >= 2
    print("[OK] History listing passed.")
    
    # 5. Stats endpoint test
    print("\n---> Testing GET /history/stats...")
    response = client.get("/history/stats")
    assert response.status_code == 200
    stats = response.json()
    print("Stats:", stats)
    assert stats["total_assessments"] >= 2
    print("[OK] Dashboard stats calculation passed.")
    
    print("\n" + "="*60)
    print("ALL API INTEGRATION TESTS PASSED SUCCESSFULLY!")
    print("="*60 + "\n")

if __name__ == "__main__":
    # Ensure test database is clean or set up
    Base.metadata.create_all(bind=engine)
    try:
        test_api_endpoints()
    except AssertionError as e:
        print("\n[FAIL] Test assertion error occurred!")
        sys.exit(1)
    except Exception as e:
        print(f"\n[FAIL] Unexpected error: {str(e)}")
        sys.exit(1)
