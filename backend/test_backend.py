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
    
    # 1.1 Register test user
    print("\n---> Testing POST /auth/register...")
    reg_payload = {
        "email": "testuser@cardio.org",
        "password": "testpassword123",
        "full_name": "Dr. Test User"
    }
    response = client.post("/auth/register", json=reg_payload)
    if response.status_code == 400:
        print("User already exists, attempting login...")
        login_payload = {
            "email": "testuser@cardio.org",
            "password": "testpassword123"
        }
        response = client.post("/auth/login", json=login_payload)
        assert response.status_code == 200
    else:
        assert response.status_code == 200
        
    auth_data = response.json()
    assert "token" in auth_data
    assert "user" in auth_data
    token = auth_data["token"]
    headers = {"Authorization": f"Bearer {token}"}
    print("[OK] Authentication setup passed. Token:", token)

    # 2. Prediction request test
    print("\n---> Testing POST /predict (First Assessment)...")
    payload = {
        "patient_id": "PAT-999",
        "patient_name": "Test Patient",
        "age": 42.0,
        "gender": "Male",
        "weight": 75.0,
        "height": 182.0,
        "heart_rate": 70.0,
        "systolic_bp": 122.0,
        "diastolic_bp": 80.0,
        "sport_type": "ATH",
        "rr_interval": 857.0,
        "pp_interval": 850.0,
        "qt_interval": 400.0,
        "qtc_interval": 420.0,
        "qrs_duration": 96.0,
        "pq_interval": 150.0,
        "family_history_heart_disease": 1.0,
        "personal_history_heart_disease": 0.0,
        "syncope": 0.0,
        "pectus_excavatum": 0.0
    }
    response = client.post("/predict", json=payload, headers=headers)
    if response.status_code != 200:
        print("FAIL DETAILED RESPONSE:", response.status_code, response.text)
    assert response.status_code == 200
    res_data = response.json()
    print("Response keys:", list(res_data.keys()))
    assert "record" in res_data
    assert "suggestions" in res_data
    assert "contributions" in res_data
    assert len(res_data["suggestions"]) > 0
    assert len(res_data["contributions"]) > 0
    assert res_data["record"]["patient_id"] == "PAT-999"
    print(f"[OK] Prediction passed. Risk level: {res_data['record']['risk_level']} ({res_data['record']['risk_score']}%)")
    
    # 3. Second prediction for same patient
    print("\n---> Testing POST /predict (Second Assessment for same patient)...")
    payload_2 = {
        "patient_id": "PAT-999",
        "patient_name": "Test Patient",
        "age": 42.0,
        "gender": "Male",
        "weight": 75.0,
        "height": 182.0,
        "heart_rate": 75.0,
        "systolic_bp": 120.0,
        "diastolic_bp": 80.0,
        "sport_type": "ATH",
        "rr_interval": 800.0,
        "pp_interval": 800.0,
        "qt_interval": 410.0,
        "qtc_interval": 425.0,
        "qrs_duration": 94.0,
        "pq_interval": 145.0,
        "family_history_heart_disease": 1.0,
        "personal_history_heart_disease": 0.0,
        "syncope": 0.0,
        "pectus_excavatum": 0.0
    }
    response = client.post("/predict", json=payload_2, headers=headers)
    assert response.status_code == 200
    res_data_2 = response.json()
    assert res_data_2["record"]["rr_interval"] == 800.0
    print("[OK] Multi-feature prediction passed.")
    
    # 4. History endpoint test
    print("\n---> Testing GET /history...")
    response = client.get("/history?search=Test", headers=headers)
    assert response.status_code == 200
    history_list = response.json()
    print(f"Found {len(history_list)} records matching 'Test'.")
    assert len(history_list) >= 2
    print("[OK] History listing passed.")
    
    # 5. Stats endpoint test
    print("\n---> Testing GET /history/stats...")
    response = client.get("/history/stats", headers=headers)
    assert response.status_code == 200
    stats = response.json()
    print("Stats:", stats)
    assert stats["total_assessments"] >= 2
    print("[OK] Dashboard stats calculation passed.")
 
    # 6. Correlations endpoint test
    print("\n---> Testing GET /history/correlations...")
    response = client.get("/history/correlations")
    assert response.status_code == 200
    corrs = response.json()
    print(f"Fetched {len(corrs)} dataset feature correlation strength rankings.")
    assert len(corrs) > 0
    print("[OK] Correlation list endpoint passed.")
    
    # 7. Model Info endpoint test
    print("\n---> Testing GET /model-info...")
    response = client.get("/model-info")
    assert response.status_code == 200
    model_info = response.json()
    print("Model Info:", model_info)
    assert "algorithm" in model_info
    assert "accuracy" in model_info
    print("[OK] Model info endpoint passed.")
 
    # 8. Analytics endpoint test
    print("\n---> Testing GET /analytics...")
    response = client.get("/analytics", headers=headers)
    assert response.status_code == 200
    analytics = response.json()
    print("Analytics:", analytics)
    assert "total_predictions" in analytics
    assert "risk_distribution" in analytics
    assert "most_common_risk_factors" in analytics
    print("[OK] Analytics endpoint passed.")
 
    # 9. PDF Report generation endpoint test
    print("\n---> Testing POST /generate-report...")
    record_id = history_list[0]["id"]
    response = client.post("/generate-report", json={"record_id": record_id}, headers=headers)
    assert response.status_code == 200
    assert response.headers["content-type"] == "application/pdf"
    pdf_content = response.content
    print(f"Generated PDF file size: {len(pdf_content)} bytes")
    assert len(pdf_content) > 1000
    print("[OK] PDF report generation passed.")
    
    print("\n" + "="*60)
    print("ALL API INTEGRATION TESTS PASSED SUCCESSFULLY!")
    print("="*60 + "\n")
 
if __name__ == "__main__":
    # Ensure test database is clean or set up
    Base.metadata.create_all(bind=engine)
    
    # Pre-test cleanup to remove any stale test patient and user data
    from app.database import SessionLocal
    from app.models import PredictionRecord, User
    print("Pre-test cleanup: removing old test patient and user data...")
    db = SessionLocal()
    try:
        db.query(PredictionRecord).filter(
            (PredictionRecord.patient_id == "PAT-999") | 
            (PredictionRecord.patient_name == "Test Patient")
        ).delete()
        db.query(User).filter(User.email == "testuser@cardio.org").delete()
        db.commit()
    except Exception as e:
        db.rollback()
        print(f"Warning: Pre-test cleanup failed: {e}")
    finally:
        db.close()
 
    try:
        test_api_endpoints()
    except AssertionError as e:
        print("\n[FAIL] Test assertion error occurred!")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    except Exception as e:
        print(f"\n[FAIL] Unexpected error: {str(e)}")
        sys.exit(1)
    finally:
        # Post-test cleanup to leave the database clean
        print("Post-test cleanup: removing test patient and user data...")
        db = SessionLocal()
        try:
            deleted_records = db.query(PredictionRecord).filter(
                (PredictionRecord.patient_id == "PAT-999") | 
                (PredictionRecord.patient_name == "Test Patient")
            ).delete()
            deleted_user = db.query(User).filter(User.email == "testuser@cardio.org").delete()
            db.commit()
            print(f"Successfully cleaned up {deleted_records} test records and {deleted_user} test users.")
        except Exception as e:
            db.rollback()
            print(f"Warning: Post-test cleanup failed: {e}")
        finally:
            db.close()

