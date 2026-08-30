from fastapi import FastAPI, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from supabase import create_client, Client
from google import genai
import os
import json

app = FastAPI(title="DocScribe EMR Pro Engine")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if SUPABASE_URL and SUPABASE_KEY:
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

gemini_client = None
if GEMINI_API_KEY:
    gemini_client = genai.Client(api_key=GEMINI_API_KEY)

class AuthSchema(BaseModel):
    email: str
    password: str

class DictationSchema(BaseModel):
    raw_text: str

class ConsultationSchema(BaseModel):
    doctor_id: str
    doctor_name: str = ""
    doctor_qualification: str = ""
    hospital_name: str = ""
    patient_name: str
    patient_phone: str
    vitals: str = ""
    diagnosis: str = ""
    soap_note: str = ""
    medications: str = ""
    followup_date: str = ""

@app.get("/")
def health_check():
    return {"status": "online", "app": "DocScribe Cloud Engine"}

# --- AUTHENTICATION ENDPOINTS ---

@app.post("/api/v1/auth/signup")
def signup(data: AuthSchema):
    try:
        res = supabase.auth.sign_up({"email": data.email, "password": data.password})
        if not res.user:
            raise HTTPException(status_code=400, detail="Signup failed.")
        return {"status": "success", "user_id": res.user.id, "email": res.user.email}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/api/v1/auth/login")
def login(data: AuthSchema):
    try:
        res = supabase.auth.sign_in_with_password({"email": data.email, "password": data.password})
        return {
            "status": "success",
            "access_token": res.session.access_token,
            "user_id": res.user.id,
            "email": res.user.email
        }
    except Exception as e:
        raise HTTPException(status_code=401, detail="Invalid Email or Password.")

# --- CLINICAL ENDPOINTS ---

@app.post("/api/v1/ai/process-dictation")
def process_dictation(data: DictationSchema):
    if not gemini_client:
        raise HTTPException(status_code=500, detail="GEMINI_API_KEY is missing on server.")
    
    prompt = f"""
    You are an expert AI clinical documentation assistant for an OPD clinic.
    Extract and structure the following dictation into raw JSON format.

    JSON Keys required:
    "patient_name": (string, patient name if mentioned, else empty string),
    "patient_phone": (string, phone number if mentioned, else empty string),
    "vitals": (string, blood pressure / vitals if mentioned),
    "diagnosis": (string, primary clinical diagnosis),
    "soap_note": (string, clinical SOAP notes structure),
    "medications": (string, prescribed drug list with dosage and duration),
    "followup_date": (string, YYYY-MM-DD format if mentioned, else empty string)

    Clinical Dictation:
    {data.raw_text}

    Return ONLY raw valid JSON with no markdown tags.
    """

    candidate_models = ["gemini-3.6-flash", "gemini-2.5-flash", "gemini-1.5-flash"]
    last_error = None
    for model_name in candidate_models:
        try:
            response = gemini_client.models.generate_content(
                model=model_name,
                contents=prompt,
            )
            cleaned_json = response.text.strip().replace("```json", "").replace("```", "")
            return json.loads(cleaned_json)
        except Exception as e:
            last_error = e
            continue

    raise HTTPException(status_code=500, detail=f"AI processing failed on all models: {str(last_error)}")

@app.post("/api/v1/consultations/save")
def save_consultation(data: ConsultationSchema):
    try:
        response = supabase.table("consultations").insert({
            "doctor_id": data.doctor_id,
            "doctor_name": data.doctor_name,
            "doctor_qualification": data.doctor_qualification,
            "hospital_name": data.hospital_name,
            "patient_name": data.patient_name,
            "patient_phone": data.patient_phone,
            "vitals": data.vitals,
            "diagnosis": data.diagnosis,
            "soap_note": data.soap_note,
            "medications": data.medications,
            "followup_date": data.followup_date
        }).execute()
        return {"status": "success", "data": response.data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/consultations/search")
def search_consultations(doctor_id: str, query: str):
    try:
        response = supabase.table("consultations") \
            .select("*") \
            .eq("doctor_id", doctor_id) \
            .or_(f"patient_name.ilike.%{query}%,patient_phone.ilike.%{query}%") \
            .order("created_at", desc=True) \
            .execute()
        return {"status": "success", "data": response.data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/analytics/summary")
def get_analytics_summary(doctor_id: str):
    try:
        response = supabase.table("consultations") \
            .select("diagnosis") \
            .eq("doctor_id", doctor_id) \
            .execute()
        
        records = response.data
        total_count = len(records)
        
        if total_count == 0:
            return {
                "total_consultations": 0,
                "top_diagnosis": "N/A",
                "breakdown": {}
            }
        
        counts = {}
        for r in records:
            diag = r.get("diagnosis") or "General OPD"
            counts[diag] = counts.get(diag, 0) + 1
            
        top_diag = max(counts, key=counts.get)
        
        return {
            "total_consultations": total_count,
            "top_diagnosis": top_diag,
            "breakdown": counts
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
