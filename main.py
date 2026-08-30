from fastapi import FastAPI, HTTPException
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
    medications_gujarati: str = ""
    followup_date: str = ""

class DictationSchema(BaseModel):
    raw_text: str

@app.get("/")
def health_check():
    return {"status": "online", "app": "DocScribe Cloud Engine"}

@app.post("/api/v1/ai/process-dictation")
def process_dictation(data: DictationSchema):
    if not gemini_client:
        raise HTTPException(status_code=500, detail="GEMINI_API_KEY is missing on server.")
    
    prompt = f"""
    You are an expert AI clinical documentation assistant for an Indian General Practice OPD clinic.
    Extract and structure the following clinical dictation into raw JSON format.

    JSON Keys required:
    "vitals": (string, e.g. "BP: 120/80, Pulse: 72"),
    "diagnosis": (string, primary clinical diagnosis),
    "soap_note": (string, clinical history & examination summary),
    "medications": (string, clear dosage and duration in English),
    "medications_gujarati": (string, clear translation of dosage instructions in Gujarati script for patient clarity),
    "followup_date": (string, YYYY-MM-DD format if mentioned, else empty string)

    Dictation Text:
    {data.raw_text}

    Return ONLY raw valid JSON with no markdown formatting.
    """

    try:
        response = gemini_client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
        )
        cleaned_json = response.text.strip().replace("```json", "").replace("```", "")
        return json.loads(cleaned_json)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"AI processing failed: {str(e)}")

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
            "medications_gujarati": data.medications_gujarati,
            "followup_date": data.followup_date
        }).execute()
        return {"status": "success", "data": response.data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
