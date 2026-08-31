from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from supabase import create_client, Client
from google import genai
from google.genai import types
import os
import json
import urllib.request
import xml.etree.ElementTree as ET

app = FastAPI(title="OPD Express AI Engine")

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
    doctor_reg_no: str = ""
    hospital_name: str = ""
    patient_name: str
    patient_age: str = ""
    patient_sex: str = ""
    patient_phone: str
    vitals: str = ""
    diagnosis: str = ""
    soap_note: str = ""
    medications: str = ""
    followup_date: str = ""

class CustomMedicineSchema(BaseModel):
    doctor_id: str
    brand_name: str
    dosage: str = ""
    frequency: str = "BD"
    duration: str = ""

class ForumPostSchema(BaseModel):
    doctor_id: str
    doctor_name: str
    title: str
    content: str
    category: str = "General Case"

class ForumCommentSchema(BaseModel):
    post_id: str
    doctor_id: str
    doctor_name: str
    comment: str

@app.get("/")
def health_check():
    return {"status": "online", "app": "OPD Express AI Engine"}

# --- AUTHENTICATION ---

@app.post("/api/v1/auth/signup")
def signup(data: AuthSchema):
    try:
        res = supabase.auth.admin.create_user({
            "email": data.email,
            "password": data.password,
            "email_confirm": True
        })
        if not res.user:
            raise HTTPException(status_code=400, detail="Registration failed.")
        return {"status": "success", "user_id": res.user.id, "email": res.user.email}
    except Exception:
        try:
            res_std = supabase.auth.sign_up({"email": data.email, "password": data.password})
            if not res_std.user:
                raise HTTPException(status_code=400, detail="Registration failed.")
            return {"status": "success", "user_id": res_std.user.id, "email": res_std.user.email}
        except Exception as err:
            raise HTTPException(status_code=400, detail=str(err))

@app.post("/api/v1/auth/login")
def login(data: AuthSchema):
    try:
        res = supabase.auth.sign_in_with_password({"email": data.email, "password": data.password})
        if not res.session:
            raise HTTPException(status_code=401, detail="Invalid email or password.")
        return {
            "status": "success",
            "access_token": res.session.access_token,
            "user_id": res.user.id,
            "email": res.user.email
        }
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid credentials or email unconfirmed.")

# --- AI PROCESSING ENGINE ---

PROMPT_INSTRUCTION = """
You are an expert AI clinical documentation assistant for an OPD clinic.
Extract clinical information (which may be in English, Hindi, or Gujarati) and format into RAW JSON.

JSON Keys required:
"patient_name": (string, patient name if mentioned, else ""),
"patient_age": (string, e.g. "35 Y" or "5 M" if mentioned, else ""),
"patient_sex": (string, "Male", "Female", or "Other" if mentioned, else ""),
"patient_phone": (string, phone number if mentioned, else ""),
"v_bp": (string, blood pressure e.g. "120/80" if mentioned, else ""),
"v_pulse": (string, pulse rate e.g. "72" if mentioned, else ""),
"v_spo2": (string, SpO2 percentage e.g. "98" if mentioned, else ""),
"v_temp": (string, temperature e.g. "98.6" if mentioned, else ""),
"v_rbs": (string, random blood sugar e.g. "110" if mentioned, else ""),
"diagnosis": (string, primary clinical diagnosis),
"soap_note": (string, clinical SOAP notes structure),
"medications": (string, prescribed drug list with dosage and duration),
"followup_date": (string, YYYY-MM-DD format if mentioned, else "")

Return ONLY valid raw JSON without markdown codeblock backticks.
"""

# Candidate models used in fallback order
CANDIDATE_MODELS = ["gemini-3.6-flash", "gemini-3.5-flash", "gemini-2.5-flash"]

@app.post("/api/v1/ai/process-dictation")
def process_dictation(data: DictationSchema):
    if not gemini_client:
        raise HTTPException(status_code=500, detail="GEMINI_API_KEY missing on server.")
    
    prompt = f"{PROMPT_INSTRUCTION}\n\nClinical Dictation Text:\n{data.raw_text}"
    last_error = None

    for model_name in CANDIDATE_MODELS:
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

    raise HTTPException(status_code=500, detail=f"AI processing failed across all models: {str(last_error)}")

@app.post("/api/v1/ai/process-audio")
async def process_audio(file: UploadFile = File(...)):
    if not gemini_client:
        raise HTTPException(status_code=500, detail="GEMINI_API_KEY missing on server.")
    
    try:
        content = await file.read()
        mime_type = file.content_type or "audio/webm"
        
        # Pass audio bytes directly in-memory to prevent file path errors
        audio_part = types.Part.from_bytes(data=content, mime_type=mime_type)
        contents = [audio_part, PROMPT_INSTRUCTION]

        last_error = None
        for model_name in CANDIDATE_MODELS:
            try:
                response = gemini_client.models.generate_content(
                    model=model_name,
                    contents=contents
                )
                cleaned_json = response.text.strip().replace("```json", "").replace("```", "")
                return json.loads(cleaned_json)
            except Exception as e:
                last_error = e
                continue

        raise HTTPException(status_code=500, detail=f"AI audio processing failed across all models: {str(last_error)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Audio processing error: {str(e)}")

# --- CONSULTATIONS & MEDICINES MASTER ---

@app.post("/api/v1/consultations/save")
def save_consultation(data: ConsultationSchema):
    try:
        response = supabase.table("consultations").insert({
            "doctor_id": data.doctor_id,
            "doctor_name": data.doctor_name,
            "doctor_qualification": data.doctor_qualification,
            "doctor_reg_no": data.doctor_reg_no,
            "hospital_name": data.hospital_name,
            "patient_name": data.patient_name,
            "patient_age": data.patient_age,
            "patient_sex": data.patient_sex,
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

@app.post("/api/v1/medicines/add")
def add_custom_medicine(data: CustomMedicineSchema):
    try:
        response = supabase.table("medicines").insert({
            "doctor_id": data.doctor_id,
            "brand_name": data.brand_name,
            "dosage": data.dosage,
            "frequency": data.frequency,
            "duration": data.duration
        }).execute()
        return {"status": "success", "data": response.data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/medicines/list")
def list_custom_medicines(doctor_id: str):
    try:
        response = supabase.table("medicines").select("*").eq("doctor_id", doctor_id).execute()
        return {"status": "success", "data": response.data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/analytics/summary")
def get_analytics_summary(doctor_id: str):
    try:
        response = supabase.table("consultations").select("diagnosis").eq("doctor_id", doctor_id).execute()
        records = response.data
        total_count = len(records)
        if total_count == 0:
            return {"total_consultations": 0, "top_diagnosis": "N/A", "breakdown": {}}
        
        counts = {}
        for r in records:
            diag = r.get("diagnosis") or "General OPD"
            counts[diag] = counts.get(diag, 0) + 1
            
        top_diag = max(counts, key=counts.get)
        return {"total_consultations": total_count, "top_diagnosis": top_diag, "breakdown": counts}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# --- MEDICAL NEWS RSS FEED ---

@app.get("/api/v1/news/medical")
def get_medical_news():
    rss_url = "https://news.google.com/rss/search?q=medical+science+clinical+guidelines+ICMR+WHO&hl=en-IN&gl=IN&ceid=IN:en"
    try:
        req = urllib.request.Request(rss_url, headers={'User-Agent': 'Mozilla/5.0'})
        html = urllib.request.urlopen(req).read()
        root = ET.fromstring(html)
        
        articles = []
        for item in root.findall('.//channel/item')[:10]:
            title = item.find('title').text if item.find('title') is not None else "Medical Update"
            link = item.find('link').text if item.find('link') is not None else "#"
            pubDate = item.find('pubDate').text if item.find('pubDate') is not None else ""
            articles.append({
                "title": title,
                "link": link,
                "pubDate": pubDate
            })
        return {"status": "success", "articles": articles}
    except Exception as e:
        return {"status": "error", "articles": [], "detail": str(e)}

# --- DOCTORS' LOUNGE FORUM ---

@app.post("/api/v1/forum/posts/create")
def create_forum_post(data: ForumPostSchema):
    try:
        res = supabase.table("forum_posts").insert({
            "doctor_id": data.doctor_id,
            "doctor_name": data.doctor_name,
            "title": data.title,
            "content": data.content,
            "category": data.category
        }).execute()
        return {"status": "success", "data": res.data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/forum/posts/list")
def list_forum_posts():
    try:
        res = supabase.table("forum_posts").select("*").order("created_at", desc=True).execute()
        return {"status": "success", "data": res.data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/forum/comments/add")
def add_forum_comment(data: ForumCommentSchema):
    try:
        res = supabase.table("forum_comments").insert({
            "post_id": data.post_id,
            "doctor_id": data.doctor_id,
            "doctor_name": data.doctor_name,
            "comment": data.comment
        }).execute()
        return {"status": "success", "data": res.data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
