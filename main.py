import joblib
import pandas as pd
import requests
import io

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from fastapi import Request, Form
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.templating import Jinja2Templates

from gtts import gTTS
from explainability import get_shap_chart, get_counterfactuals

templates = Jinja2Templates(directory="templates")


app = FastAPI()

model = joblib.load("Loan_model.joblib")
features = joblib.load("Loan_features.joblib")
ss = joblib.load("standard_scaler.joblib")
mms = joblib.load("minmax_scaler.joblib")


import sqlite3

DB_PATH = "users.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            mobile_number TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            birthdate TEXT NOT NULL,
            email TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()

init_db()

class LoginCheckRequest(BaseModel):
    mobile_number: str

class RegisterRequest(BaseModel):
    mobile_number: str
    name: str
    birthdate: str
    email: str

class LoanFeatures(BaseModel):
    name: str
    age: int
    email: str
    phone: str
    person_income: float
    person_home_ownership: str
    loan_int_rate: float
    loan_percent_income: float
    previous_loan_defaults_on_file: str

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html"
    )

@app.post("/predict")
def predict(data: LoanFeatures):

    try:
        home_ownership_mapping = {
            "RENT": 0,
            "OWN": 1,
            "MORTGAGE": 2,
            "OTHER": 3
        }

        previous_loan_defaults_mapping = {
            "NO": 0,
            "YES": 1
        }

        # Original scaling and formatting for prediction
        input_df = pd.DataFrame([{
            "person_income": data.person_income,
            "person_home_ownership": home_ownership_mapping[data.person_home_ownership],
            "loan_int_rate": data.loan_int_rate,
            "loan_percent_income": data.loan_percent_income,
            "previous_loan_defaults_on_file": previous_loan_defaults_mapping[data.previous_loan_defaults_on_file]
        }])

        input_df[['person_income', 'loan_percent_income']] = ss.transform(
            input_df[['person_income', 'loan_percent_income']]
        )

        input_df[['loan_int_rate']] = mms.transform(
            input_df[['loan_int_rate']]
        )

        input_df = input_df[features]

        prediction = model.predict(input_df)[0]
        result = "Approved" if prediction == 1 else "Rejected"

        # Generate explainability features
        raw_df = pd.DataFrame([{
            "person_income": data.person_income,
            "person_home_ownership": data.person_home_ownership,
            "loan_int_rate": data.loan_int_rate,
            "loan_percent_income": data.loan_percent_income,
            "previous_loan_defaults_on_file": data.previous_loan_defaults_on_file
        }])

        shap_chart = get_shap_chart(raw_df)
        is_approved = (prediction == 1)
        counterfactuals = get_counterfactuals(raw_df, is_approved)

        # Calculate EMI and debt ratio (assume 60 months / 5 years term)
        loan_amount = data.person_income * data.loan_percent_income
        r_monthly = data.loan_int_rate / 12 / 100
        term_months = 60
        
        if r_monthly > 0:
            emi = loan_amount * r_monthly * ((1 + r_monthly) ** term_months) / (((1 + r_monthly) ** term_months) - 1)
        else:
            emi = loan_amount / term_months if term_months > 0 else 0
            
        monthly_income = data.person_income / 12 if data.person_income > 0 else 1
        debt_ratio = (emi / monthly_income) * 100

        payload = {
            "name": data.name,
            "age": data.age,
            "income": data.person_income,
            "home_ownership": data.person_home_ownership,
            "interest_rate": data.loan_int_rate,
            "loan_percent_income": data.loan_percent_income,
            "previous_default": data.previous_loan_defaults_on_file,
            "prediction": result,
            "email": data.email,
            "phone": data.phone
        }

        try:
            requests.post(
                "http://localhost:5678/webhook/Loan_Application",
                json=payload,
                timeout=5
            )
        except Exception as webhook_error:
            print("Webhook Error:", webhook_error)

        return {
            "prediction": result,
            "shap_chart": shap_chart,
            "counterfactuals": counterfactuals,
            "emi": round(emi, 2),
            "debt_ratio": round(debt_ratio, 2),
            "loan_amount": round(loan_amount, 2)
        }

    except Exception as e:
        print("ERROR OCCURRED:", repr(e))
        raise HTTPException(
            status_code=500,
            detail=f"prediction failed: {str(e)}"
        )

@app.post("/api/login-check")
def login_check(data: LoginCheckRequest):
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT name, birthdate, email FROM users WHERE mobile_number = ?", (data.mobile_number.strip(),))
        row = cursor.fetchone()
        conn.close()
        
        if row:
            return {
                "exists": True,
                "user": {
                    "mobile_number": data.mobile_number.strip(),
                    "name": row[0],
                    "birthdate": row[1],
                    "email": row[2]
                }
            }
        else:
            return {
                "exists": False,
                "user": None
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database query failed: {str(e)}")

@app.post("/api/register")
def register(data: RegisterRequest):
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute(
            "INSERT OR REPLACE INTO users (mobile_number, name, birthdate, email) VALUES (?, ?, ?, ?)",
            (data.mobile_number.strip(), data.name.strip(), data.birthdate.strip(), data.email.strip())
        )
        conn.commit()
        conn.close()
        return {
            "success": True,
            "user": {
                "mobile_number": data.mobile_number.strip(),
                "name": data.name.strip(),
                "birthdate": data.birthdate.strip(),
                "email": data.email.strip()
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database insertion failed: {str(e)}")

@app.get("/tts")
def tts(text: str, lang: str = "en"):
    try:
        tts_obj = gTTS(text=text, lang=lang, slow=False)
        fp = io.BytesIO()
        tts_obj.write_to_fp(fp)
        fp.seek(0)
        return StreamingResponse(fp, media_type="audio/mpeg")
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"TTS generation failed: {str(e)}"
        )


    