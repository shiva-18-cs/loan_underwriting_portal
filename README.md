# Credence — Loan Decision Engine

A complete, ready-to-run package: trained model + FastAPI backend + the new UI.

## 1. What's in this folder

```
loan_app/
├── main.py                    ← FastAPI backend (yours, unchanged)
├── templates/
│   └── index.html              ← NEW professional UI (replaces your old index.html)
├── Loan_model.joblib           ← your trained KNN model
├── Loan_features.joblib        ← the 5 feature names the model expects
├── standard_scaler.joblib      ← StandardScaler (income, loan_percent_income)
├── minmax_scaler.joblib        ← MinMaxScaler (loan_int_rate)
└── requirements.txt            ← Python dependencies
```

This mirrors the structure your `main.py` already expects — `Jinja2Templates(directory="templates")` looks for `templates/index.html`, so the new UI drops straight in.

## 2. Run it locally

```bash
cd loan_app
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt

uvicorn main:app --reload --port 8000
```

Open **http://localhost:8000** — you should see the new UI.

## 3. How the pieces connect

```
Browser (templates/index.html)
   │  user fills form → clicks "Run decision"
   │  fetch POST /predict   (JSON body)
   ▼
FastAPI (main.py)
   │  validates with Pydantic (LoanFeatures)
   │  maps category strings → integers
   │  scales numeric fields with the saved scalers
   │  reorders columns to match Loan_features.joblib
   │  model.predict(...)
   │  fires a webhook (optional lead capture) to n8n at
   │  http://localhost:5678/webhook-test/Loan_lead
   ▼
JSON { "prediction": "Approved" | "Rejected" }
   │
   ▼
Browser renders result card + decision panel
```

The form fields, field names, and JSON payload in the new UI match `LoanFeatures` in `main.py` **exactly** — no backend changes were needed:

| UI field | JSON key | Type |
|---|---|---|
| Full name | `name` | string |
| Age | `age` | int |
| Email | `email` | string |
| Phone | `phone` | string |
| Annual income | `person_income` | float |
| Home ownership | `person_home_ownership` | `"RENT"` \| `"OWN"` \| `"MORTGAGE"` \| `"OTHER"` |
| Interest rate | `loan_int_rate` | float |
| Loan / income | `loan_percent_income` | float (0–1) |
| Previous default | `previous_loan_defaults_on_file` | `"YES"` \| `"NO"` |

## 4. What's new vs. your old `index.html`

- Same form fields, same `/predict` contract — **zero backend changes required**.
- Added inline validation (red border + message) before it ever hits the API.
- Added a live "decision panel" on the right: a risk gauge and signal checklist that update as the user types. **This is a frontend-only heuristic preview** for engagement — the real decision still comes only from `model.predict()` on submit. This is clearly labeled in the UI so it's never mistaken for the model's output.
- Loading state on the submit button, graceful error toast if `/predict` fails (e.g. webhook timeout, bad input).
- "Approved" / "Declined" result rendered as a styled card instead of plain text in an `<h3>`.

## 5. What else you'll need before this is production-ready

**Must-have:**
1. **n8n webhook** — `main.py` posts the lead to `http://localhost:5678/webhook-test/Loan_lead`. That's a local n8n test URL; it will silently fail (caught and logged) if n8n isn't running. Either:
   - run n8n locally (`npx n8n`) and import/build that workflow, or
   - point the URL at your production webhook, or
   - remove the webhook call if you don't need lead capture.
2. **CORS** — if you ever serve the frontend from a different origin/port than the API (e.g. deploying the UI on Vercel and the API on Render), add `fastapi.middleware.cors.CORSMiddleware` to `main.py`. Not needed if Jinja2 serves the page from the same FastAPI app (current setup).
3. **Environment-based config** — the webhook URL and any secrets should move to environment variables (`.env` + `python-dotenv` or `pydantic-settings`) rather than being hardcoded, especially before deploying.

**Recommended before real users touch it:**
4. **Server-side validation hardening** — Pydantic already validates types; consider adding range constraints (e.g. `Field(ge=0)` on income/rate) so malformed requests fail before reaching the model.
5. **HTTPS** — deploy behind a reverse proxy (Nginx/Caddy) or a platform that terminates TLS (Render, Railway, Fly.io) — loan applications carry PII (email, phone) and shouldn't go over plain HTTP.
6. **Persistent storage for applications** — right now a prediction is fired-and-forgotten to a webhook. If you want an audit trail, add a database (Postgres/SQLite) to log every application + prediction + timestamp.
7. **Rate limiting / abuse protection** — e.g. `slowapi`, since this is a public-facing form.
8. **Model versioning** — keep the `.joblib` files in version control or a model registry (e.g. MLflow) so you can track which model produced which decision, especially if you retrain `Train_ML_model.py` later.

**Deployment options (pick one):**
- **Render / Railway / Fly.io** — easiest for a FastAPI + static-template app like this; push the whole `loan_app/` folder, set the start command to `uvicorn main:app --host 0.0.0.0 --port $PORT`.
- **Docker** — containerize for portability; a minimal `Dockerfile` would `COPY` this folder, `pip install -r requirements.txt`, and `CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]`. Ask if you'd like one written out.

## 6. Quick sanity checklist before going live

- [ ] `uvicorn main:app --reload` starts with no errors
- [ ] Submitting the form with valid data returns "Approved" or "Rejected"
- [ ] Submitting with an invalid field (e.g. letters in age) shows the inline red error, not a raw server error
- [ ] n8n webhook either works or is intentionally disabled
- [ ] `.joblib` files load without a scikit-learn version mismatch warning (if it appears, re-pickle with the same sklearn version used at inference)
