from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import joblib
import numpy as np

# Load the trained XGBoost model
xgboost_model = joblib.load("xgboost_model.pkl")

# Create FastAPI instance
app = FastAPI()
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def read_form(request: Request):
    # Render the form template for user input
    return templates.TemplateResponse("form.html", {"request": request})

@app.post("/predict", response_class=HTMLResponse)
async def predict(
    request: Request,
    age: int = Form(...),
    gender: int = Form(...),  # 0 for female, 1 for male
    bmi: float = Form(...),
    children: int = Form(...),
    smoker: int = Form(...),  # 0 for no, 1 for yes
    region: int = Form(...),  # 0 = northeast, 1 = northwest, 2 = southeast, 3 = southwest
    medical_history: int = Form(...),
    family_medical_history: int = Form(...),
    exercise_frequency: int = Form(...),
    occupation: int = Form(...),
    coverage_level: int = Form(...)
):
    # Prepare the input array for prediction
    input_array = np.array([[age, gender, bmi, children, smoker, region, medical_history, 
                             family_medical_history, exercise_frequency, occupation, coverage_level]])
    
    # Predict using the model
    try:
        prediction = xgboost_model.predict(input_array)
        formatted_prediction = f"${prediction[0]:,.2f}"  # Format prediction with commas and two decimal places
    except Exception as e:
        formatted_prediction = f"Error in prediction: {str(e)}"

    # Render the result template with the prediction
    return templates.TemplateResponse("result.html", {
        "request": request, 
        "prediction": formatted_prediction
    })
