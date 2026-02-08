from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import logging
import os
import google.generativeai as genai
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

class RecipeRequest(BaseModel):
    ingredients: str

class RecipeResponse(BaseModel):
    title: str
    instructions: str

# --- Configuración Gemini ---
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)

# USAMOS EL MODELO QUE CONFIRMAMOS EN LA LISTA
MODEL_NAME = 'models/gemini-flash-latest'
model = genai.GenerativeModel(
    model_name=MODEL_NAME,
    generation_config={"response_mime_type": "application/json"}
)

@app.get("/health")
async def health():
    return {"status": "ok", "model": MODEL_NAME, "info": "Ready for action"}

@app.post("/generate-recipe", response_model=RecipeResponse)
async def generate_recipe(request: RecipeRequest):
    if not GEMINI_API_KEY:
        raise HTTPException(status_code=500, detail="API Key missing")

    prompt = f"""Genera una receta de cocina en JSON con:
    {{
        "title": "nombre",
        "instructions": "pasos"
    }}
    Ingredientes: {request.ingredients}
    """

    try:
        response = await model.generate_content_async(prompt)
        recipe_data = json.loads(response.text)
        return RecipeResponse(
            title=recipe_data.get("title", "Receta IA"),
            instructions=recipe_data.get("instructions", "Pasos no disponibles")
        )
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))