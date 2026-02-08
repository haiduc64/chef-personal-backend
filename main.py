from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import logging
import os
import google.generativeai as genai
import json

# Configuración de logs
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

class RecipeRequest(BaseModel):
    ingredients: str

class RecipeResponse(BaseModel):
    title: str
    instructions: str

# --- Configuración de Gemini ---
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    logger.error("GEMINI_API_KEY no encontrada en el entorno.")

genai.configure(api_key=GEMINI_API_KEY)

# Usamos el nombre largo oficial que funciona en todas las versiones de la API
MODEL_NAME = 'models/gemini-1.5-flash'
model = genai.GenerativeModel(
    model_name=MODEL_NAME,
    generation_config={"response_mime_type": "application/json"}
)
# --- Fin de Configuración ---

@app.get("/health")
async def health():
    return {"status": "ok", "model": MODEL_NAME, "mode": "json"}

@app.post("/", response_model=RecipeResponse)
def root_check(request: RecipeRequest):
    return RecipeResponse(title="Server Active", instructions="Use /generate-recipe")

@app.post("/generate-recipe", response_model=RecipeResponse)
async def generate_recipe(request: RecipeRequest):
    if not GEMINI_API_KEY:
        raise HTTPException(status_code=500, detail="API Key no configurada.")

    prompt = f"""Genera una receta de cocina basada en estos ingredientes: {request.ingredients}.
    Responde estrictamente en formato JSON con este esquema:
    {{
        "title": "nombre de la receta",
        "instructions": "pasos numerados"
    }}
    """

    try:
        response = await model.generate_content_async(prompt)
        recipe_data = json.loads(response.text)
        
        return RecipeResponse(
            title=recipe_data.get("title", "Receta de Chef IA"),
            instructions=recipe_data.get("instructions", "No se pudieron generar los pasos.")
        )
    except Exception as e:
        logger.error(f"Error en la llamada a Gemini: {str(e)}")
        # Si falla el modo JSON, intentamos un mensaje de error claro
        raise HTTPException(status_code=500, detail=f"Error de la IA: {str(e)}")
