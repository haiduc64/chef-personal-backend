from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import logging
import os
import google.generativeai as genai
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

class RecipeRequest(BaseModel):
    ingredients: str

class RecipeResponse(BaseModel):
    title: str
    instructions: str

# --- LLM Integration ---
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    logger.error("GEMINI_API_KEY environment variable not set.")

genai.configure(api_key=GEMINI_API_KEY)

# NATIVELY FORCE JSON OUTPUT
model = genai.GenerativeModel(
    model_name='models/gemini-1.5-flash',
    generation_config={"response_mime_type": "application/json"}
)

# --- End LLM Integration ---

@app.get("/health")
async def health():
    return {"status": "ok", "model": "gemini-1.5-flash-json-mode"}

@app.post("/", response_model=RecipeResponse)
def generate_recipe_at_root(request: RecipeRequest):
    return RecipeResponse(
        title=f"Receta con {request.ingredients}",
        instructions="1. Mezclar todo.\n2. Cocinar.\n3. Servir."
    )

@app.post("/generate-recipe", response_model=RecipeResponse)
async def generate_recipe(request: RecipeRequest):
    if not GEMINI_API_KEY:
        raise HTTPException(status_code=500, detail="GEMINI_API_KEY no configurada.")

    logger.info(f"Received ingredients: {request.ingredients}")

    prompt = f"""Genera una receta de cocina en JSON con este esquema exacto:
    {{
        "title": "nombre de la receta",
        "instructions": "pasos detallados de la receta"
    }}
    Ingredientes disponibles: {request.ingredients}
    """

    try:
        response = await model.generate_content_async(prompt)
        recipe_data = json.loads(response.text)
        
        return RecipeResponse(
            title=recipe_data.get("title", "Receta Personalizada"),
            instructions=recipe_data.get("instructions", "Instrucciones no generadas.")
        )
    except Exception as e:
        logger.error(f"Error generating recipe: {e}")
        raise HTTPException(status_code=500, detail=f"Error al generar receta: {str(e)}")