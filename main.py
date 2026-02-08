from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import logging
import os
import google.generativeai as genai
import json # To parse the LLM's JSON response

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
# Load API key from environment variable
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    logger.error("GEMINI_API_KEY environment variable not set.")
    # In a real app, you might want to raise an exception or handle this more gracefully
    # For now, we'll let the app start but LLM calls will fail.

genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel('gemini-1.5-flash') # Usando gemini-1.5-flash para mayores límites y velocidad

# --- End LLM Integration ---

@app.post("/", response_model=RecipeResponse)
def generate_recipe_at_root(request: RecipeRequest):
    logger.info(f"Received ingredients at root: {request.ingredients}")
    return RecipeResponse(
        title=f"Receta con {request.ingredients}",
        instructions="1. Mezclar todo.\n2. Cocinar.\n3. Servir."
    )

@app.get("/health")
async def health():
    return {"status": "ok", "model": "gemini-1.5-flash"}

@app.post("/generate-recipe", response_model=RecipeResponse)
async def generate_recipe(request: RecipeRequest):
    if not GEMINI_API_KEY:
        raise HTTPException(status_code=500, detail="GEMINI_API_KEY no configurada en el servidor.")

    logger.info(f"Received ingredients at /generate-recipe: {request.ingredients}")

    prompt = f"""Genera una receta de cocina en formato JSON. La receta debe incluir un título y una lista de instrucciones.
    Los ingredientes disponibles son: {request.ingredients}.
    El formato JSON debe ser:
    {{
        "title": "Título de la receta",
        "instructions": "1. Paso uno.\n2. Paso dos."
    }}
    Asegúrate de que la respuesta sea solo el JSON válido, sin texto adicional.
    """

    try:
        response = await model.generate_content_async(prompt)
        # Assuming the LLM response is directly a JSON string
        # Need to handle cases where LLM might return extra text or invalid JSON
        llm_response_text = response.text
        logger.info(f"LLM Raw Response: {llm_response_text}")

        # Attempt to parse the JSON response
        try:
            # Find the start and end of the JSON object
            json_start = llm_response_text.index('{')
            json_end = llm_response_text.rindex('}') + 1
            clean_json_str = llm_response_text[json_start:json_end]
            recipe_data = json.loads(clean_json_str)
        except ValueError:
            logger.error(f"Could not find valid JSON in LLM response: {llm_response_text}")
            raise HTTPException(status_code=500, detail="Error al procesar la respuesta del modelo de IA (JSON no encontrado).")
        
        # Validate the structure
        if "title" not in recipe_data or "instructions" not in recipe_data:
            raise ValueError("La respuesta del LLM no contiene 'title' o 'instructions'.")

        return RecipeResponse(
            title=recipe_data["title"],
            instructions=recipe_data["instructions"]
        )
    except json.JSONDecodeError as e:
        logger.error(f"Error al decodificar JSON de la respuesta del LLM: {e}")
        logger.error(f"Respuesta del LLM: {llm_response_text}")
        raise HTTPException(status_code=500, detail="Error al procesar la respuesta del modelo de IA (JSON inválido).")
    except Exception as e:
        logger.error(f"Error al generar receta con LLM: {e}")
        raise HTTPException(status_code=500, detail=f"Error al generar receta con el modelo de IA: {e}")
