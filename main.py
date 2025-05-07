from fastapi import FastAPI, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import util

app = FastAPI()

# Optional: CORS middleware (handles Access-Control-Allow-Origin)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/classify_image")
async def classify_image(image_data: str = Form(...)):
    # print(f"Received image data: {image_data}")  # Log the received data
    result = util.classify_image(image_data)
    print(f"Classification result: {result}")
    return JSONResponse(content=result)

@app.on_event("startup")
def load_model():
    print("Starting FastAPI Server For Sports Celebrity Image Classification")
    util.load_saved_artifacts()
