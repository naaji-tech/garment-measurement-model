import os
import shutil
import tempfile
import traceback
from fastapi import FastAPI, File, HTTPException, UploadFile

from app.service.call_measure_proc import call_measure_proc

app = FastAPI()


@app.post("/v1/productMeasurements")
async def measure_garment(image: UploadFile = File(...)):
    """
    description:
        - API URL endpoint to measure garment measurements from an image of garment with reference object
    parameters:
        - File : image with garment and reference object
    return:
        - garment measurements with JSON string
    """
    try:
        # Validating parameter request parameter
        print(f"image : {image.filename}, size : {image.size}")
        if image.filename == "" and image.size == 0:
            raise HTTPException(status_code=400, detail="No image file provided")
        
        # Save image temporarily
        print("Saving image to temp location...")
        temp_image_path = tempfile.mktemp(suffix=".jpg")
        with open(temp_image_path, "wb") as buffer:
            shutil.copyfileobj(image.file, buffer)
        print("Image saved successfully.")

        # Calling the measurement procedure and return the dict
        print("calling garment measurement procedures...")
        res = call_measure_proc(temp_image_path)
        print(f"response : {res}")

        # Remove the temporary image
        os.remove(temp_image_path)

        return res

    except Exception as e:
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))
