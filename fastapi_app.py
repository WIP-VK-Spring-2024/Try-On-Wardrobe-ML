# to launch: 
# uvicorn fastapi_app:app

import os

from fastapi import File, UploadFile
from fastapi import FastAPI
import logging


logging.basicConfig(level=logging.INFO, filename="logs/fastapi_logs.log",filemode="w")

app = FastAPI()


@app.post("/data/human")
def upload(user_id: int, image_id: int, file: UploadFile = File(...)):
    logging.info("Got upload human full growth request")
    try:
        contents = file.file.read()
        logging.info("human object")
        path_to_save = os.path.join("data", "from_fast_api", file.filename)
        with open(path_to_save, 'wb') as f:
            f.write(contents)
    except Exception:
        return {"message": "There was an error uploading the file"}
    finally:
        file.file.close()

    return {"message": f"Successfully uploaded {file.filename}"}

