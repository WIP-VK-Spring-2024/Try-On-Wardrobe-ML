# to launch: 
# uvicorn fastapi_app:app --port=8000
# or
# python fastapi_app.py 

import os

import uvicorn
from fastapi import File, UploadFile
from fastapi import FastAPI
import logging

DATABASE_PATH = "app/data/database"
PATH_TO_CLOTHES = "{}/user_{}/clothes/image_{}/original"
PATH_TO_FULL_BODY = "{}/user_{}/full_body/image_{}/original"

logging.basicConfig(level=logging.INFO, filename="logs/fastapi_logs.log",filemode="w")

app = FastAPI()

def check_path(path):
    if not os.path.exists(path):
        os.makedirs(path)


@app.post("/data/load")
def upload(image_type:str, user_id: int, image_id: int, file: UploadFile = File(...)):
    logging.info("Got POST image request")
    try:
        contents = file.file.read()
        extension = file.filename.split(".")[-1]
        save_filename = f'{image_id}.{extension}'
        
        if image_type == "full-body":
            path_to_save = PATH_TO_FULL_BODY.format(DATABASE_PATH, user_id, image_id)
        elif image_type == "cloth":
            path_to_save = PATH_TO_CLOTHES.format(DATABASE_PATH, user_id, image_id)
        else:
            raise Exception("Incorrect input image_type. Should be one of: 'full-body', 'cloth'")

        check_path(path_to_save)
        file_path = os.path.join(path_to_save, save_filename)

        with open(file_path, 'wb') as f:
            f.write(contents)
    except Exception as e:
        return {"message": f"There was an error uploading the full-body file: {e}"}
    finally:
        file.file.close()

    return {"message": f"Successfully uploaded full-body"}

# if __name__ == "__main__":
#     uvicorn.run(app, port=1234)

# @app.post("/data/cloth")
# def upload(user_id: int, image_id: int, file: UploadFile = File(...)):
#     logging.info("Got cloth POST request")
#     try:
#         contents = file.file.read()
#         extension = file.filename.split(".")[-1]
#         save_filename = f'{image_id}.{extension}'
        
#         path_to_save = PATH_TO_CLOTHES.format(DATABASE_PATH, user_id, image_id)
#         check_path(path_to_save)
#         file_path = os.path.join(path_to_save, save_filename)
#         with open(file_path, 'wb') as f:
#             f.write(contents)
#     except Exception as e:
#         return {"message": f"There was an error uploading the cloth file: {e}"}
#     finally:
#         file.file.close()

#     return {"message": f"Successfully uploaded cloth-body"}
