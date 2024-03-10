import requests
url = 'http://localhost:5000/predictions/1'

import cv2
path_to_image = "data/example/1.jpg"

files = {'file': open(path_to_image, 'rb') }
resp = requests.post(url, files=files, auth=("user", "password"))
print(resp.content)