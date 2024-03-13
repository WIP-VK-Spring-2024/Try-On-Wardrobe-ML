import requests

# example of loading image with human
def human_image_example():
    url = 'http://localhost:7329/data/load'

    path_to_image = "../data/example/human.jpg"

    payload = {"user_id":0, "image_id":1, "image_type":"full-body" }
    files = {"file":open(path_to_image, 'rb'), }
    resp = requests.post(url, params=payload,files=files )#auth=("user", "password"))
    print(resp.json())


def cloth_image_example():
    url = 'http://localhost:7329/data/load'

    path_to_image = "../data/example/t_shirt.png"

    payload = {"user_id":0, "image_id":1, "image_type":"cloth" }
    files = {"file":open(path_to_image, 'rb'), }
    resp = requests.post(url, params=payload,files=files )#auth=("user", "password"))
    print(resp.json())

if __name__ == '__main__':
    human_image_example()
   # cloth_image_example()