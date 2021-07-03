# coding: utf-8
import requests 
import base64
import json
# with open("test_img.jpg", "rb") as image_file:
    # encoded_string = base64.b64encode(image_file.read())
url = "http://ppt.np.is/img/2566921-back-1000-0.jpg"
data = {"image": url, "model_type": "back"} #model_type in {"front", "back"}  default "front"
# data = {"image": encoded_string}
r = requests.post(url = 'http://127.0.0.1:5022/image_rotation', data=data)
r = json.loads(r.content)
print(json.dumps(r, indent=4))
