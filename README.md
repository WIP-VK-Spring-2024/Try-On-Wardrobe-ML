Need python Python 3.10.12

Наследоваться от docker образа torch serve: 
'docker pull pytorch/torchserve'

Run serve:
</br>./serve/serve_utils/serve.sh 

Stop serve:
</br>torchserve --stop

Example of serve request:
</br> curl -X POST http://localhost:8080/predictions/keypoints -T data/example/1.jpg