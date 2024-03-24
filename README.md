# Model api

## Fast setup
Предварительно нужно создать файл ```.env``` в корне проекта по шаблону .env.example

Далее надо освободить порт 5432, используемый обычно Postgres, после чего запускаем контейнер:

```shell
docker network create shared-api-network
docker-compose up -d --build
```

Чтобы ml заработал нужно скачать веса. Для этого скачиваем архив с весами (ссылку можно попросить у @KOTOBOPOT). Его содержимое распаковываем в папку ML__WEIGHTS_PATH (переменная в .env). Указанные операции можно сделать при помощи следующих команд:

```shell
wget /link/to/weights/archieve -O weights.zip
source .env
unzip weights.zip -d $ML__WEIGHTS_PATH
rm weights.zip
```


## Launch dev container
 To launch container for developing ml models:  
```shell
docker build -t ml_virt_ward:latest -f docker/model/Dockerfile_dev .
docker run -id -v .:/usr/src/app/ -p 8843:22 --name virt_ward_dev ml_virt_ward:latest
```

