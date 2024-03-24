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
wget /link/to/weights/archieve -O weights.zip \
&& source .env \
&& unzip weights.zip -d $ML__WEIGHTS_PATH\
&& rm weights.zip
```

## ML usage
Классы для обработки имеют следующий вид:
1) Предобработка фоток одежды: app/pkg/ml/try_on/preprocessing/aggregator.py - класс ClothProcessor
2) Предобработка фоток людей: app/pkg/ml/try_on/preprocessing/aggregator.py - класс HumanProcessor
3) Автопримерка (пока что мок) - app/pkg/ml/try_on/lady_vton.py  класс - LadyVtonAggregator

## Launch dev container
 To launch container for developing ml models:  
```shell
docker build -t dev_virt_ward:latest -f docker/model/Dockerfile_dev .
docker run -id --rm --gpus all -v .:/usr/src/app/ -p 8843:22 --name virt_ward_dev dev_virt_ward:latest
```

