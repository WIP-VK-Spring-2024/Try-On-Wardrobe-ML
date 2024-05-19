# Model api

## Fast setup
Предварительно нужно создать файл ```.env``` в корне проекта по шаблону .env.example

Далее надо освободить порт 5432, используемый обычно Postgres, после чего запускаем контейнер:

### FastAPI and RabbitMQ
```shell
docker network create shared-api-network
docker-compose up -d --build
```
### All model runners for queing tasks
```shell
docker compose -f docker-compose-model.yml up -d --build --force-recreate
```

## System requirements
Глобально в проекте используются четыре разные составные модели:
#### 1. Модель примерки Try On.
- Требует 19 GB GPU
#### 2. Модель обрезки Cut.
Возможны конфигурации
- Требует 1 GB GPU. Облегченная версия модели, не рекомендуется использовать в полноценном продукте
    > Задается установкой в ```.env``` ```ML__IS_CUT_LIGHT_WEIGHT=True```
- Требует 8 GB GPU. Рекомендуется использовать в полноценном продукте.
    > Задается установкой в ```.env``` ```ML__IS_CUT_LIGHT_WEIGHT=False```
#### 3. Модель генерации комплектов одежды Outfit gen.
- Требует 1 GB GPU
#### 4. Модель рекомендательной системы комплектов Rec Sys.
- Не требует GPU, считается на CPU

### Примеры конфигураций
**Максимальный режим**: 2 x 24 Гб GPU. Например, 2 x Tesla A5000

**Облегченный режим**: 1 x 24 Гб GPU. Например, 1 x Tesla A5000. Допускается использование облегченных моделей обрезки 


## ML usage
> За неимением времени на разработку в проекте используется requirements.txt для запуска ML части. В планах его перенести в pyproject.toml.

Чтобы ml заработал нужно скачать веса. Для этого скачиваем архив с весами (ссылку можно попросить у @KOTOBOPOT). Его содержимое распаковываем в папку ML__WEIGHTS_PATH (переменная в .env). Указанные операции можно сделать при помощи следующих команд:

```shell
wget /link/to/weights/archieve -O weights.zip \
&& source .env \
&& mkdir $ML__WEIGHTS_PATH \
&& unzip weights.zip -d $ML__WEIGHTS_PATH\
&& rm weights.zip
```

Classes for working has folowing view:
1) __Clothes Preprocessing__: app/pkg/ml/try_on/preprocessing/aggregator.py - class: ClothProcessor. method: consistent_forward(image_bytes)
<br>Clothes processing launches when user has uploded cloth image

2) __Human Preprocessing__: app/pkg/ml/try_on/preprocessing/aggregator.py - class: HumanProcessor. method: consistent_forward(image_bytes)
<br>Human processing launches when user has uploded human image

3) __Try on__: app/pkg/ml/try_on/lady_vton.py  class: TryOnAggregator. method: \_\_call\_\_(dict)
<br>Try on processing launches when user has requested to try on. Input dict format is:
```
{
"image_human_orig":io.BytesIO,  # - image with human
"parsed_human":io.BytesIO,  # - image with parsed human. 
"keypoints_json":io.BytesIO # human keypoints json
"cloth":io.BytesIO # cloth (without background) image bytes
"category":str, # one of ['dresses', 'upper_body','lower_body']
}
```
Keys from dict above corresponds with output keys from 1) and 2) methods.

4) __RecSys (global)__: /usr/src/app/app/pkg/ml/outfits_recsys/recsys.py, class CrossUsersOutfitRecSys. Main methods:
    <br> 4.1. update_global_outfits_list_format(outfits, cloth_id_to_list)_ - updates system with new outfits. Must be called every 10% of data gain (for example there were 100 outfits in common, then became 110, so need to call that method). Also can be called at system intialization
    <br> 4.2. _recommend(uuid, samples)_ - recommends outfits directly from input user's uuid. Returns only ids. For example [93, 21, 453].

Formats for recsys:
<br>outfits format: [ {"user_id":int/str, "outfit_id":int/str, "clothes": [cloth_id, ...]  }, ...  ]
<br> cloth_id_to_list format - dict[cloth_id] -> list (cloth tensor)
<br> Key "tensor" can be caught from autoset system.

___
P.S. It's recommended to init instances once in each class above. Initialization process can require more time, than image processing. 


## Launch dev container
### Dockerfile  
```shell
docker build -t dev_virt_ward:latest -f docker/model/Dockerfile_dev .
docker run -id --rm --gpus all -v .:/usr/src/app/ --name virt_ward_dev dev_virt_ward:latest
```
### Docker compose
```
docker compose -f docker-compose-dev.yml up -d --build --force-recreate
```

