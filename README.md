# Model api

## Fast setup
Предварительно нужно создать файл ```.env``` в корне проекта по шаблону .env.example

Далее надо освободить порт 5432, используемый обычно Postgres, после чего запускаем контейнер:

```shell
docker network create shared-api-network
docker-compose up -d --build
```

## Примеры запросов
Обработка одежды:
```json
{
  "user_id":0,
  "image_id":0,
  "image_type": "cloth", # must be one of "cloth",  "full-body"
  "tasks":{
    "remove_background":"HEX_BACKGROUND_COLOR",
    "tags":["tag1", "tag2",], # output is probability of each tag
  },

}
```

RecSys
```json
{
  "user_id":0,
  "prompt":"",
  "outfits_amount":1,
}
```



