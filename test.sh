torchserve --stop
. ./serve/serve_utils/model_convert.sh
rm -r logs/
. ./serve/serve_utils/serve.sh