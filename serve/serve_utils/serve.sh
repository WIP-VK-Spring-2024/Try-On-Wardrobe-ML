MODELS_PATH=serve/models
torchserve --start \
           --ncs \
           --model-store $MODELS_PATH \
           --models keypoints=$MODELS_PATH/keypoints.mar
#           --models keypoints=${MODELS_PATH}/keypoints.mar
           #--ts-config deployment/config.properties \
 #          ${MODELS_PATH}/