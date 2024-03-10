python_model_path=preprocess/openpose/src/body_all_in_one.py
pth_model_path=preprocess/openpose/model/jit/body_pose_model.pth
handler_path=preprocess/openpose/src/keypoints_handler.py

torch-model-archiver --model-name keypoints \
                     --version 1.0 \
                     --model-file $python_model_path \
                     --serialized-file $pth_model_path \
                     --handler $handler_path \
 #                     --extra-files \
#                      "preprocess/openpose/src/util.py,preprocess/openpose/src/model.py,preprocess/openpose/src/body.py"

                     #index_to_name.json \
mv keypoints.mar serve/models/keypoints.mar