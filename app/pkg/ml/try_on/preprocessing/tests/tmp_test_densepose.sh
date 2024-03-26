cd detectron2/projects/DensePose/
# python3 apply_net.py show configs/densepose_rcnn_R_50_FPN_s1x.yaml \
#     /usr/src/app/app/pkg/ml/weights/dense_pose.pkl \
#     /usr/src/app/volume/data/resized/ dp_segm -v --output /usr/src/app/volume/data/dense_pose/

python3 apply_net.py dump configs/densepose_rcnn_R_50_FPN_s1x.yaml \
    /usr/src/app/app/pkg/ml/weights/dense_pose.pkl \
    /usr/src/app/volume/data/resized/ -v --output /usr/src/app/volume/data/dense_pose/
