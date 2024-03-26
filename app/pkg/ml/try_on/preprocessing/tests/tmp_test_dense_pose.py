def check_type(el):
    assert el is int or el is float
def check_arrays(a1, a2):

    for i in range(len(a1)):
        # check_type(a1[i])
        # check_type(a2[i])
        if abs(a1[i] - a2[i]) > 0.00001:
            print(f'{i=} {a1[i]=} {a2[i]=}')
            return False
    return True

import torch
fp1 = "/usr/src/app/volume/data/dense_pose/colab.npz"
fp2 = "/usr/src/app/volume/data/dense_pose/dense_pose_human.npz"

f1 = torch.load(fp1)
f2 = torch.load(fp2)

assert len(f1) == len(f2)
assert type(f1[0]) == type(f2[0])

# assert check_arrays(list(f1[0].keys()), list(f2[0].keys())) 

print(f1[0].keys())
assert f1[0]['scores'] == f2[0]['scores']
assert all((f1[0]['pred_boxes_XYXY'] == f2[0]['pred_boxes_XYXY']).flatten())

assert len(f1[0]['pred_densepose']) == 1
assert len(f2[0]['pred_densepose']) == 1

assert all((f1[0]['pred_densepose'][0].labels == f2[0]['pred_densepose'][0].labels).flatten())

# print(f1[0]['pred_densepose'][0].uv)
# print(f2[0]['pred_densepose'][0].uv)

t1 = f1[0]['pred_densepose'][0].uv
t2 = f2[0]['pred_densepose'][0].uv

#print(type(t1.flatten().tolist()[0]))
assert check_arrays(t1.flatten().tolist(), t2.flatten().tolist())

#print(f2[0]['pred_densepose'][0].uv)

# assert all((f1[0]['pred_densepose'][0].uv == f2[0]['pred_densepose'][0].uv).flatten())

#print( all((f1[0]['pred_densepose'] == f2[0]['pred_densepose']).flatten()))
#print(f1[0]['pred_densepose'])
