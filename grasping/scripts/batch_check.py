import os
import subprocess
import argparse
import numpy as np


for root, dirs, files in os.walk('/apdcephfs/private_qihangfang/Codes/IBS-Grasping/Grasp_Dataset_v4/'):
    dirs = np.unique(np.array(dirs))
    break

train_split = []
test_split = []
for dir in dirs:
    dataset = dir.split('_')[0]
    if dataset =='ycb' or dataset=='bigbird':
        test_split.append(dir)
    else:
        train_split.append(dir)

print(len(train_split))
nums = []

new_train = []

aa = 0
for j, dir in enumerate(test_split):
    
    #if not os.path.exists(f'/apdcephfs/private_qihangfang/ibsshadow/{dir}'):
    #    num = 0
    #    #print(j // 6 * 6)
    #else:
    f = np.loadtxt(f'/apdcephfs/private_qihangfang/ibsdatasetgraspdtest/{dir}')
    #    if len(f.shape) == 1:
    #        num = 1
    #    else:
    #        num = f.shape[0]
    if f.shape[0] < 100:
        print(dir, f.shape[0])
    
        f = open(f'/apdcephfs/private_qihangfang/script/{aa * 5}.sh', 'w')
        print(f'/apdcephfs/private_qihangfang/script/{aa * 5}.sh')
        aa += 1

        f.write(f'/opt/conda/envs/py36/bin/python /apdcephfs/private_qihangfang/Codes/Form-closure/graspd/grasping/scripts2/collect_grasps.py collector_config={dir}\n')
        f.close()

    #if num != 10:
    #    print(j//6*6)
    #    print(dir, num)
    #nums.append(num)

    #a = f[:, -4:]
    #p = f[:, 8: -4]
    #print(p.shape)
    #exit()
    #print(p.shape)
    #succ_num = 0
    #for i in range(10):
    #    if a[i, 0] < 5e-4 and a[i, 1] < 0.03 and a[i, 2] < 0.02 and a[i, 3] == 6:
    #        succ_num += 1
    #if succ_num < 2:
    #    print(dir, succ_num)
    #    continue
    #else:
    #    new_train.append(dir)
    #print((a[:, 0] < 5e-4))
    #print((a[:, 1] < 0.03))
    #print((a[:, -5:] < 0.005).sum(axis=1) >= 2)
    #print((a[:, 0] < 5e-4) * (a[:, 1] < 0.03) * ((a[:, -5:] < 0.005).sum(axis=1) >= 2))
    #idx = (a[:, 0] < 5e-4) * (a[:, 1] < 0.03) * (a[:, 2] < 0.02) * (a[:, 3] == 6)
    #print(idx)
    #if (idx.sum()< 2):
    #    print(dir)
    #    exit()
    #p = p[idx]
    #np.savetxt(f'/apdcephfs/share_1330077/qihangfang/Data/ibs/{dir}/target_pose_shadow.txt', p)


    #nums.append(a)

    #print(i // 5 * 5, train_split[i], a.shape[0])

#nums = np.concatenate(nums, axis=0)

#np.savetxt('/apdcephfs/private_qihangfang/nums.txt', nums)

#print(np.mean(nums))


#print(new_train)