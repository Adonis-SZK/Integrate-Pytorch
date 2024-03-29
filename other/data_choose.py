#根据指定条件筛选图片和标签
import os
import cv2
import pandas as pd
#设置
# -------------------------------------------------------------------------------------------------------------------- #
path=r'./all.csv'
path_img=r'./img'
path_label=r'./label'
save_img=r'./img_save'
save_label=r'./label_save'
if not os.path.exists(save_img):
    os.makedirs(save_img)
if not os.path.exists(save_label):
    os.makedirs(save_label)
# -------------------------------------------------------------------------------------------------------------------- #
#程序
df = pd.read_csv(path)
dir_img = sorted(os.listdir(path_img))
dir_label = sorted(os.listdir(path_label))
number=df['number'].values
count=0
for i in range(len(number)):
    if number[i]<50:
        cv2.imwrite(save_img+'/'+dir_img[i],cv2.imread(path_img+'/'+dir_img[i]))
        pd.read_csv(path_label+'/'+dir_label[i]).to_csv(save_label+'/'+dir_label[i])
        count+=1
print('总数:',count)
