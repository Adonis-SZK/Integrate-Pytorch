#根据指定条件筛选图片和标签
import os
import cv2
import pandas as pd
#设置
# -------------------------------------------------------------------------------------------------------------------- #
path=r'.\all.csv'
path_img=r'.\img'
path_label=r'.\label'
img_save=r'.\img_save'
label_save=r'.\label_save'
# -------------------------------------------------------------------------------------------------------------------- #
#程序
df = pd.read_csv(path)
dir_img = os.listdir(path_img)
dir_label = os.listdir(path_label)
number=df['number'].values
count=0
for i in range(len(number)):
    if number[i]<50:
        cv2.imwrite(img_save+'/'+dir_img[i],cv2.imread(path_img+'/'+dir_img[i]))
        pd.read_csv(path_label+'/'+dir_label[i]).to_csv(label_save+'/'+dir_label[i])
        count+=1
print('总数:',count)