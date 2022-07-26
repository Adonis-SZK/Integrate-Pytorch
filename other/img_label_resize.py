#将图片等比缩放为指定形状，并填充像素(0,0,0)成为正方形
import os
import cv2
import numpy as np
import pandas as pd

#设置
path_img=r'C:\Users\twk10\Desktop\data\lamp\img' #待更改项
path_label=r'C:\Users\twk10\Desktop\data\lamp\label_csv' #待更改项
save_img=r'C:\Users\twk10\Desktop\data\lamp\img_resize' #待更改项
save_label=r'C:\Users\twk10\Desktop\data\lamp\label_resize' #待更改项
size=640 #图片变成的形状

#程序
dir_img = os.listdir(path_img)
dir_label = os.listdir(path_label)
for i in range(len(dir_img)):
    img=cv2.imread(path_img+'/'+dir_img[i])
    label=pd.read_csv(path_label+'/'+dir_label[i])
    w0=len(img[0])
    h0=len(img)
    if w0==h0==size:
        cv2.imwrite(save_img + '/' + dir_img[i], img)
        label.to_csv(save_label + '/' + dir_label[i])
        continue
    if w0>=h0: #宽大于高
        w=size
        h=int(len(img)*size/len(img[0]))
        img=cv2.resize(img,(w,h))
        add_y=(w-h)//2
        img=cv2.copyMakeBorder(img,add_y,w-h-add_y,0,0,cv2.BORDER_CONSTANT,value=(0,0,0))
        Cx=label['Cx'].values*w/w0
        Cy = label['Cy'].values * h / h0+add_y
        w=label['w'].values * w / w0
        h=label['h'].values * h / h0
    else: #宽小于高
        w=int(len(img[0])*size/len(img))
        h=size
        img=cv2.resize(img,(w,h))
        add_x=(h-w)//2
        img=cv2.copyMakeBorder(img,0,0,add_x,h-w-add_x,cv2.BORDER_CONSTANT,value=(0,0,0))
        Cx=label['Cx'].values*w/w0+add_x
        Cy = label['Cy'].values * h / h0
        w=label['w'].values * w / w0
        h=label['h'].values * h / h0
    label=label.drop(['Cx','Cy','w','h'],axis=1)
    label['Cx']=np.around(Cx).astype(np.int32)
    label['Cy']=np.around(Cy).astype(np.int32)
    label['w']=np.around(w).astype(np.int32)
    label['h']=np.around(h).astype(np.int32)
    cv2.imwrite(save_img + '/' + dir_img[i],img.astype(np.float32))
    label.to_csv(save_label+'/'+dir_label[i])

# 检验
xmin = label['Cx'].values-1/2*label['w'].values
ymin = label['Cy'].values-1/2*label['h'].values
xmax = label['Cx'].values+1/2*label['w'].values
ymax = label['Cy'].values+1/2*label['h'].values
for i in range(len(label)):
    cv2.rectangle(img,(int(xmin[i]),int(ymin[i])),(int(xmax[i]),int(ymax[i])),color=(0,255,0),thickness=2)
cv2.imshow('123',img)
cv2.waitKey(0)
cv2.destroyAllWindows()
