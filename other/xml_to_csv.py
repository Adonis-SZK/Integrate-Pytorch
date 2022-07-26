#将目标检测的标签由xml格式转换为csv格式,同时将左上右下坐标转为中心点和宽高
import os
import numpy as np
import pandas as pd
import xml.etree.cElementTree as ET
path_read=r'C:\Users\twk10\Desktop\data\lamp\label' #待更改项
path_save=r'C:\Users\twk10\Desktop\data\lamp\label_csv' #待更改项
list_dir=os.listdir(path_read)
list_column=['name','class','Cx','Cy','w','h']
for i in range(len(list_dir)):
    list_value = []
    path_file=path_read+'/'+list_dir[i]
    root=ET.parse(path_file).getroot()
    name=root.find('filename').text
    for object in root.findall('object'):
        value=[name,
               object[0].text,
               int(object[4][0].text),
               int(object[4][1].text),
               int(object[4][2].text),
               int(object[4][3].text)
               ]
        list_value.append(value)
    df=pd.DataFrame(list_value,columns=list_column)
    w=df['w'].values-df['Cx'].values
    h=df['h'].values - df['Cy'].values
    Cx=df['Cx'].values+1/2*w
    Cy=df['Cy'].values+1/2*h
    df = df.drop(['Cx', 'Cy', 'w', 'h'], axis=1)
    df['Cx']=np.around(Cx).astype(np.int32)
    df['Cy']=np.around(Cy).astype(np.int32)
    df['w']=np.around(w).astype(np.int32)
    df['h'] =np.around(h).astype(np.int32)
    df.to_csv(path_save+'/'+list_dir[i].split('.')[0]+'.csv')
print('1')


