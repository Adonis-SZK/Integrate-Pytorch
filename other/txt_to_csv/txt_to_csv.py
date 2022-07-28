#将txt格式转换为csv格式
import os
import numpy as np
import pandas as pd
#设置
# -------------------------------------------------------------------------------------------------------------------- #
path_read=r'.\txt'
path_save=r'.\csv'
list_columns=['name','class','Cx','Cy','w','h','_','_']
# -------------------------------------------------------------------------------------------------------------------- #
list_dir=os.listdir(path_read)
for i in range(len(list_dir)):
    with open(path_read+'/'+list_dir[i],'r') as f:
        data=f.readlines()
        data=list(map(lambda x:x.strip().split(' '),data))
        list_name=[list_dir[i].split('.')[0] for j in range(len(data))]
        list_class=['head' for j in range(len(data))]
        df=pd.DataFrame(list_name,columns=['name'])
        df['class']=list_class
        df[list_columns[2:]]=data
        df.to_csv(path_save+'/'+list_dir[i].split('.')[0]+'.csv')
print('总数:',len(list_dir))