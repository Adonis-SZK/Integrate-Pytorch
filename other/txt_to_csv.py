#将txt格式转换为csv格式
import os
import numpy as np
import pandas as pd
#设置
# -------------------------------------------------------------------------------------------------------------------- #
path_label=r'.\label'
save_label=r'.\label_save'
list_columns=['class','Cx','Cy','w','h']
# -------------------------------------------------------------------------------------------------------------------- #
list_dir=os.listdir(path_label)
for i in range(len(list_dir)):
    with open(path_label+'/'+list_dir[i],'r') as f:
        data=f.readlines()
        data=list(map(lambda x:x.strip().split(' '),data))
        list_class=['person' for j in range(len(data))]
        df=pd.DataFrame(list_class,columns=['class'])
        df[list_columns[1:]]=data
        df.to_csv(save_label+'/'+list_dir[i].split('.')[0]+'.csv')
print('总数:',len(list_dir))

