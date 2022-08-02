import os
import cv2
import numpy as np
import pandas as pd

def data_get(args):
    dict_choice={'TSF':'TSF(args)._load()',
                 'OD': 'OD(args)._load()'
                 }
    return eval(dict_choice[args.type])

class TSF(object):
    def __init__(self,args):
        self.args=args
    def _load(self):
        df1=pd.read_csv(self.args.data_root + self.args.data_name + '.csv')
        df1.columns=np.arange(len(df1.columns))
        self.df1_data=df1[self.args.TSF_column].astype(np.float32)
        return self._normalization()
    def _normalization(self):
        data=self.df1_data.values
        mean=np.mean(data,axis=0)
        std=np.std(data,axis=0)
        self.data_normalization=(data-mean)/std
        self.dict_dataset={'mean':mean,'std':std}
        return self._divide()
    def _divide(self):
        data_len=len(self.data_normalization)
        n=self.args.data_divide[0]+self.args.data_divide[1]
        self.data_train=self.data_normalization[0:data_len*self.args.data_divide[0]//n]
        self.data_test=self.data_normalization[len(self.data_train):]
        return self._prepare()
    def _prepare(self):
        len_index = len(self.data_train) - self.args.TSF_input - self.args.TSF_output + 1
        data_train=[0 for i in range(len_index)]
        true_train=[0 for i in range(len_index)]
        for i in range(len_index):
            data_train[i]=self.data_train[i:i+self.args.TSF_input]
            true_train[i] =self.data_train[i+self.args.TSF_input:i+self.args.TSF_input + self.args.TSF_output]
        self.dict_dataset['data_train']=np.stack(data_train[:],axis=0)
        self.dict_dataset['true_train'] = np.stack(true_train[:], axis=0)
        data_test=[0 for i in range(len(self.data_test)-self.args.TSF_input - self.args.TSF_output+1)]
        true_test=[0 for i in range(len(self.data_test)-self.args.TSF_input - self.args.TSF_output+1)]
        for i in range(len(self.data_test)-self.args.TSF_input - self.args.TSF_output+1):
            data_test[i]=self.data_test[i:i+self.args.TSF_input]
            true_test[i] =self.data_test[i+self.args.TSF_input:i+self.args.TSF_input + self.args.TSF_output]
        self.dict_dataset['data_test'] = np.stack(data_test[:], axis=0)
        self.dict_dataset['true_test'] = np.stack(true_test[:], axis=0)
        return self.dict_dataset

class OD(object):
    def __init__(self, args):
        self.args = args
    def _load(self):
        args=self.args
        path_img=args.data_root+args.data_name+'/img'
        path_label=args.data_root+args.data_name+'/label'
        dir_img = sorted(os.listdir(path_img))
        dir_label = sorted(os.listdir(path_label))
        self.len_data=len(dir_img)
        self.list_img = [0 for i in range(self.len_data)]
        self.list_label = [0 for i in range(self.len_data)]
        class_list=[]
        class_dict = {}
        for i in range(self.len_data):
            file_name=str(dir_img[i].split('.')[0])
            img=(cv2.imread(path_img+'/'+dir_img[i])/255).astype(np.float32)
            df_label=pd.read_csv(path_label+'/'+dir_label[i])
            len_df_label=len(df_label)
            frame=df_label[['Cx','Cy','w','h']].values.astype(np.int32)
            img,frame=self._resize(img,frame)
            frame=np.clip(frame,0,args.OD_size-1)
            self.list_img[i] = img
            class_name=df_label['class'].values
            class_onehot=np.zeros((len_df_label,args.OD_class),dtype=np.float32)
            class_onehot[:,:]=args.OD_smooth[0]
            for j in range(len_df_label):
                if class_name[j] not in class_list:
                    class_dict[class_name[j]]=[len(class_list),0,0,0,0]
                    class_onehot[j, len(class_list)]=args.OD_smooth[1]
                    class_list.append(class_name[j])
                    class_dict[class_name[j]][1:5]=[frame[j,2],frame[j,3],frame[j,2],frame[j,3]]
                else:
                    class_onehot[j,class_dict[class_name[j]][0]]=args.OD_smooth[1]
                    class_dict[class_name[j]][1] = min(class_dict[class_name[j]][1], frame[j, 2])
                    class_dict[class_name[j]][2] = min(class_dict[class_name[j]][3], frame[j, 3])
                    class_dict[class_name[j]][3] = max(class_dict[class_name[j]][2], frame[j, 2])
                    class_dict[class_name[j]][4] = max(class_dict[class_name[j]][4], frame[j, 3])
            confidence=np.zeros((len_df_label,1),dtype=np.float32)
            self.list_label[i]=[file_name,np.concatenate((frame,confidence,class_onehot),axis=1)]
        self.dict_dataset={}
        self.dict_dataset['class_list']=class_list
        self.dict_dataset['class_dict']=class_dict
        self.dict_dataset['stride'] = [args.OD_size//i for i in args.OD_output[0]]
        return self._divide()
    def _divide(self):
        n = self.args.data_divide[0] + self.args.data_divide[1]
        self.dict_dataset['img_train'] = self.list_img[0:self.len_data * self.args.data_divide[0] //n]
        self.dict_dataset['img_test'] = self.list_img[self.len_data * self.args.data_divide[0] // n:]
        self.dict_dataset['label_train'] = self.list_label[0:self.len_data * self.args.data_divide[0] //n]
        self.dict_dataset['label_test'] = self.list_label[self.len_data * self.args.data_divide[0] // n:]
        return self.dict_dataset
    def _resize(self,img,frame):
        w0 = len(img[0])
        h0 = len(img)
        if w0 == h0 == self.args.OD_size:
            pass
        elif w0 >= h0:  # 宽大于高
            w = self.args.OD_size
            h = int(len(img) * self.args.OD_size / len(img[0]))
            img = cv2.resize(img, (w, h))
            add_y = (w - h) // 2
            img = cv2.copyMakeBorder(img, add_y, w - h - add_y, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))
            frame[:,0] = frame[:,0] * w / w0
            frame[:,1] = frame[:,1] * h / h0 + add_y
            frame[:,2] = frame[:,2] * w / w0
            frame[:,3] = frame[:,3] * h / h0
        else:  # 宽小于高
            w = int(len(img[0]) * self.args.OD_size / len(img))
            h = self.args.OD_size
            img = cv2.resize(img, (w, h))
            add_x = (h - w) // 2
            img = cv2.copyMakeBorder(img, 0, 0, add_x, h - w - add_x, cv2.BORDER_CONSTANT, value=(0, 0, 0))
            frame[:, 0] = frame[:, 0] * w / w0 + add_x
            frame[:, 1] = frame[:, 1] * h / h0
            frame[:, 2] = frame[:, 2] * w / w0
            frame[:, 3] = frame[:, 3] * h / h0
        return img,frame



