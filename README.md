***
#&#10084;项目使用须知&#10084;
***
####&#9733;更新时间: 2022.07.25
***
####&#9733;内容
>&#10003;包含<时间序列预测>和<YOLO5目标检测>模块，有少量数据，只需要调整args中的参数，在run.py中就可直接运行  
>&#10003;代码结构清晰，分为数据处理(data_get)、模型加载(loss_get)、损失函数(loss_get)、训练(train_get)、测试(train_get)和指标(metric)模块  
>&#10003;对初学者友好，变量名完整，没有各种错综复杂的嵌套调用，可以一步步的调试、查看细节，便于学习整个项目是怎么运行的  
***
####&#9733;时间序列预测TSF
>&#10003;LSTM  
>```
>'--model', default='LSTM'
>'--model_param', default=[64,2] #[LSTM输出维度,LSTM层数]
>```
>&#10003;LSTM_CNN  
>```
>'--model', default='LSTM_CNN'
>'--model_param', default=[64,2] #[LSTM输出维度,LSTM层数]
>```
>&#10003;SCINet  
>```
>'--model', default='SCINet'
>'--model_param', default=[3,1] #[奇偶分解层数，模型堆叠数]
>```
>&#10003;待更新中...
***
####&#9733;目标检测OD
>&#10003;YOLO5  
>```
>'--model', default='YOLO5'
>'--model_param', default=[16,1] #[第一层输出维度,C3层残差数] #YOLO5n[16,1],YOLO5s[32,1],YOLO5m[48,2],YOLO5l[64,3],YOLO5x[80,4]
>'--OD_output', default=[[80,40,20],[3]]
>'--OD_anchor', default=[[[10,13],[16,30],[33,23]],[[30,61],[62,45],[59,119]],[[116,90],[156,198],[373,326]]]
>>```
>&#10003;待更新中...
***
####&#9733;其他
>&#10003;项目github地址:https://github.com/TWK2022/Integrate-Pytorch  
>&#10003;作者: TWK2022  
>&#10003;QQ及邮箱: 1024565378@qq.com  
***
