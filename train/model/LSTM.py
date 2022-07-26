#LSTM:model_param---[LSTM输出维度,LSTM层数]---推荐[64,2]
import torch
class LSTM(torch.nn.Module):
    def __init__(self,args):
        super(LSTM, self).__init__()
        self.args=args
        self.lstm = torch.nn.LSTM(input_size=len(args.TSF_column), hidden_size = args.model_param[0],
                             num_layers=args.model_param[1], dropout=0.2)
        self.Dropout = torch.nn.Dropout(0.5)
        self.dense=torch.nn.Linear(args.model_param[0]*args.TSF_input,args.TSF_output*len(args.TSF_column))
    def forward(self,input1):
        args=self.args
        x=input1.permute(1,0,2)
        x, (h_n, c_n) = self.lstm(x)
        x = self.Dropout(x)
        x = x.permute(1,2,0)
        x=x.reshape(len(x),-1)
        x= self.dense(x).reshape(len(x),args.TSF_output,len(args.TSF_column))
        return x