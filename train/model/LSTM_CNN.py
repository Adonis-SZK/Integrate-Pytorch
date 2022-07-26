#LSTM_CNN:model_param---[LSTM输出维度,LSTM层数]---推荐[64,2]
import torch
class conv1d_bn_relu(torch.nn.Module):
    def __init__(self,in1,out1,kernel_size=1,stride=1,padding=9):
        super().__init__()
        self.Conv1d = torch.nn.Conv1d(in1, out1, kernel_size=kernel_size, stride=stride,padding=padding)
        self.BatchNorm1d = torch.nn.BatchNorm1d(out1)
        self.SiLU = torch.nn.SiLU()
        self.Dropout =torch.nn.Dropout(0.2)
    def forward(self,input1):
        x = self.Conv1d(input1)
        x = self.BatchNorm1d(x)
        x = self.SiLU(x)
        x = self.Dropout(x)
        return x

class residual(torch.nn.Module):
    def __init__(self,in1):
        super().__init__()
        self.conv1d_bn_relu_1=conv1d_bn_relu(in1,in1,kernel_size=1,stride=1,padding=0)
        self.conv1d_bn_relu_2=conv1d_bn_relu(in1,in1,kernel_size=3,stride=1,padding=1)
        self.conv1d_bn_relu_3 = conv1d_bn_relu(in1, in1, kernel_size=1, stride=1, padding=0)
    def forward(self, input1):
        x = self.conv1d_bn_relu_1(input1)
        x = self.conv1d_bn_relu_2(x)
        input1=self.conv1d_bn_relu_3(input1)
        return x+input1

class residual_max(torch.nn.Module):
    def __init__(self, in1):
        super().__init__()
        self.conv1d_bn_relu_1=conv1d_bn_relu(in1, in1//2, kernel_size=1, stride=1, padding=0)
        self.residual_1 = residual(in1//2)
        self.residual_2 = residual(in1//2)
        self.residual_3 = residual(in1//2)
        self.residual_4 = residual(in1//2)
        self.conv1d_bn_relu_2 = conv1d_bn_relu(in1, in1//2, kernel_size=1, stride=1, padding=0)
        self.conv1d_bn_relu_3 = conv1d_bn_relu(in1, in1, kernel_size=1, stride=1, padding=0)
    def forward(self, input1):
        input2=self.conv1d_bn_relu_1(input1)
        x=self.residual_1(input2)
        x=self.residual_2(x)
        x=self.residual_3(x)
        x=self.residual_4(x)
        x=x+input2
        input1=self.conv1d_bn_relu_2(input1)
        x=torch.cat([x,input1],axis=1)
        x=self.conv1d_bn_relu_3(x)
        return x

class LSTM_CNN(torch.nn.Module):
    def __init__(self,args):
        super().__init__()
        self.args=args
        self.lstm = torch.nn.LSTM(input_size=len(args.TSF_column), hidden_size = args.model_param[0],
                             num_layers=args.model_param[1], dropout=0.2)
        self.Dropout_1=torch.nn.Dropout(0.2)
        self.conv1d_bn_relu = conv1d_bn_relu(args.model_param[0], args.model_param[0], kernel_size=1,stride=1, padding=0)
        self.Dropout_2=torch.nn.Dropout(0.2)
        self.residual_max=residual_max(args.model_param[0])
        self.Dropout_3 = torch.nn.Dropout(0.5)
        self.dense=torch.nn.Linear(args.model_param[0]*args.TSF_input,args.TSF_output*len(args.TSF_column))
    def forward(self,input1):
        args=self.args
        x=input1.permute(1,0,2)
        x, (h_n, c_n) = self.lstm(x)
        x = x.permute(1, 2, 0)
        x = self.Dropout_1(x)
        x=self.conv1d_bn_relu(x)
        x = self.Dropout_2(x)
        x = self.residual_max(x)
        x = self.Dropout_3(x)
        x=x.reshape(len(x),-1)
        x= self.dense(x).reshape(len(x),args.TSF_output,len(args.TSF_column))
        return x