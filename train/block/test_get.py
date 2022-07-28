import time
import torch
import numpy as np
import pandas as pd
from block import metric


def test_get(args,dict_dataset,model):
    dict_choice={'TSF':'TSF(args)._test(dict_dataset,model)',
                 'OD': 'OD(args)._test(dict_dataset,model)'
                 }
    return eval(dict_choice[args.type])

class TSF(object):
    def __init__(self,args):
        self.args=args
    def _test(self,dict_dataset,model):
        args=self.args
        model.eval()
        time_start = time.time()
        test_all = torch.tensor(np.stack(dict_dataset['data_test'], axis=0)).to(args.device)
        true_all = torch.tensor(np.stack(dict_dataset['true_test'], axis=0)).to(args.device)
        with torch.no_grad():
            pred_all = model(test_all)
        pred_denormalization = pred_all * torch.tensor(dict_dataset['std']).to(args.device)\
                               + torch.tensor(dict_dataset['mean']).to(args.device)
        true_denormalization = true_all * torch.tensor(dict_dataset['std']).to(args.device)\
                               + torch.tensor(dict_dataset['mean']).to(args.device)
        time_end = time.time()
        print('| 模型:{} | 数据集:{} | 变量:{} | 输入长度:{} | 预测长度:{} | 批量大小:{} | 损失函数:{} |'
              .format(args.model,args.data_name,args.TSF_column,args.TSF_input,args.TSF_output,args.batch,args.loss))
        print('| 测试集数量:{} | 测试时间:{:.2f}s | 归一化 mae:{:.4f} mse:{:.4f} | 真实 mae:{:.4f} mse:{:.4f} |'
              .format(len(true_denormalization),time_end - time_start,metric.mae(pred_all,true_all),
                      metric.mse(pred_all,true_all),metric.mae(pred_denormalization,true_denormalization),
                      metric.mse(pred_denormalization,true_denormalization)))
        if args.TSF_save:
            pred_save=pred_denormalization[:,-1].detach().cpu().numpy()
            true_save=true_denormalization[:,-1].detach().cpu().numpy()
            for i in range(len(args.TSF_column)):
                pd.DataFrame(np.stack([pred_save[:,i],true_save[:,i]],axis=1),
                             columns=[str(args.TSF_column[i])+'_pred_last',str(args.TSF_column[i])+'_true_last'])\
                    .to_csv(r'../result/'+args.name+'_'+'column'+str(args.TSF_column[i])+'_last.csv')
        if args.TSF_plot[0]:
            import matplotlib.pyplot as plt
            x=np.arange(min(args.TSF_plot[1],len(true_denormalization)))
            pred_plot = pred_denormalization[:,-1][0:len(x)].detach().cpu().numpy()
            true_plot = true_denormalization[:,-1][0:len(x)].detach().cpu().numpy()
            for i in range(len(args.TSF_column)):
                plt.title(args.name+'_column:'+str(args.TSF_column[i])+'_last')
                plt.plot(x,true_plot[:,i],color='green',label='true')
                plt.plot(x,pred_plot[:,i],color='cyan',label='pred')
                plt.savefig('../result/'+args.name+'_'+'column'+str(args.TSF_column[i])+'_last.png')
                plt.show()

class OD(object):
    def __init__(self,args):
        self.args=args
    def _test(self,dict_dataset,model):
        args = self.args
        model.eval()
        time_start = time.time()
        test_loader=test_get_loader(args,dict_dataset)
        for test_all, mask_all, true_all, name_all in test_loader:
            with torch.no_grad():
                pred_all = model(test_all)
        accuracy,precision = metric.accuracy_precision(
            len(args.OD_output[0]), args.OD_confidence_threshold, pred_all, mask_all, true_all)
        time_end = time.time()
        print('| 模型:{} | 数据集:{} | 图片大小:{} | 类别数:{} | 置信度阈值:{} | 批量大小:{} | 损失函数:{} |'
              .format(args.model, args.data_name, args.OD_size, args.OD_class, args.OD_confidence_threshold,
                      args.batch, args.loss))
        print('| 测试集数量:{} | 测试时间:{:.2f}s | 准确率:{:.4f} 精确率:{:.4f} |'
              .format(len(test_all), time_end - time_start, accuracy,precision))
        if args.OD_plot[0]:
            import cv2
            anchor = torch.tensor(args.OD_anchor).to(args.device)
            for i in range(min(len(test_all),args.OD_plot[1])):
                list_choose=[0 for j in range(len(args.OD_output[0]))]
                for j in range(len(args.OD_output[0])):
                    pred_all[j][i][:, :, :, 0]=(2*pred_all[j][i][:, :, :, 0]-0.5+torch.arange(args.OD_output[0][j])
                                                .to(args.device))*dict_dataset['stride'][j]
                    pred_all[j][i][:, :, :, 1]=(2*pred_all[j][i][:, :, :, 1]-0.5+torch.arange(args.OD_output[0][j])
                                                .unsqueeze(1).to(args.device))*dict_dataset['stride'][j]
                    for k in range(args.OD_output[1][0]):
                        pred_all[j][i][k, :, :, 2:4]=4*pred_all[j][i][k, :, :, 2:4]*anchor[j][k]
                    mask_choose = torch.where(pred_all[j][i][:, :, :, 4] > args.OD_confidence_threshold, True, False)
                    list_choose[j]=pred_all[j][i][mask_choose]
                choose = torch.cat(list_choose, axis=0)
                if len(choose)==0:
                    print('!没有任何达标的预测框!')
                    break
                choose = choose[0:min(len(choose),args.OD_plot_screen)]
                choose = torch.stack(sorted(list(choose), key=lambda x: x[4], reverse=True)[:],axis=0)
                choose[:,0:2]=choose[:,0:2]-1/2*choose[:,2:4]
                choose[:,2:4]=choose[:,0:2]+choose[:,2:4]
                img = (test_all[i].cpu().numpy() * 255).astype(np.uint8)
                img_all=img.copy()
                all_choose=choose.type(torch.int32).cpu().numpy()
                for j in range(len(all_choose)):
                    cv2.rectangle(img_all, (all_choose[j][0],all_choose[j][1]), (all_choose[j][2],all_choose[j][3]),
                                  color=(0,255,0), thickness=2)
                cv2.imshow(name_all[i]+'_99frame', img_all)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
                mask_choose=torch.ones(len(choose),dtype=bool)
                index_choose=torch.arange(0,len(choose))
                for j in range(len(choose)-1):
                    if mask_choose[j]:
                        for k,index in zip(choose[j+1:][mask_choose[j+1:]],index_choose[j+1:][mask_choose[j+1:]]):
                            if metric.iou(k[0:4],choose[j][0:4])>args.OD_plot_threshold:
                                mask_choose[index]=False
                choose=choose[mask_choose].cpu().numpy()
                _xy=choose[:,0:4].astype(np.int32)
                _class=np.argmax(choose[:,5:],axis=1)
                for j in range(len(choose)):
                    cv2.rectangle(img, (_xy[j][0],_xy[j][1]), (_xy[j][2],_xy[j][3]),color=(0,255,0), thickness=2)
                    cv2.putText(img,dict_dataset['class_list'][_class[j]], (_xy[j][0]+3,_xy[j][1]+10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1)
                cv2.imshow(str(name_all[i])+'_NMS_'+len(choose)+'frame', img)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
                cv2.imwrite('../result/' + args.name + '_' + name_all[i] + '.png', img)

 # ------------------------------------------------------------------------------------------------------------------ #
def test_get_loader(args,dict_dataset):
    dict_choice={'OD': 'OD_dataset(args,dict_dataset)'
                 }
    return torch.utils.data.DataLoader(eval(dict_choice[args.type]),
                                       batch_size=len(dict_dataset['img_test']),shuffle=True,drop_last=True)

class OD_dataset(torch.utils.data.Dataset):
    def __init__(self,args,dict_dataset):
        self.args=args
        self.dict_dataset=dict_dataset
        self.anchor=torch.tensor(args.OD_anchor).to(args.device)
    def __len__(self):
        return len(self.dict_dataset['img_test'])
    def __getitem__(self, index):
        args=self.args
        dict_dataset=self.dict_dataset
        test=torch.tensor(dict_dataset['img_test'][index]).to(args.device)
        list_mask=[0 for i in range(len(args.OD_output[0]))]
        list_label=[0 for i in range(len(args.OD_output[0]))]
        for i,stride in enumerate(dict_dataset['stride']):
            mask=torch.zeros((args.OD_output[1][0],args.OD_output[0][i],args.OD_output[0][i]),dtype=bool).to(args.device)
            label=torch.zeros((args.OD_output[1][0],args.OD_output[0][i],args.OD_output[0][i],5+args.OD_class)).to(args.device)
            label_test = torch.tensor(dict_dataset['label_test'][index][1]).to(args.device)
            xy_stride=label_test[:,0:2]/stride
            label_test[:, 0:2]=xy_stride%1
            label_test=label_test.repeat(len(args.OD_anchor[i]),1,1)
            for j in range(len(args.OD_anchor[i])):
                label_test[j,:,2:4]=label_test[j,:,2:4]/ self.anchor[i][j]
            x_grid,y_grid=xy_stride[:,0].type(torch.int32),xy_stride[:,1].type(torch.int32) #原标签
            x_add=torch.clamp(x_grid+2*torch.round(xy_stride[:,0]-x_grid).type(torch.int32)-1,0,args.OD_output[0][i]-1) #增加的标签
            y_add=torch.clamp(y_grid+2*torch.round(xy_stride[:,1]-y_grid).type(torch.int32)-1,0,args.OD_output[0][i]-1)
            for j in range(len(xy_stride)): #原标签
                mask[:,y_grid[j],x_grid[j]]=True
                label[:,y_grid[j],x_grid[j],:]=label_test[:,j]
            for j in range(len(xy_stride)): #增加的标签
                if not mask[0,y_grid[j],x_add[j]]:
                    mask[:, y_grid[j], x_add[j]]=True
                    label[:, y_grid[j], x_add[j],:] = label_test[:,j]
                    if x_add[j]<x_grid[j]:
                        label[:, y_grid[j], x_add[j],0] = label[:, y_grid[j], x_add[j],0] + 1
                    else:
                        label[:, y_grid[j], x_add[j], 0] = label[:, y_grid[j], x_add[j], 0] - 1
                if not mask[0,y_add[j],x_grid[j]]:
                    mask[:, y_add[j], x_grid[j]]=True
                    label[:, y_add[j], x_grid[j],:] = label_test[:,j]
                    if y_add[j] < y_grid[j]:
                        label[:, y_add[j], x_grid[j], 1] = label[:, y_add[j], x_grid[j], 1] + 1
                    else:
                        label[:, y_add[j], x_grid[j], 1] = label[:, y_add[j], x_grid[j], 1] - 1
            mask=torch.where((0.25<label[:,:,:,2])&(0.25<label[:,:,:,3])&(label[:,:,:,2]<4)&(label[:,:,:,3]<4),True,False)
            list_mask[i] = mask
            list_label[i] = label
            name=dict_dataset['label_test'][index][0]
        return test, list_mask, list_label, name
