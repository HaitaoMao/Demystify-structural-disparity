import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transfer.model.GCN import GCN
# from backward.model.MLP import MLP
from backward.model.MLPdrop import MLP
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from ogb.nodeproppred import Evaluator


class student(nn.Module):
    def __init__(self, args):
        super(student, self).__init__()
        self.args = args
        if args.model_arch:
            hidden_dimensions = args.model_arch
        else:
            hidden_dimensions = [args.hidden_dimension] * (args.num_layers - 1)
        
        self.hidden_dimensions = [args.num_feat] + hidden_dimensions + [args.num_class]

        self.num_layer = len(self.hidden_dimensions) - 1
        self.student_model = MLP(args, self.hidden_dimensions)
        # TODO: check whether here is good or not

    def set_others(self):
        self.student_optimizer = torch.optim.Adam(self.student_model.parameters(), self.args.lr, weight_decay=self.args.weight_decay)
        # self.student_optimizer2 = torch.optim.Adam(self.student_model.parameters(), self.args.lr, weight_decay=self.args.weight_decay)
 

    def train_student(self, data, out):
        # import ipdb; ipdb.set_trace()
        self.student_optimizer.zero_grad()
        criterion_train = torch.nn.NLLLoss()
        criterion_pseudo = torch.nn.KLDivLoss(reduction="batchmean", log_target=True) # 
        hidden = self.student_model(data.x)
        out_student = torch.log_softmax(hidden, dim=-1)        
        
        # import ipdb; ipdb.set_trace()                                                                                                 
        
        loss_train = criterion_train(out_student[data.train_mask], data.y[data.train_mask])
        loss_pseudo = criterion_pseudo(out_student, out)
        # ipdb.set_trace()
        loss = (self.args.lamda) * loss_train + (1-self.args.lamda) * loss_pseudo
        loss.backward()
        self.student_optimizer.step()
        # ipdb.set_trace()

        return loss_pseudo.item(), loss_train.item(), loss_pseudo.item()

    def forward(self, data):
        hidden = self.student_model(data.x)
        return hidden, None, None

    def test_student(self, data):
        self.student_model.eval()
        with torch.no_grad():
            hidden = self.student_model(data.x)
            out_test = torch.log_softmax(hidden, dim=-1)        
            results = self.test(data, out_test)    
        
        self.student_model.train()
        return out_test, results

    def test(self, data, out):
        y_pred = out.argmax(dim=-1, keepdim=True)

        if len(data.y.shape) == 1:
            y = data.y.unsqueeze(dim=1)  # for non ogb datas
        else:
            y = data.y

        # y = torch.squeeze(data.y)

        evaluator = Evaluator(name='ogbn-arxiv')
        train_acc = evaluator.eval({
            'y_true': y[data.train_mask],
            'y_pred': y_pred[data.train_mask],
        })['acc']
        valid_acc = evaluator.eval({
            'y_true': y[data.val_mask],
            'y_pred': y_pred[data.val_mask]
        })['acc']
        test_acc = evaluator.eval({
            'y_true': y[data.test_mask],
            'y_pred': y_pred[data.test_mask],
        })['acc']
        
        return train_acc, valid_acc, test_acc, y_pred

# CUDA_VISIBLE_DEVICES=1 python main_transfer.py  --algo_name GCN_MLP --te_lr 0.05 --st_lr 0.01  --te_weight_decay 5e-07 --te_dropout 0.5  --num_split 1 --lamda 0.6  --is_distill 1 --dataset Chameleon --st_dropout 0.4