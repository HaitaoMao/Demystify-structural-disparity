import torch
import torch.nn as nn
import torch.nn.functional as F
from backward.model.MLPdrop import MLP

import pickle

class MLPnoreg(nn.Module):
    def __init__(self, args):
        super(MLPnoreg, self).__init__()
        self.args = args
        if args.model_arch:
            hidden_dimensions = args.model_arch
        else:
            hidden_dimensions = [args.hidden_dimension] * (args.num_layers - 1)
        
        self.hidden_dimensions = [args.num_feat] + hidden_dimensions + [args.num_class]
        # import ipdb; ipdb.set_trace()

        self.num_layer = len(self.hidden_dimensions) - 1
        self.MLP = MLP(args, self.hidden_dimensions)
        
        # self.model = MLP(self.hidden_dimensions+[args.num_class], args, is_encoder=False)
        # self.smooth_fn = smooth_loss(args, predefine_group="I-D2AD2")
        

        # TODO: load two part of parameters
    def set_others(self):
        # for param in self.parameters():
        #     print(param.shape)
        self.optimizer = torch.optim.Adam(self.parameters(), self.args.lr, weight_decay=self.args.weight_decay)

    def train_full_batch(self, data, epoch):
        # import ipdb; ipdb.set_trace()
        torch.cuda.empty_cache()
        self.train()
        self.optimizer.zero_grad()
        out, _, _ = self.forward(data)
        main_loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
        # reg_loss = self.args.lamda * self.smooth_fn.forward_check(data, hidden, False, order=1)  # self.model.hidden_list[self.args.smoothidx] 
        loss = main_loss
        loss.backward()        
        self.optimizer.step()

        return loss.item()

    def forward(self, data):
        # hidden = self.encoder(data.x)
        # check = self.decoder(hidden)
        self.hidden = self.MLP(data.x, is_retain_hidden=True)
        self.out = torch.log_softmax(self.hidden, dim=-1)
        return self.out, self.hidden, None
        # return self.decoder(self.encoder(data.x))
        
    # def set_grad_checker(self):
    #     self.grad_checker = Grad_checker(self.args, self.optimizer)

            
        