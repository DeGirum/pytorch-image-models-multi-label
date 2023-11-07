import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    "Non weighted version of Focal Loss"
    def __init__(self, alpha=.25, gamma=2):
        super(FocalLoss, self).__init__()
        # self.alpha = torch.tensor([alpha, 1-alpha]).cuda()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, input, target):
        # print ("initial input size----------------",input.size(0),input.size(1), input.dim())
        # if input.dim()>2:
        #     input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
        #     print ("--------------------",input.size)
        #     input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
        #     print ("after transpose-------------",input.size)
        #     input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
        #     print ("input final shape ------------------",input.size)
        # target = target.view(-1,1)

        BCE_loss = F.binary_cross_entropy_with_logits(input, target, reduction='none')
        # targets = targets.double()
        target = target.to(torch.int64)
        # if self.alpha is not None:
        #     if self.alpha.type()!=input.data.type():
        #         self.alpha = self.alpha.type_as(input.data)
        #     at = self.alpha.gather(0,target.data)

        at = self.alpha*target + (1-self.alpha)*(1-target) 

        # at = self.alpha.gather(0, target.view(-1))
        pt = torch.exp(-BCE_loss)
        F_loss = at*(1-pt)**self.gamma * BCE_loss
        return F_loss.mean()