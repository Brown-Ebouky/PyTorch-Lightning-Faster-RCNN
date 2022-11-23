import torch
from torch import nn

class RPNLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.epsilon_log = 1e-5

    def forward(self, x_pred, x_gt, anchors):
        return self._region_loss(x_pred, x_gt, anchors)

    def _region_loss(self, x_pred, x_gt, anchors):
        """
        Desc: calculate the region proposal loss function

        Args:
            x_pred: prediction data containing the probabilities bg/fg and coordinates of proposals
            x_gt: ground truth probabilities and coordinates
            anchors: anchor associated with each prediction
        """
        p_pred, t_pred = x_pred
        p_gt, t_gt = x_gt
        
        n_anchors, _ = p_pred.shape
        
        p_loss = torch.sum(nn.CrossEntropyLoss()(p_pred, p_gt)) / n_anchors


        t_x = torch.div(torch.sub(t_pred[:, 0], anchors[:, 0]), anchors[:, 2]).reshape(-1,1)
        t_y = torch.div(torch.sub(t_pred[:, 1], anchors[:, 1]), anchors[:, 3]).reshape(-1,1)
        t_w = torch.log(torch.abs(torch.div(t_pred[:, 2], anchors[:, 2])) + self.epsilon_log).reshape(-1,1)
        t_h = torch.log(torch.abs(torch.div(t_pred[:, 3], anchors[:, 3])) + self.epsilon_log).reshape(-1,1)

        t_x_star = torch.div(torch.sub(t_gt[:, 0], anchors[:, 0]), anchors[:, 2]).reshape(-1,1)
        t_y_star = torch.div(torch.sub(t_gt[:, 1], anchors[:, 1]), anchors[:, 3]).reshape(-1,1)
        t_w_star = torch.log(torch.abs(torch.div(t_gt[:, 2], anchors[:, 2])) + self.epsilon_log).reshape(-1,1)
        t_h_star = torch.log(torch.abs(torch.div(t_gt[:, 3], anchors[:, 3])) + self.epsilon_log).reshape(-1,1)

        t_glob = torch.cat((t_x, t_y, t_w, t_h), dim=1)
        t_glob_star = torch.cat((t_x_star, t_y_star, t_w_star, t_h_star), dim=1)

        
        # print(t_glob_star)
        reg_loss = nn.functional.smooth_l1_loss(t_glob, t_glob_star)
        
        # TODO: add the lambda coefficient on the reg loss 
        return torch.add(p_loss, reg_loss)


class DetectionLoss(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.cross_entropy = nn.CrossEntropyLoss()



    def forward(self, x_pred, x_gt):
        p_pred, t_pred = x_pred
        p_gt, t_gt = x_gt
        
        p_loss = self.cross_entropy(p_pred, torch.max(p_gt,1)[1])

        # select only the reg prediction of the right class and calculate the loss on that one
        idx_pos_class  = torch.max(p_gt,1)[1]
        reshaped_t_pred = t_pred[:,idx_pos_class,:].reshape(4,)

        reg_loss = nn.functional.smooth_l1_loss(reshaped_t_pred, t_gt)
      

        return torch.add(p_loss, reg_loss)