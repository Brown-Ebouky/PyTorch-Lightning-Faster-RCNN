import os
# from turtle import back, forward
import torch
from torch import optim, nn, utils, Tensor
from torchvision.transforms import ToTensor
from torchvision.ops import box_iou
from torchvision import models
import pytorch_lightning as pl
from datasets.utils import to_right_box_format, create_anchors, associate_anchor_with_gt_bbox, min_max_scaler_anchors_with_image


# TODO: define the loss function to use
# use appropriately the different classes

class FasterRCNN(pl.LightningModule):
    def __init__(self, sliding_window, stride, roi_output_size, n_proposals = 9, pos_threshold = 0.7, neg_threshold = 0.3):
        super().__init__()
        self.cnn_backbone = models.vgg16(pretrained=True).features

        # TODO: find a way to know the in_channels automatically
        # the current value has been obtained by printing the backbone model
        
        self.rpn_network = RegionProposalNetwork(in_channels=512, sliding_window=sliding_window, stride=stride, n_proposals=n_proposals)
        self.detector = FastRegionCNN(roi_output_size=roi_output_size, n_proposals=n_proposals)
        self.rpn_loss = RPNLoss()

        self.pos_threshold = pos_threshold
        self.neg_threshold = neg_threshold
        self.sliding_window = sliding_window
        self.stride = stride # need to check afterwards that the stride lorequal to sliding_window

        self.loc_mean = torch.tensor([0.,0.,0.,0.])
        self.loc_std = torch.tensor([])
        


    def training_step(self, batch, batch_idx):
        # y shape = (nb_rois, 4)
        # y might be a tuple with right rois
        im, target = batch

        bboxes = target["bbox"][0] # remove dim=1 of the batch at the beginning 
        class_bboxes = target["class_bbox"][0]

        z = self.cnn_backbone(im) #je_decouvre_nana_mais_cest_complique_dans_tout_ca
        
        prob_props, coor_props = self.rpn_network(z)
        
        # calculate ground truth box coordinates and fg labels
        anchors = create_anchors(z.shape[-2], z.shape[-1], w_size=self.sliding_window, stride=self.stride)
        coor_bbox_anchors, label_bg_anchors, label_pos_anchors = associate_anchor_with_gt_bbox(anchors, bboxes, class_bboxes, self.pos_threshold, self.neg_threshold)


        # TODO:normalize value of the coordinates of bboxes before calculating the loss function
        # we scale the anchors (gt, predictions) coordinates using the image shape
        anchors = min_max_scaler_anchors_with_image(im.shape[-2:], anchors)
        coor_bbox_anchors = min_max_scaler_anchors_with_image(im.shape[-2:], coor_bbox_anchors)

        # calculate rpn loss
        loss = self.rpn_loss((prob_props, coor_props), (label_bg_anchors, coor_bbox_anchors), anchors)
        self.log("train_loss", loss)

        # apply the detection network on the proposals from rpn
        

        # transform the predictions into the right format (x1,y1,x2,y2)
        # pred_boxes = to_right_box_format(pred_boxes)

        # pred_ious = box_iou(pred_boxes, y)
        # inds_thre = (pred_ious >= self.threshold).nonzero()[:,1] #0
        # inds_max = torch.argmax(pred_ious, dim=0)

        # assert inds_thre.size() ==  inds_max.size()

        # inds = torch.cat((inds_thre, inds_max)).unique()

        # construct ground truth values for the proposals using the indices
        # gt_probs = torch.zeros(probs.size())
        # gt_probs[inds] = 1



        # perform the roi pooling by using the proposals predicted as positive
        # for loop ??
        # for i in inds:
        #     x1, y1, x2, y2 = pred_boxes[i]
        #     roi = z[y1:y2, x1:x2]
        #     prob_roi, reg_roi = self.detector(roi)

            # calculate loss
        
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


class RegionProposalNetwork(nn.Module):
    def __init__(self, in_channels, sliding_window, stride, n_proposals) -> None:
        super().__init__()
        self.n_proposals = n_proposals

        # TODO:may be consider to have the sliding window done with a stride equals to the window size instead of a s=1 stride ??
        self.mid_layer = nn.Sequential(
            nn.Conv2d(in_channels, 512, sliding_window, stride), 
            nn.ReLU()
        )
        self.cls_layer = nn.Conv2d(512, 2*n_proposals, 1,)
        # nn.Flatten(start_dim=-3, end_dim=-2),
        # nn.Softmax(dim=-1)
        
        self.reg_layer = nn.Conv2d(512, 4*n_proposals, 1)
        self.softmax = nn.Softmax(dim=-1)
        self.flatten= nn.Flatten(start_dim=0, end_dim=-2)

    def forward(self, x):
        # x is the last feature map from the backbone conv network

        x = self.mid_layer(x) # dim: 512 X m X n 
        _, c, m, n = x.shape

        probs = self.cls_layer(x) # dim: 2*k X m X n 

        probs = probs.view(m, n, -1, 2)
        probs = self.softmax(probs)
        probs = self.flatten(probs) # dim: m*n*k X 2

        regs = self.reg_layer(x) # dim: 4*k X m X n
        regs = regs.view(m, n, -1, 4)
        regs = self.flatten(regs) # dim: m*n*k X 4

        return probs, regs    


class FastRegionCNN(nn.Module):
    def __init__(self, roi_output_size, n_proposals) -> None:
        super().__init__()
        
        # flatten somewhere ??
        # there's already a roi pooling layer
        self.roi_layer = nn.AdaptiveMaxPool2d((roi_output_size, roi_output_size))
        self.mid_layer = nn.Sequential(
            nn.Linear(roi_output_size*roi_output_size, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU()
        )
        self.cls_layer = nn.Sequential(
            nn.Linear(256, 2*n_proposals),
            nn.Softmax()
        )
        self.reg_layer = nn.Linear(256, 4*n_proposals)


    def forward(self, x):
        x = self.roi_layer(x)
        x = self.mid_layer(x)
        probs = self.cls_layer(x)
        regs = self.reg_layer(x)
        return probs, regs



class RPNLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.epsilon_log = 1e-5

    def forward(self, x_pred, x_gt, anchors):
        return self.region_loss(x_pred, x_gt, anchors)

    def region_loss(self, x_pred, x_gt, anchors):
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
        
        print(reg_loss)
        print("\\"*30)

        # TODO: add the lambda coefficient on the reg loss 
        return torch.add(p_loss, reg_loss)