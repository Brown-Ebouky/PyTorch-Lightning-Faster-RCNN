import os
# from turtle import back, forward
import torch
from torch import optim, nn, utils, Tensor
from torchvision.transforms import ToTensor
from torchvision.ops import box_iou, roi_pool, RoIPool
from torchvision import models
import pytorch_lightning as pl
from datasets.utils import to_right_box_format, create_anchors, associate_anchor_with_gt_bbox, min_max_scaler_anchors_with_image, unscale_min_max_anchors
from losses.__init__ import RPNLoss, DetectionLoss


# TODO: define the loss function to use
# use appropriately the different classes

class FasterRCNN(pl.LightningModule):
    def __init__(self, sliding_window, stride, roi_output_size, n_classes, n_proposals = 9, pos_threshold = 0.7, neg_threshold = 0.3):
        super().__init__()
        self.cnn_backbone = models.vgg16(pretrained=True).features

        # TODO: find a way to know the in_channels automatically
        # the current value has been obtained by printing the backbone model
        self.in_channels = 512
        self.rpn_network = RegionProposalNetwork(in_channels=self.in_channels, sliding_window=sliding_window, stride=stride, n_proposals=n_proposals)
        self.detector = FastRegionCNN(in_channels=self.in_channels, roi_output_size=roi_output_size, n_classes=n_classes)
        self.rpn_loss = RPNLoss()
        self.detection_loss = DetectionLoss()

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
        idx_pos, coor_bbox_anchors, label_bg_anchors, label_pos_anchors = associate_anchor_with_gt_bbox(anchors, bboxes, class_bboxes, self.pos_threshold, self.neg_threshold)


        # TODO:normalize value of the coordinates of bboxes before calculating the loss function
        # we scale the anchors (gt, predictions) coordinates using the image shape
        # anchors = min_max_scaler_anchors_with_image(im.shape[-2:], anchors)
        # coor_bbox_anchors = min_max_scaler_anchors_with_image(im.shape[-2:], coor_bbox_anchors)
        
        # calculate rpn loss
        rpn_loss = self.rpn_loss((prob_props, coor_props), (label_bg_anchors, coor_bbox_anchors), anchors)
        self.log("train_rpn_loss", rpn_loss)

        # apply the detection network on the proposals from rpn
        # print(torch.argmax(p_pred, dim=-1)) /// TAKE ONLY POSITIVE proposals? HOW ?? Thresholding probabilities?
        # pos_pred_inds = (label_bg_anchors[:,1] == 1).nonzero()[:,0]

        detect_loss = torch.zeros(1)

        # we consider the positive proposals for the detection
        for i in idx_pos:
            # TODO:check these values during training, the results are strange (TOO SMALL, WHY??)
            # print(coor_props[i])
            # unscaled_props = unscale_min_max_anchors(z.shape[-2:], coor_props[i])
            unscaled_props = coor_props[i].reshape(1,-1)
            # print("\n")
            print(unscaled_props)

            roi_probs, roi_regs = self.detector((z, [unscaled_props])) # List(bbox) to satisfy the format for roi_pool
            gt_roi_prob = label_pos_anchors[i].reshape(1, -1)
            gt_roi_reg = coor_bbox_anchors[i]

            # calculate the fast rcnn loss / detection loss
            roi_loss = self.detection_loss((roi_probs, roi_regs), (gt_roi_prob, gt_roi_reg))
            detect_loss = torch.add(detect_loss, roi_loss)
            
        print("\*"*30 + "\n")
        print(rpn_loss, detect_loss)

        return torch.add(rpn_loss, detect_loss)

    def configure_optimizers(self):
        optimizer = optim.SGD(self.parameters(), lr=1e-3)
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
    def __init__(self, in_channels, roi_output_size, n_classes) -> None:
        super().__init__()
        
        # flatten somewhere ??
        # there's already a roi pooling layer
        self.n_classes = n_classes
        self.roi_pool = RoIPool(roi_output_size, spatial_scale=1.0) # MAYBE Change the scale factor regarding to the size of the feature map
        self.roi_layer = nn.AdaptiveMaxPool2d((roi_output_size, roi_output_size))
        self.mid_layer = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_channels*roi_output_size*roi_output_size, 4096),
            nn.ReLU(),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            # nn.Linear(1024, 512),
            # nn.ReLU(),
            # nn.Linear(512, 256),
            # nn.ReLU()
        )
        self.cls_layer = nn.Sequential(
            nn.Linear(4096, n_classes),
            nn.Softmax()
        )
        self.reg_layer = nn.Linear(4096, 4*n_classes)


    def forward(self, x):
        # x = self.roi_layer(x)
        feature_map, boxes = x
        x = self.roi_pool(feature_map, boxes)
        # x = nn.Flatten(x) # flatten x to have 2 dim vector which can be processed by linear layers

        x = self.mid_layer(x)
        
        probs = self.cls_layer(x)

        regs = self.reg_layer(x)
        regs = regs.view(-1, self.n_classes, 4)

        return probs, regs



