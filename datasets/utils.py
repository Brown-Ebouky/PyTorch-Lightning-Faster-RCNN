import torch
from torchvision.ops import box_iou


def associate_anchor_with_gt_bbox(anchors, bbox, label_bbox, iou_pos_threshold, iou_neg_threshold):
        """
        Desc: this function labels the anchors with the corresponding bbox and label wrt 
        background/foreground

        Arguments:
            anchors: tensor of anchors 
            bbox: ground truth bounding boxes coordinates for the image
            label_bbox: ground truth label of the object in the bboxes
            iou_pos_threshold: threshold used to define foreground anchors
            iou_neg_threshold: threshold used to define background anchors
        """

        # put the entries to the right format for box_iou (x1,y1,x2,y2)
        tmp_anchors = to_right_box_format(anchors)
   
        # TODO: transform bbox to the right format afterwards to have a better calculation of the loss because right now
        # bbox: (x1,y1,w,h) but anchors is in (x_c, y_c, w, h)
        tmp_bbox = from_coco_format_to_right(bbox) # because bbox is in coco format
        
        ious = box_iou(tmp_anchors, tmp_bbox) # matrix n X m with anchors n*4 and bbox m*4
      
        # TODO: check the iou calculation to make sure that it's working well

        # transform bbox to format of anchors (x_c, y_c, w, h)
        bbox = from_coco_format_to_anchor_format(bbox)

        y_anchors = torch.zeros(anchors.shape[0], 2) # initialize with -1 values *(-1)

        bbox_anchors = torch.zeros(anchors.shape)

        obj_label_anchors = torch.zeros(anchors.shape[-2], label_bbox.shape[-1])

        # label positive anchors
         
        idx_max = torch.argmax(ious, dim=0)
        label_box_max = bbox

        idx_respect_threshold = (ious >= iou_pos_threshold).nonzero()


        idx_respect_threshold_anchors = idx_respect_threshold[:, 0]
        label_box_respect_threshold = bbox[idx_respect_threshold[:, 1], :]
        

        y_anchors[idx_respect_threshold_anchors,1] = 1
        y_anchors[idx_max,1] = 1

        obj_label_anchors[idx_respect_threshold_anchors,:] = label_bbox[idx_respect_threshold[:,1],:]
        obj_label_anchors[idx_max,:] = label_bbox

        # if idx_respect_threshold.shape[0] != 0:
        bbox_anchors[idx_respect_threshold_anchors,:] = label_box_respect_threshold
        bbox_anchors[idx_max] = label_box_max

        # assign object labels to positive anchors for the detection part (Fast RCNN)
        # idx_pos = torch.cat((idx_respect_threshold_anchors, idx_max)).unique()
        
        # label_pos_anchors = torch.zeros((idx_pos.shape[0], label_bbox.shape[1]))
        # label_pos_anchors[range(idx_respect_threshold_anchors.shape[0]),:] = label_bbox[idx_respect_threshold[:, 1], :]
        # TODO: how to consider the other indexes where we use the maximum? (need to find the except btw the index tensors)


        # label negative anchors

        idx_neg_anchors = (ious < iou_neg_threshold).nonzero()[:,0]
        y_anchors[idx_neg_anchors,0] = 1

        # indices of positive anchors
        idx_pos = torch.cat((idx_max, idx_respect_threshold_anchors)).unique()

        return idx_pos, bbox_anchors, y_anchors, obj_label_anchors



def create_anchors(height, width, w_size, stride, prop_scales = [128,256,512], prop_ratios = [0.5, 1, 2]):
    """
    Desc: this function create the different anchors with the desired size

    Arguments:
        height: height of the image used
        width: width of the image
        w_size: size of the window slided unto the image for the proposals
        stride: stride used when sliding the window 
        n_proposals: number of proposals to consider per window
        prop_sizes: base size of the proposals to consider
        prop_scales: scales to use on the base size of proposals
    """

    prop_per_window = len(prop_ratios)*len(prop_scales)
    n_iter = ((height - w_size) // stride + 1) * ((width - w_size) // stride + 1)
    
   
    base_anchors = create_base_anchors(base_size=1, prop_scales=prop_scales, prop_ratios=prop_ratios)
   
    anchors = base_anchors.repeat(n_iter, 1)

    shift_width = torch.arange(0, width - w_size + stride - (w_size - 1), stride) + w_size / 2 # -(w_size-1) is to make sure that the last value will be ok

    shift_height = torch.arange(0, height - w_size + stride - (w_size - 1), stride) + w_size / 2
    # anchors = torch.zeros((prop_per_window*n_iter, 4))

    # TODO: find a way to perform the operation in a vectorized manner
    # we need to have a matrix with all the combinations of shift_width and shift_height    
    
    # In this case, the following lines help to have a flatten array of the different center 
    # positions in the width dim for all the anchors
    # including the one extracted with ratio and scale
    tmp_shift_width = shift_width.repeat(prop_per_window, 1)
    tmp_shift_width = tmp_shift_width.T.reshape(1, -1)

    iter_per_line = prop_per_window * ((width - w_size) // stride + 1) 
    
    for i in range(len(shift_height)):
        # go through the loop line by line and we'll consider the corresponding shift the line
        deb_line = iter_per_line * i
        end_line =  iter_per_line * (i+1)

        anchors[deb_line:end_line, 0] = shift_height[i]
        anchors[deb_line:end_line, 1] = tmp_shift_width
        

    return anchors


def create_base_anchors(base_size, prop_scales = [128,256,512], prop_ratios = [0.5, 1, 2]):
    """
    Desc: create the basic anchors that will be used for the region proposal network

    Args:
        base_size: base size of an anchor
        prop_scale: scale of the basic anchor proposals
        prop_scales: scale to use for the basic anchors
    """

    n_anchors = len(prop_ratios) * len(prop_scales)
    base_anchors = torch.zeros(n_anchors, 4)
    for i in range(len(prop_scales)):
        for j in range(len(prop_ratios)):
            idx_anchor = i*len(prop_ratios) + j

            h_a = prop_scales[i] * base_size
            w_a = prop_scales[i] * base_size * prop_ratios[j]
            
            base_anchors[idx_anchor, 2] = w_a
            base_anchors[idx_anchor, 3] = h_a
    
    return base_anchors


def to_right_box_format(boxes):
    # transform box format from (x_c, y_c, w, h) to (x1,y1,x2,y2)
    new_boxes = torch.zeros(boxes.shape)
    new_boxes[:,0] = boxes[:,0] - torch.div(boxes[:, 2], 2, )
    new_boxes[:,1] = boxes[:,1] - torch.div(boxes[:, 3], 2, )
    new_boxes[:,2] = boxes[:,0] + torch.div(boxes[:, 2], 2, )
    new_boxes[:,3] = boxes[:,1] + torch.div(boxes[:, 3], 2, )
    
    return new_boxes

def from_coco_format_to_right(boxes):
    # transform box format from (x_min, y_min, w, h) to (x1, y1, x2, y2)
    new_boxes = torch.zeros(boxes.shape)
    new_boxes[:,0] = boxes[:,0]
    new_boxes[:,1] = boxes[:,1]
    new_boxes[:,2] = boxes[:,0] + boxes[:,2]
    new_boxes[:,3] = boxes[:,1] + boxes[:,3]
    
    return new_boxes

def from_coco_format_to_anchor_format(boxes):
    # args (boxes) in format (x1,y1,w,h) -- (x_c,y_c,w,h)
    new_boxes = torch.zeros(boxes.shape)
    new_boxes[:,0] = boxes[:,0] + torch.div(boxes[:,2], 2)
    new_boxes[:,1] = boxes[:,1] + torch.div(boxes[:,3], 2)
    new_boxes[:,2] = boxes[:,2] + boxes[:,2]
    new_boxes[:,3] = boxes[:,3] + boxes[:,3]
    
    return new_boxes

# TODO: these information must be calculated using the dataset directly and not this way

def min_max_scaler_anchors_with_image(img_shape, to_scale_anchors):
    height, width = img_shape
    min_cor_anchors = torch.tensor([0.5, 0.5, 0, 0])
    max_cor_anchors = torch.tensor([width-0.5, height-0.5, width, height])
    
    scaled_anchors = torch.zeros(to_scale_anchors.shape)
    scaled_anchors = ((to_scale_anchors - min_cor_anchors) / (max_cor_anchors - min_cor_anchors))
    scaled_anchors = 2*scaled_anchors - 1
    
    return scaled_anchors

def unscale_min_max_anchors(img_shape, anchors):
    height, width = img_shape
    min_cor_anchors = torch.tensor([0.5, 0.5, 0, 0])
    max_cor_anchors = torch.tensor([width-0.5, height-0.5, width, height])

    anchors = (anchors + 1) / 2
    anchors = anchors* (max_cor_anchors - min_cor_anchors) + min_cor_anchors

    return anchors