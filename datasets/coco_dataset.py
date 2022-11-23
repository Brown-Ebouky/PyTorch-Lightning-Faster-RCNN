import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, utils
# from .utils import create_anchors, create_base_anchors, associate_anchor_with_bbox

# from pycocotools import coco
import skimage.io as io
from pycocotools.coco import COCO
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

class CustomTransform(object):
    def __init__(self, desired_size = 600):
        # TODO: instead of using standard transforms, use only the albumentions one
        self.img_transform = transforms.Compose([
            # transforms.Resize(desired_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        self.common_transform = A.Compose([
            # A.RandomCrop(width=450, height=450),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.2),
            # ToTensorV2()
        ], bbox_params=A.BboxParams(format='coco', label_fields=['class_labels']))

        
        
    def __call__(self, image, bboxes, class_labels):
        
        transformed = self.common_transform(image=image, bboxes=bboxes, class_labels=class_labels)
        image = self.img_transform(transformed["image"])
        # image = transformed["image"]
        
        # transform list of tuples into tensors
        bbox = list(map(list, transformed['bboxes']))
        bbox = torch.tensor(bbox, dtype=torch.float)

        class_labels = list(map(list, transformed['class_labels']))
        class_labels = torch.tensor(class_labels, dtype=torch.float)

        return image, bbox, class_labels #transformed['bboxes'], transformed['class_labels']


class CocoDataset(Dataset):
    def __init__(self, dataDir, dataType):
        super().__init__()
        self.dataDir = dataDir
        self.dataType = dataType
        
        # self.w_size = w_size
        # self.stride = stride
        
        self.transforms = CustomTransform()

        self.annFile='{}/annotations/instances_{}.json'.format(dataDir, dataType)
        
        self.coco = COCO(self.annFile)
        self.categories = [cat['name'] for cat in self.coco.loadCats(self.coco.getCatIds())]
        
        imgIds = self.coco.getImgIds()
        self.images = self.coco.loadImgs(imgIds)

        # annIds = self.coco.getAnnIds(imgIds)
        # self.annotations = self.coco.loadAnns(annIds)



    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        
        # TODO: do the supsampling of negative / positive anchors to have the good ratio
        # consider only anchors which are *inside* the image and drop the other

        path_img = '{}/images/{}/{}'.format(self.dataDir, self.dataType, self.images[idx]["file_name"])
        img = io.imread(path_img)

        annIds = self.coco.getAnnIds(imgIds=self.images[idx]['id'])
        anns = self.coco.loadAnns(annIds) 

        

        
        bbox = []
        classes = []
        named_classes = []

        for a in anns:
            bbox.append(a['bbox'])
            
            cat_name = self.coco.loadCats(a['category_id'])[0]['name']
            one_hot_cat = torch.zeros(len(self.categories))
            one_hot_cat[self.categories.index(cat_name)] = 1

            classes.append(one_hot_cat.tolist())
            named_classes.append(cat_name)
            # target['area_bbox'].append(a['area'])

        
         
        bbox = torch.tensor(bbox)
        classes = torch.tensor(classes)

        # transformed image, bboxes and labels
        # TODO: classes parameter in the transform might not work because one hotss. CHECK THAT!!
        img, bbox, classes = self.transforms(img, bbox, classes)


        # anchors = create_anchors(img.shape[-2], img.shape[-1], w_size=self.w_size, stride=self.stride)
        # bbox_anchors, y_anchors, label_pos_anchors = associate_anchor_with_bbox(anchors, bbox, classes, iou_pos_threshold=0.7, iou_neg_threshold=0.2)

        target = {}
        # target['img'] = img
        target['image_id'] = self.images[idx]['id']
        target["bbox"] = bbox
        target["class_bbox"] = classes

        # target['anchors'] = anchors
        # target['bbox'] = bbox_anchors
        # target['class_bg'] = y_anchors
        # target['class_anchors'] = label_pos_anchors

        # target['area_bbox'] = []
        # if self.transforms:
        #     target = self.transforms(img, target)

        return img, target

    