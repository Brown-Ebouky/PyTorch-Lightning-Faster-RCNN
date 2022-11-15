# sample for fiftyone
"""
import fiftyone as fo
import fiftyone.zoo as foz

dataset = foz.load_zoo_dataset("quickstart")
session = fo.launch_app(dataset)
session.wait()

"""


# sample for the use of coco
from pycocotools.coco import COCO
import skimage.io as io

dataDir = "/home/brown/Documents/PaperImplementations/datasets/coco"
dataType = "train2014"
annFile='{}/annotations/instances_{}.json'.format(dataDir, dataType)
coco = COCO(annFile)
categories = coco.loadCats(coco.getCatIds())

catIds = coco.getCatIds(catNms=['dog', 'person'])
imgIds = coco.getImgIds(catIds=catIds) #catIds=catIds

images = coco.loadImgs(imgIds)
annIds = coco.getAnnIds(imgIds)
annotations = coco.loadAnns(annIds)

print(imgIds[0])
print(images[0])
# print(categories)
# print(len(categories))

# print(images[0])
# print("\n")
# print(annotations[0])
# print("\n", len(images))





# for i in range(5):
#     print(annotations[i]['image_id'])
#     print(len(annotations[i]['bbox']))
#     print(len(annotations[i]["segmentation"][0]))
#     print('\n')
# img = io.imread(dataDir + "/images/" + dataType + "/" + images[0]["file_name"])
# print(img.shape)



