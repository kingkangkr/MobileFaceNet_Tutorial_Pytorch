from PIL import Image
import torch
from MTCNN_nets import PNet, RNet, ONet
import math 
import numpy as np
from utils.util import*
import cv2
import matplotlib.pyplot as plt
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
pnet = PNet().to(device)
rnet = RNet().to(device)
onet = ONet().to(device)

pnet.load_state_dict(torch.load('MTCNN/weights/pnet_Weights', map_location=lambda storage, loc: storage))
rnet.load_state_dict(torch.load(r'MTCNN\weights\rnet_Weights', map_location=lambda storage, loc: storage))
onet.load_state_dict(torch.load('MTCNN\weights\Onet_Weights', map_location=lambda storage, loc: storage))

pnet.eval()
rnet.eval()
onet.eval()

image = cv2.imread('MTCNN\images\office1.jpg')
min_face_size = 20.0
thresholds = [0.6, 0.7, 0.8]  # face detection thresholds
nms_thresholds = [0.7, 0.7, 0.7] # nms threshold

height, width, channel = image.shape
min_length = min(height, width)
#이미지 형태 바꾸기
def preprocess(img):
    img = img[:,:,::-1]
    img = np.asarray(img, 'float32')
    img = img.transpose((2, 0, 1))
    img = np.expand_dims(img, 0)
    img = (img - 127.5)*0.0078125
    return img
#1단계(pnet을 이용한 얼굴 탐지)
def nms(boxes, overlap_threshold=0.5, mode='union'):
    """Non-maximum suppression.

    Arguments:
        boxes: a float numpy array of shape [n, 5],
            where each row is (xmin, ymin, xmax, ymax, score).
        overlap_threshold: a float number.
        mode: 'union' or 'min'.

    Returns:
        list with indices of the selected boxes
    """

    # if there are no boxes, return the empty list
    if len(boxes) == 0:
        return []

    # list of picked indices
    pick = []

    # grab the coordinates of the bounding boxes
    x1, y1, x2, y2, score = [boxes[:, i] for i in range(5)]

    area = (x2 - x1 + 1.0)*(y2 - y1 + 1.0)
    ids = np.argsort(score)  # in increasing order

    while len(ids) > 0:

        # grab index of the largest value
        last = len(ids) - 1
        i = ids[last]
        pick.append(i)

        # compute intersections of the box with the largest score with the rest of boxes

        # left top corner of intersection boxes
        ix1 = np.maximum(x1[i], x1[ids[:last]])
        iy1 = np.maximum(y1[i], y1[ids[:last]])

        # right bottom corner of intersection boxes
        ix2 = np.minimum(x2[i], x2[ids[:last]])
        iy2 = np.minimum(y2[i], y2[ids[:last]])

        # width and height of intersection boxes
        w = np.maximum(0.0, ix2 - ix1 + 1.0)
        h = np.maximum(0.0, iy2 - iy1 + 1.0)

        # intersections' areas
        inter = w * h
        if mode == 'min':
            overlap = inter/np.minimum(area[i], area[ids[:last]])
        elif mode == 'union':
            # intersection over union (IoU)
            overlap = inter/(area[i] + area[ids[:last]] - inter)

        # delete all boxes where overlap is too big
        ids = np.delete(
            ids,
            np.concatenate([[last], np.where(overlap > overlap_threshold)[0]])
        )

    return pick
min_detection_size = 12
factor = 0.707  # sqrt(0.5)

# scales for scaling the image
scales = []

# scales the image so that minimum size that we can detect equals to minimum face size that we want to detect
m = min_detection_size / min_face_size
min_length *= m

factor_count = 0
while min_length > min_detection_size:
    scales.append(m * factor ** factor_count)
    min_length *= factor
    factor_count += 1

# it will be returned
bounding_boxes = []

with torch.no_grad():
    #run P-Net on different scales
    for scale in scales:
        sw, sh = math.ceil(width*scale), math.ceil(height*scale)
        img = cv2.resize(image, (sw, sh), interpolation=cv2.INTER_LINEAR)
        img = torch.FloatTensor(preprocess(img)).to(device)
        offset, prob = pnet(img)
        probs = prob.cpu().data.numpy()[0, 1, :, :] # probs: probability of a face at each sliding window
        offsets = offset.cpu().data.numpy()  # offsets: transformations to true bounding boxes
        # applying P-Net is equivalent, in some sense, to moving 12x12 window with stride 2
        stride, cell_size = 2, 12
        # indices of boxes where there is probably a face
        # returns a tuple with an array of row idx's, and an array of col idx's:
        inds = np.where(probs > thresholds[0])
        
        if inds[0].size == 0:
            boxes = None
        else:
            # transformations of bounding boxes
            tx1, ty1, tx2, ty2 = [offsets[0, i, inds[0], inds[1]] for i in range(4)]
            offsets = np.array([tx1, ty1, tx2, ty2])
            score = probs[inds[0], inds[1]]
            # P-Net is applied to scaled images
            # so we need to rescale bounding boxes back
            bounding_box = np.vstack([
            np.round((stride*inds[1] + 1.0)/scale),
            np.round((stride*inds[0] + 1.0)/scale),
            np.round((stride*inds[1] + 1.0 + cell_size)/scale),
            np.round((stride*inds[0] + 1.0 + cell_size)/scale),
            score, offsets])
            boxes = bounding_box.T
            keep = nms(boxes[:, 0:5], overlap_threshold=0.5)
            boxes[keep]
            
        bounding_boxes.append(boxes)
        
#박스함수
def calibrate_box(bboxes, offsets):
    
    """Transform bounding boxes to be more like true bounding boxes.
    'offsets' is one of the outputs of the nets.

    Arguments:
        bboxes: a float numpy array of shape [n, 5].
        offsets: a float numpy array of shape [n, 4].

    Returns:
        a float numpy array of shape [n, 5].
    """

    x1, y1, x2, y2 = [bboxes[:, i] for i in range(4)]
    w = x2 - x1 + 1.0
    h = y2 - y1 + 1.0
    w = np.expand_dims(w, 1)
    h = np.expand_dims(h, 1)

    translation = np.hstack([w, h, w, h])*offsets
    bboxes[:, 0:4] = bboxes[:, 0:4] + translation
    return bboxes
def convert_to_square(bboxes):
    
    """Convert bounding boxes to a square form.

    Arguments:
        bboxes: a float numpy array of shape [n, 5].

    Returns:
        a float numpy array of shape [n, 5],
            squared bounding boxes.
    """

    square_bboxes = np.zeros_like(bboxes)
    x1, y1, x2, y2 = [bboxes[:, i] for i in range(4)]
    h = y2 - y1 + 1.0
    w = x2 - x1 + 1.0
    max_side = np.maximum(h, w)
    square_bboxes[:, 0] = x1 + w*0.5 - max_side*0.5
    square_bboxes[:, 1] = y1 + h*0.5 - max_side*0.5
    square_bboxes[:, 2] = square_bboxes[:, 0] + max_side - 1.0
    square_bboxes[:, 3] = square_bboxes[:, 1] + max_side - 1.0
    return square_bboxes
# collect boxes (and offsets, and scores) from different scales
bounding_boxes = [i for i in bounding_boxes if i is not None]
bounding_boxes = np.vstack(bounding_boxes)

keep = nms(bounding_boxes[:, 0:5], nms_thresholds[0])
bounding_boxes = bounding_boxes[keep]

# use offsets predicted by pnet to transform bounding boxes
bboxes = calibrate_box(bounding_boxes[:, 0:5], bounding_boxes[:, 5:])
# shape [n_boxes, 5],  x1, y1, x2, y2, score

bboxes = convert_to_square(bboxes)
bboxes[:, 0:4] = np.round(bboxes[:, 0:4])
print(bboxes.shape)
#pnet을 통한 시각화

import cv2


# OpenCV로 이미지 로드
cv2_img = cv2.imread('MTCNN\images\office1.jpg')

# OpenCV 이미지를 NumPy 배열로 변환
img_rgb = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)

# 이미지에 바운딩 박스 그리기
for i in range(bboxes.shape[0]):
    bbox = bboxes[i, :4]
    cv2.rectangle(img_rgb, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 0, 255), 1)

# matplotlib로 이미지 시각화
plt.figure(figsize=(15, 12))
plt.imshow(img_rgb)

