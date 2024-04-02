from PIL import Image
import torch
from MTCNN_nets import PNet, RNet, ONet
import math 
import numpy as np
from utils.util import*
import cv2
import matplotlib.pyplot as plt
def load_networks(device):
    pnet = PNet().to(device)
    rnet = RNet().to(device)
    onet = ONet().to(device)
    pnet.load_state_dict(torch.load('MTCNN/weights/pnet_Weights', map_location=device))
    rnet.load_state_dict(torch.load('MTCNN/weights/rnet_Weights', map_location=device))
    onet.load_state_dict(torch.load('MTCNN/weights/Onet_Weights', map_location=device))
    pnet.eval()
    rnet.eval()
    onet.eval()
    return pnet, rnet, onet
def preprocess(img):
    img = img[:,:,::-1]
    img = np.asarray(img, 'float32')
    img = img.transpose((2, 0, 1))
    img = np.expand_dims(img, 0)
    img = (img - 127.5)*0.0078125
    return img

def nms(boxes, overlap_threshold=0.5, mode='union'):
    
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

def calibrate_box(bboxes, offsets):
    x1, y1, x2, y2 = [bboxes[:, i] for i in range(4)]
    w = x2 - x1 + 1.0
    h = y2 - y1 + 1.0
    w = np.expand_dims(w, 1)
    h = np.expand_dims(h, 1)

    translation = np.hstack([w, h, w, h])*offsets
    bboxes[:, 0:4] = bboxes[:, 0:4] + translation
    return bboxes

def convert_to_square(bboxes):
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

def correct_bboxes(bboxes, width, height):
    
    x1, y1, x2, y2 = [bboxes[:, i] for i in range(4)]
    w, h = x2 - x1 + 1.0,  y2 - y1 + 1.0
    num_boxes = bboxes.shape[0]

    # 'e' stands for end
    # (x, y) -> (ex, ey)
    x, y, ex, ey = x1, y1, x2, y2

    # we need to cut out a box from the image.
    # (x, y, ex, ey) are corrected coordinates of the box
    # in the image.
    # (dx, dy, edx, edy) are coordinates of the box in the cutout
    # from the image.
    dx, dy = np.zeros((num_boxes,)), np.zeros((num_boxes,))
    edx, edy = w.copy() - 1.0, h.copy() - 1.0

    # if box's bottom right corner is too far right
    ind = np.where(ex > width - 1.0)[0]
    edx[ind] = w[ind] + width - 2.0 - ex[ind]
    ex[ind] = width - 1.0

    # if box's bottom right corner is too low
    ind = np.where(ey > height - 1.0)[0]
    edy[ind] = h[ind] + height - 2.0 - ey[ind]
    ey[ind] = height - 1.0

    # if box's top left corner is too far left
    ind = np.where(x < 0.0)[0]
    dx[ind] = 0.0 - x[ind]
    x[ind] = 0.0

    # if box's top left corner is too high
    ind = np.where(y < 0.0)[0]
    dy[ind] = 0.0 - y[ind]
    y[ind] = 0.0

    return_list = [dy, edy, dx, edx, y, ey, x, ex, w, h]
    return_list = [i.astype('int32') for i in return_list]

    return return_list
def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    pnet, rnet, onet = load_networks(device)
    image = cv2.imread('MTCNN\images\office1.jpg')
    min_face_size = 20.0
    thresholds = [0.6, 0.7, 0.8]
    nms_thresholds = [0.7, 0.7, 0.7]
    height, width, channel = image.shape
    min_length = min(height, width)
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
    size = 24
    num_boxes = len(bboxes)
    [dy, edy, dx, edx, y, ey, x, ex, w, h] = correct_bboxes(bboxes, width, height)

    img_boxes = np.zeros((num_boxes, 3, size, size))

    for i in range(num_boxes):
        img_box = np.zeros((h[i], w[i], 3))

        img_box[dy[i]:(edy[i] + 1), dx[i]:(edx[i] + 1), :] =\
            image[y[i]:(ey[i] + 1), x[i]:(ex[i] + 1), :]

        # resize
        img_box = cv2.resize(img_box, (size, size), interpolation=cv2.INTER_LINEAR)
    
        img_boxes[i, :, :, :] = preprocess(img_box)

    img_boxes = torch.FloatTensor(img_boxes).to(device)
    offset, prob = rnet(img_boxes)
    offsets = offset.cpu().data.numpy()  # shape [n_boxes, 4]
    probs = prob.cpu().data.numpy()  # shape [n_boxes, 2]

    keep = np.where(probs[:, 1] > thresholds[1])[0]
    bboxes = bboxes[keep]
    bboxes[:, 4] = probs[keep, 1].reshape((-1,)) # assign score from stage 2
    offsets = offsets[keep] #

    keep = nms(bboxes, nms_thresholds[1])
    bboxes = bboxes[keep]
    bboxes = calibrate_box(bboxes, offsets[keep])
    bboxes = convert_to_square(bboxes)
    bboxes[:, 0:4] = np.round(bboxes[:, 0:4])
    #3단계
    size = 48
    num_boxes = len(bboxes)
    [dy, edy, dx, edx, y, ey, x, ex, w, h] = correct_bboxes(bboxes, width, height)

    img_boxes = np.zeros((num_boxes, 3, size, size))

    for i in range(num_boxes):
        img_box = np.zeros((h[i], w[i], 3))

        img_box[dy[i]:(edy[i] + 1), dx[i]:(edx[i] + 1), :] =\
            image[y[i]:(ey[i] + 1), x[i]:(ex[i] + 1), :]

        # resize
        img_box = cv2.resize(img_box, (size, size), interpolation=cv2.INTER_LINEAR)

        img_boxes[i, :, :, :] = preprocess(img_box)

    img_boxes = torch.FloatTensor(img_boxes).to(device)
    landmark, offset, prob = onet(img_boxes)
    landmarks = landmark.cpu().data.numpy()  # shape [n_boxes, 10]
    offsets = offset.cpu().data.numpy()  # shape [n_boxes, 4]
    probs = prob.cpu().data.numpy()  # shape [n_boxes, 2]

    keep = np.where(probs[:, 1] > thresholds[2])[0]
    bboxes = bboxes[keep]
    bboxes[:, 4] = probs[keep, 1].reshape((-1,)) # assign score from stage 2
    offsets = offsets[keep] 
    landmarks = landmarks[keep]

    # compute landmark points
    width = bboxes[:, 2] - bboxes[:, 0] + 1.0
    height = bboxes[:, 3] - bboxes[:, 1] + 1.0
    xmin, ymin = bboxes[:, 0], bboxes[:, 1]
    landmarks[:, 0:5] = np.expand_dims(xmin, 1) + np.expand_dims(width, 1) * landmarks[:, 0:5]
    landmarks[:, 5:10] = np.expand_dims(ymin, 1) + np.expand_dims(height, 1) * landmarks[:, 5:10]

    bboxes = calibrate_box(bboxes, offsets)
    keep = nms(bboxes, nms_thresholds[2], mode='min')
    bboxes = bboxes[keep]
    landmarks = landmarks[keep]

    #3단계 후 시각화
    cv2_img = cv2.imread('MTCNN\images\office1.jpg')

    # OpenCV 이미지를 NumPy 배열로 변환
    img_rgb = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(15.0, 12.0))  # 플롯 크기 설정
    plt.rcParams['image.interpolation'] = 'nearest'  # 이미지 보간 방식 설정
    plt.rcParams['image.cmap'] = 'gray'  # 컬러맵 설정 (grayscale)

    # 이미지에 바운딩 박스 그리기
    for i in range(bboxes.shape[0]):
        bbox = bboxes[i, :4]
        cv2.rectangle(img_rgb, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 0, 255), 2)

    # 이미지에 랜드마크 그리기
    if 'landmarks' in locals():
        for i in range(landmarks.shape[0]):
            landmarks_one = landmarks[i, :]
            landmarks_one = landmarks_one.reshape(2, 5).T
            for j in range(5):
                cv2.circle(img_rgb, (int(landmarks_one[j, 0]), int(landmarks_one[j, 1])), 2, (0, 255, 0), 1)

    plt.imshow(img_rgb)  # RGB 이미지 출력
    plt.show()
    
if __name__ == '__main__':
    main()