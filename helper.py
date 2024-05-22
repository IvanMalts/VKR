from torchvision import transforms as T
import torch

model = torch.load('model.pth', map_location=torch.device('cpu'))
model.eval()

def get_iob(box1, box2, model):
    if model == False:
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2
        y11, x11, y21, x21 = y1, x1, y1 + h1, x1 + w1
        y12, x12, y22, x22 = y2, x2, y2 + h2, x2 + w2
    else:
        x1, y1, w1, h1 = box1
        x12, y12, x22, y22 = box2
        y11, x11, y21, x21 = y1, x1, y1 + h1, x1 + w1


    yi1 = max(y11, y12)
    xi1 = max(x11, x12)
    yi2 = min(y21, y22)
    xi2 = min(x21, x22)
    inter_area = max(((xi2 - xi1) * (yi2 - yi1)), 0)
    box1_area = (x21 - x11) * (y21 - y11)
    iou = inter_area / box1_area
    return iou

def inside(box1, box2):
    x1_min, y1_min, w1, h1 = box1
    x2_min, y2_min, x2_max, y2_max = box2
    

def get_center(box, model):
    if model:
        x1, y1, x2, y2 = box
        x_c = (x2 - x1)/2
        y_c = (y2 - y1)/2
    else:
        x, y, w, h = box
        x_c = x + w/2
        y_c = y + h/2
    return x_c, y_c

def apply_model(frame):
    md_boxes = []
    img_tensor = T.ToTensor()(frame)
    output = model([img_tensor.to(torch.device('cpu'))])
    for i in range(len(output[0]['boxes'])):
        if output[0]['scores'][i] > 0.7:
            pred_box = output[0]['boxes'][i].detach().numpy()
            md_boxes.append([int(pred_box[0]), int(pred_box[1]), int(pred_box[2]), int(pred_box[3]), int(output[0]['labels'][i])])
    return md_boxes