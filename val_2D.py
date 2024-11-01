import numpy as np
import torch
import torch.nn.functional as F
from medpy import metric
from scipy.ndimage import zoom


def calculate_metric_percase(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    if pred.sum() > 0:
        dice = metric.binary.dc(pred, gt)
        hd95 = metric.binary.hd95(pred, gt)
        iou = metric.binary.jc(pred, gt)

        return dice, hd95, iou
    else:
        return 0, 0, 0


def test_single_volume(image, label, net, classes, patch_size=[256, 256], image_size=[1003, 972]):
    input = F.interpolate(image, size=(224, 224), mode='bicubic', align_corners=False)
    input = input.float().cuda()
    label = label.squeeze(0).cpu().detach().numpy() # (1, 224, 224) / (224, 224)
    
    prediction = np.zeros_like(label) # (224, 224) init

    # image, label from the val dataloader are already zoom/resized to 224x224
    # input = torch.from_numpy(image).unsqueeze(
    #         0).float().cuda() #(1x1x224x224) 
    
    net.eval()
    with torch.no_grad():
        out = torch.softmax(net(input), dim=1)
        # resize
        out = F.interpolate(out, size=(1003, 1003), mode='bilinear', align_corners=False)
        out = torch.argmax(out, dim=1).squeeze() # remove batch size 1 -> 224 224
        out = out.cpu().detach().numpy()
        
        # pred = zoom(out, (image_size[0] / patch_size[0], image_size[1] / patch_size[1]), order=5) # resize back to the original image size
        # label = zoom(label, (image_size[0] / patch_size[0], image_size[1] / patch_size[1]), order=5)
        prediction = out # (1003x972) without batch size
        
    metric_list = []
    for i in range(1, classes):
        metric_list.append(calculate_metric_percase(
            prediction == i, label == i
        ))

    return metric_list

    # from original paper
    for ind in range(image.shape[0]): 
        slice = image[ind, :, :]
        x, y = slice.shape[0], slice.shape[1]
        slice = zoom(slice, (patch_size[0] / x, patch_size[1] / y), order=0)
        input = torch.from_numpy(slice).unsqueeze(
            0).unsqueeze(0).float().cuda()
        net.eval()
        with torch.no_grad():
            out = torch.argmax(torch.softmax(
                net(input), dim=1), dim=1).squeeze(0)
            out = out.cpu().detach().numpy()
            pred = zoom(out, (x / patch_size[0], y / patch_size[1]), order=0)
            prediction[ind] = pred
    metric_list = []
    for i in range(1, classes):
        metric_list.append(calculate_metric_percase(
            prediction == i, label == i))
    return metric_list


def test_single_volume_ds(image, label, net, classes, patch_size=[256, 256]):
    image, label = image.squeeze(0).cpu().detach(
    ).numpy(), label.squeeze(0).cpu().detach().numpy()
    prediction = np.zeros_like(label)
    for ind in range(image.shape[0]):
        slice = image[ind, :, :]
        x, y = slice.shape[0], slice.shape[1]
        slice = zoom(slice, (patch_size[0] / x, patch_size[1] / y), order=0)
        input = torch.from_numpy(slice).unsqueeze(
            0).unsqueeze(0).float().cuda()
        net.eval()
        with torch.no_grad():
            output_main, _, _, _ = net(input)
            out = torch.argmax(torch.softmax(
                output_main, dim=1), dim=1).squeeze(0)
            out = out.cpu().detach().numpy()
            pred = zoom(out, (x / patch_size[0], y / patch_size[1]), order=0)
            prediction[ind] = pred
    metric_list = []
    for i in range(1, classes):
        metric_list.append(calculate_metric_percase(
            prediction == i, label == i))
    return metric_list
