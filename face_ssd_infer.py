import torch
import torchvision
import torch.nn as nn
# from data.config import TestBaseTransform, widerface_640 as cfg
from data import TestBaseTransform, widerface_640 as cfg
# from layers import Detect, get_prior_boxes, FEM, pa_multibox, mio_module, upsample_product
from layers import *
# from utils import resize_image
import torch.nn.functional as F


class DeepHeadModule(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(DeepHeadModule, self).__init__()
        self._input_channels = input_channels
        self._output_channels = output_channels
        self._mid_channels = min(self._input_channels, 256)

        self.conv1 = nn.Conv2d(self._input_channels, self._mid_channels, kernel_size=3, dilation=1, stride=1, padding=1)
        self.conv2 = nn.Conv2d(self._mid_channels, self._mid_channels, kernel_size=3, dilation=1, stride=1, padding=1)
        self.conv3 = nn.Conv2d(self._mid_channels, self._mid_channels, kernel_size=3, dilation=1, stride=1, padding=1)
        self.conv4 = nn.Conv2d(self._mid_channels, self._output_channels, kernel_size=1, dilation=1, stride=1,
                               padding=0)

    def forward(self, x):
        return self.conv4(
            F.relu(self.conv3(F.relu(self.conv2(F.relu(self.conv1(x), inplace=True)), inplace=True)), inplace=True))


class FEM(nn.Module):
    def __init__(self, channel_size):
        super(FEM, self).__init__()
        self.cs = channel_size
        self.cpm1 = nn.Conv2d(self.cs, 256, kernel_size=3, dilation=1, stride=1, padding=1)
        self.cpm2 = nn.Conv2d(self.cs, 256, kernel_size=3, dilation=2, stride=1, padding=2)
        self.cpm3 = nn.Conv2d(256, 128, kernel_size=3, dilation=1, stride=1, padding=1)
        self.cpm4 = nn.Conv2d(256, 128, kernel_size=3, dilation=2, stride=1, padding=2)
        self.cpm5 = nn.Conv2d(128, 128, kernel_size=3, dilation=1, stride=1, padding=1)

    def forward(self, x):
        x1_1 = F.relu(self.cpm1(x), inplace=True)
        x1_2 = F.relu(self.cpm2(x), inplace=True)
        x2_1 = F.relu(self.cpm3(x1_2), inplace=True)
        x2_2 = F.relu(self.cpm4(x1_2), inplace=True)
        x3_1 = F.relu(self.cpm5(x2_2), inplace=True)
        return torch.cat((x1_1, x2_1, x3_1), 1)


def upsample_product(x, y):
    '''Upsample and add two feature maps.
       Args:
         x: (Variable) top feature map to be upsampled.
         y: (Variable) lateral feature map.
       Returns:
         (Variable) added feature map.
       Note in PyTorch, when input size is odd, the upsampled feature map
       with `F.upsample(..., scale_factor=2, mode='nearest')`
       maybe not equal to the lateral feature map size.
       e.g.
       original input size: [N,_,15,15] ->
       conv2d feature map size: [N,_,8,8] ->
       upsampled feature map size: [N,_,16,16]
       So we choose bilinear upsample which supports arbitrary output sizes.
       '''
    _, _, H, W = y.size()

    # FOR ONNX CONVERSION
    # return F.interpolate(x, scale_factor=2, mode='nearest') * y
    return F.interpolate(x, size=(int(H), int(W)), mode='bilinear', align_corners=False) * y


def pa_multibox(output_channels):
    loc_layers = []
    conf_layers = []
    for k, v in enumerate(output_channels):
        if k == 0:
            loc_output = 4
            conf_output = 2
        elif k == 1:
            loc_output = 8
            conf_output = 4
        else:
            loc_output = 12
            conf_output = 6
        loc_layers += [DeepHeadModule(512, loc_output)]
        conf_layers += [DeepHeadModule(512, (2 + conf_output))]
    return (loc_layers, conf_layers)


def mio_module(each_mmbox, len_conf, your_mind_state='peasant'):
    # chunk = torch.split(each_mmbox, 1, 1) - !!!!! failed to export on PyTorch v1.0.1 (ONNX version 1.3)
    chunk = torch.chunk(each_mmbox, int(each_mmbox.shape[1]), 1)

    # some hacks for ONNX and Inference Engine export
    if your_mind_state == 'peasant':
        bmax = torch.max(torch.max(chunk[0], chunk[1]), chunk[2])
    elif your_mind_state == 'advanced':
        bmax = torch.max(each_mmbox[:, :3], 1)[0].unsqueeze(0)
    else: # supermind
        bmax = torch.nn.functional.max_pool3d(each_mmbox[:, :3], kernel_size=(3, 1, 1))

    cls = (torch.cat((bmax, chunk[3]), dim=1) if len_conf == 0 else torch.cat((chunk[3], bmax), dim=1))
    cls = torch.cat((cls, *list(chunk[4:])), dim=1)
    return cls


class SSD(nn.Module):

    def __init__(self, phase, nms_thresh=0.3, nms_conf_thresh=0.01):
        super(SSD, self).__init__()
        self.phase = phase
        self.num_classes = 2
        self.cfg = cfg

        resnet = torchvision.models.resnet152(pretrained=True)

        self.layer1 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool, resnet.layer1)
        self.layer2 = nn.Sequential(resnet.layer2)
        self.layer3 = nn.Sequential(resnet.layer3)
        self.layer4 = nn.Sequential(resnet.layer4)
        self.layer5 = nn.Sequential(
            *[nn.Conv2d(2048, 512, kernel_size=1),
              nn.BatchNorm2d(512),
              nn.ReLU(inplace=True),
              nn.Conv2d(512, 512, kernel_size=3, padding=1, stride=2),
              nn.BatchNorm2d(512),
              nn.ReLU(inplace=True)]
        )
        self.layer6 = nn.Sequential(
            *[nn.Conv2d(512, 128, kernel_size=1, ),
              nn.BatchNorm2d(128),
              nn.ReLU(inplace=True),
              nn.Conv2d(128, 256, kernel_size=3, padding=1, stride=2),
              nn.BatchNorm2d(256),
              nn.ReLU(inplace=True)]
        )

        output_channels = [256, 512, 1024, 2048, 512, 256]

        # FPN
        fpn_in = output_channels

        self.latlayer3 = nn.Conv2d(fpn_in[3], fpn_in[2], kernel_size=1, stride=1, padding=0)
        self.latlayer2 = nn.Conv2d(fpn_in[2], fpn_in[1], kernel_size=1, stride=1, padding=0)
        self.latlayer1 = nn.Conv2d(fpn_in[1], fpn_in[0], kernel_size=1, stride=1, padding=0)

        self.smooth3 = nn.Conv2d(fpn_in[2], fpn_in[2], kernel_size=1, stride=1, padding=0)
        self.smooth2 = nn.Conv2d(fpn_in[1], fpn_in[1], kernel_size=1, stride=1, padding=0)
        self.smooth1 = nn.Conv2d(fpn_in[0], fpn_in[0], kernel_size=1, stride=1, padding=0)

        # FEM
        cpm_in = output_channels

        self.cpm3_3 = FEM(cpm_in[0])
        self.cpm4_3 = FEM(cpm_in[1])
        self.cpm5_3 = FEM(cpm_in[2])
        self.cpm7 = FEM(cpm_in[3])
        self.cpm6_2 = FEM(cpm_in[4])
        self.cpm7_2 = FEM(cpm_in[5])

        # head
        head = pa_multibox(output_channels)
        self.loc = nn.ModuleList(head[0])
        self.conf = nn.ModuleList(head[1])

        self.softmax = nn.Softmax(dim=-1)

        if self.phase != 'onnx_export':
            self.detect = Detect(self.num_classes, 0, cfg['num_thresh'], nms_conf_thresh, nms_thresh,
                                 cfg['variance'])
            self.last_image_size = None
            self.last_feature_maps = None

        if self.phase == 'test':
            self.test_transform = TestBaseTransform((104, 117, 123))

    def forward(self, x):

        image_size = [x.shape[2], x.shape[3]]
        loc = list()
        conf = list()

        conv3_3_x = self.layer1(x)
        conv4_3_x = self.layer2(conv3_3_x)
        conv5_3_x = self.layer3(conv4_3_x)
        fc7_x = self.layer4(conv5_3_x)
        conv6_2_x = self.layer5(fc7_x)
        conv7_2_x = self.layer6(conv6_2_x)

        lfpn3 = upsample_product(self.latlayer3(fc7_x), self.smooth3(conv5_3_x))
        lfpn2 = upsample_product(self.latlayer2(lfpn3), self.smooth2(conv4_3_x))
        lfpn1 = upsample_product(self.latlayer1(lfpn2), self.smooth1(conv3_3_x))

        conv5_3_x = lfpn3
        conv4_3_x = lfpn2
        conv3_3_x = lfpn1

        sources = [conv3_3_x, conv4_3_x, conv5_3_x, fc7_x, conv6_2_x, conv7_2_x]

        sources[0] = self.cpm3_3(sources[0])
        sources[1] = self.cpm4_3(sources[1])
        sources[2] = self.cpm5_3(sources[2])
        sources[3] = self.cpm7(sources[3])
        sources[4] = self.cpm6_2(sources[4])
        sources[5] = self.cpm7_2(sources[5])

        # apply multibox head to source layers
        featuremap_size = []
        for (x, l, c) in zip(sources, self.loc, self.conf):
            featuremap_size.append([x.shape[2], x.shape[3]])
            loc.append(l(x).permute(0, 2, 3, 1).contiguous())
            len_conf = len(conf)
            cls = mio_module(c(x), len_conf)
            conf.append(cls.permute(0, 2, 3, 1).contiguous())

        face_loc = torch.cat([o[:, :, :, :4].contiguous().view(o.size(0), -1) for o in loc], 1)
        face_loc = face_loc.view(face_loc.size(0), -1, 4)
        face_conf = torch.cat([o[:, :, :, :2].contiguous().view(o.size(0), -1) for o in conf], 1)
        face_conf = self.softmax(face_conf.view(face_conf.size(0), -1, self.num_classes))

        if self.phase != 'onnx_export':

            if self.last_image_size is None or self.last_image_size != image_size or self.last_feature_maps != featuremap_size:
                self.priors = get_prior_boxes(self.cfg, featuremap_size, image_size).to(face_loc.device)
                self.last_image_size = image_size
                self.last_feature_maps = featuremap_size
            with torch.no_grad():
                output = self.detect(face_loc, face_conf, self.priors)
        else:
            output = torch.cat((face_loc, face_conf), 2)
        return output

    def detect_on_image(self, source_image, target_size, device, is_pad=False, keep_thresh=0.3):

        image, shift_h_scaled, shift_w_scaled, scale = resize_image(source_image, target_size, is_pad=is_pad)

        x = torch.from_numpy(self.test_transform(image)).permute(2, 0, 1).to(device)
        x.unsqueeze_(0)

        detections = self.forward(x).cpu().numpy()

        scores = detections[0, 1, :, 0]
        keep_idxs = scores > keep_thresh  # find keeping indexes
        detections = detections[0, 1, keep_idxs, :]  # select detections over threshold
        detections = detections[:, [1, 2, 3, 4, 0]]  # reorder

        detections[:, [0, 2]] -= shift_w_scaled  # 0 or pad percent from left corner
        detections[:, [1, 3]] -= shift_h_scaled  # 0 or pad percent from top
        detections[:, :4] *= scale

        return detections