'''
ResNet for CIFAR-10/100 Dataset.

Reference:
1. https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
2. https://github.com/facebook/fb.resnet.torch/blob/master/models/resnet.lua
3. Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
Deep Residual Learning for Image Recognition. https://arxiv.org/abs/1512.03385

'''
import logging

import torch
import torch.nn as nn

__all__ = ['ResNet', 'resnet110']


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.LeakyReLU(0.2, True)
#         self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
#         self.relu = nn.ReLU(inplace=True)
        self.relu = nn.LeakyReLU(0.2, True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=10, zero_init_residual=False, groups=1,
                 width_per_group=64, replace_stride_with_dilation=None, norm_layer=None, KD=False):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 16
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))

        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.LeakyReLU(0.2, True)
#         self.relu = nn.ReLU(inplace=True)
        # self.maxpool = nn.MaxPool2d()
        self.layer1 = self._make_layer(block, 16, layers[0])
        self.layer2 = self._make_layer(block, 32, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 64, layers[2], stride=2)
      

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x):
        input = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)  # B x 16 x 32 x 32
        lx1 = self.layer1(x)  # B x 16 x 32 x 32
        lx2 = self.layer2(lx1)  # B x 32 x 16 x 16
        lx3 = self.layer3(lx2)  # B x 64 x 8 x 8
        
        return lx3
        
        
class CovaMResnet56(nn.Module):
    in_planes = 256
    
    def __init__(self, class_num, neck, pretrained=False, num_features=0, neck_feat='after', with_cova=True, path=None, **kwargs):
        super(CovaMResnet56, self).__init__()
        self.base = ResNet(Bottleneck, [6, 6, 6], class_num, **kwargs)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.num_features = num_features
        self.neck = neck
        self.neck_feat = neck_feat
        self.num_classes = class_num
        self.with_cova = with_cova

        if self.neck == 'no':
            self.ce_classifier = nn.Linear(self.in_planes, self.num_classes)
        elif self.neck == 'bnneck':
            self.bottleneck = nn.BatchNorm1d(self.in_planes)
            self.bottleneck.bias.requires_grad_(False)  # no shift
            self.ce_classifier = nn.utils.weight_norm(nn.Linear(self.in_planes, self.num_classes, bias=False))

            self.bottleneck.apply(weights_init_kaiming)
            self.ce_classifier.apply(weights_init_classifier)
            

#         if not with_cova:
#             return
        self.covariance = CovaBlock()
        
        self.cova_classifier = nn.Sequential(
            nn.LeakyReLU(0.2, True),
            nn.Dropout(),
            nn.utils.weight_norm(nn.Conv1d(1, 1, kernel_size=64, stride=64, bias=True)),
        )
        self.cova_classifier.apply(weights_init_kaiming)
            
    # return: cls_score, cova_score, cova
    def forward(self, x, support_imgs=None, ml=None):
        output_feat = self.base(x)
        global_feat = self.avgpool(output_feat)  # (b, 256, 1, 1)
        global_feat = global_feat.view(global_feat.shape[0], -1)  #(b, 256)
        
        if self.neck == 'no':
            feat = global_feat
        elif self.neck == 'bnneck':
            feat = self.bottleneck(global_feat)  # normalize for angular softmax
        
        cls_score = self.ce_classifier(feat)
        if (not self.with_cova) or support_imgs == None:
            return cls_score, None, output_feat
        
        S = []
        for i in range(len(support_imgs)):
            S.append(self.base(support_imgs[i]))
        
        cova_score = self.covariance(output_feat, S)
        cova_score = self.cova_classifier(cova_score)
        cova_score = cova_score.squeeze(1)
#         print(cova_score)
        if not self.training:
            return cls_score, cova_score, output_feat
        return cls_score, cova_score, None


class CovaBlock(nn.Module):
    def __init__(self):
        super(CovaBlock, self).__init__()


    # calculate the covariance matrix 
    def cal_covariance(self, input):

        CovaMatrix_list = []
        for i in range(len(input)):
            support_set_sam = input[i]
            B, C, h, w = support_set_sam.size()

            support_set_sam = support_set_sam.permute(1, 0, 2, 3)
            support_set_sam = support_set_sam.contiguous().view(C, -1)
            mean_support = torch.mean(support_set_sam, 1, True)
            support_set_sam = support_set_sam-mean_support

            covariance_matrix = support_set_sam@torch.transpose(support_set_sam, 0, 1)
            covariance_matrix = torch.div(covariance_matrix, h*w*B-1)
            CovaMatrix_list.append(covariance_matrix)

        return CovaMatrix_list    


    # calculate the similarity  
    def cal_similarity(self, input, CovaMatrix_list):

        B, C, h, w = input.size()
        Cova_Sim = []

        for i in range(B):
            query_sam = input[i]
            query_sam = query_sam.view(C, -1)
            
#             mean_query = torch.mean(query_sam, 1, True)
#             query_sam = query_sam-mean_query
            
            query_sam_norm = torch.norm(query_sam, 2, 1, True)   
#             query_sam_norm = torch.clamp(query_sam_norm, min=1e-4)
            query_sam = query_sam/query_sam_norm

            if torch.cuda.is_available():
                mea_sim = torch.zeros(1, len(CovaMatrix_list)*h*w).to('cuda:1')

            for j in range(len(CovaMatrix_list)):
                temp_dis = torch.transpose(query_sam, 0, 1)@CovaMatrix_list[j]@query_sam
                mea_sim[0, j*h*w:(j+1)*h*w] = temp_dis.diag()

            Cova_Sim.append(mea_sim.unsqueeze(0))

        Cova_Sim = torch.cat(Cova_Sim, 0) # get Batch*1*(h*w*num_classes)
        return Cova_Sim 


    # input: q, S; q:(64*21*21)
    def forward(self, x1, x2):
#         print(len(x2))
#         print(x2.shape)
        CovaMatrix_list = self.cal_covariance(x2)
#         print(len(CovaMatrix_list))
#         print(CovaMatrix_list[0].shape)
        Cova_Sim = self.cal_similarity(x1, CovaMatrix_list)
#         print(Cova_Sim.shape)
        return Cova_Sim
        
        
def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)
    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)

    


def resnet110(class_num, pretrained=False, path=None, **kwargs):
    """
    Constructs a ResNet-110 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained.
    """
    logging.info("path = " + str(path))
    model = ResNet(Bottleneck, [12, 12, 12], class_num, **kwargs)
    if pretrained:
        checkpoint = torch.load(path)
        state_dict = checkpoint['state_dict']

        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            # name = k[7:]  # remove 'module.' of dataparallel
            name = k.replace("module.", "")
            new_state_dict[name] = v

        model.load_state_dict(new_state_dict)
    return model



