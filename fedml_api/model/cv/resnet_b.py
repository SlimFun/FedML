'''ResNet in PyTorch.

For Pre-activation ResNet, see 'preact_resnet.py'.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion *
                               planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


# class ResNet(nn.Module):
#     def __init__(self, block, num_blocks, num_classes=10):
#         super(ResNet, self).__init__()
#         self.in_planes = 64

#         self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
#                                stride=1, padding=1, bias=False)
#         self.bn1 = nn.BatchNorm2d(64)
#         self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
#         self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
#         self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
#         self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
#         self.linear = nn.Linear(512*block.expansion, num_classes)

#     def _make_layer(self, block, planes, num_blocks, stride):
#         strides = [stride] + [1]*(num_blocks-1)
#         layers = []
#         for stride in strides:
#             layers.append(block(self.in_planes, planes, stride))
#             self.in_planes = planes * block.expansion
#         return nn.Sequential(*layers)

#     def forward(self, x):
#         out = F.relu(self.bn1(self.conv1(x)))
#         out = self.layer1(out)
#         out = self.layer2(out)
#         out = self.layer3(out)
#         out = self.layer4(out)
#         out = F.avg_pool2d(out, 4)
#         out = out.view(out.size(0), -1)
#         out = self.linear(out)
#         return out

class CovaResNet(nn.Module):
    def __init__(self, block, num_blocks, with_cova=False, neck='no', num_classes=10):
        super(CovaResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        
        #Cova
#         self.num_features = num_features
        self.neck = neck
#         self.neck_feat = neck_feat
        self.num_classes = num_classes
        self.with_cova = with_cova
        self.in_planes = 512*block.expansion
        if self.neck == 'no':
            self.ce_classifier = nn.Linear(self.in_planes, self.num_classes)
        elif self.neck == 'bnneck':
            self.bottleneck = nn.BatchNorm1d(self.in_planes)
            self.bottleneck.bias.requires_grad_(False)  # no shift
            self.ce_classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)

            self.bottleneck.apply(weights_init_kaiming)
            self.ce_classifier.apply(weights_init_classifier)
            
        self.covariance = CovaBlock()
        
#         self.cova_classifier = nn.Sequential(
#             nn.LeakyReLU(0.2, True),
#             nn.Dropout(),
#             nn.Conv1d(1, 1, kernel_size=441, stride=441, bias=True),
#         )
#         self.cova_classifier.apply(weights_init_kaiming)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, support_covas=None, ml=None):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        base_feat = self.layer4(out)
        global_feat = F.avg_pool2d(base_feat, 4)
        
        global_feat = global_feat.view(global_feat.size(0), -1)
        
        if self.neck == 'no':
            feat = global_feat
        elif self.neck == 'bnneck':
            feat = self.bottleneck(global_feat)  # normalize for angular softmax
            
        cls_score = self.ce_classifier(feat)
        if (not self.with_cova):
            return cls_score, None, base_feat
        
        cova_score = self.covariance(base_feat, support_covas, ml)
        cova_score = cova_score.squeeze(1)
        return cls_score, cova_score, base_feat
    
class CovaBlock(nn.Module):
    def __init__(self):
        super(CovaBlock, self).__init__()
        
    def cal_similarity(self, input, CovaMatrix_list, ml):
        B, C, h, w = input.size()
        Cova_Sim = []

        for i in range(B):
            
            query_sam = input[i]
            
            query_sam = query_sam.view(C, -1)
            mean_query = torch.mean(query_sam, 1, True)
            
#             query_sam = query_sam - mean_query
            
            query_sam_norm = torch.norm(query_sam, 2, 1, True) 
#             query_sam_norm = torch.clamp(query_sam_norm, min=1e-6)
            query_sam = query_sam/query_sam_norm

            if torch.cuda.is_available():
                mea_sim = torch.zeros(1, len(CovaMatrix_list)).to('cuda:3')

            for j in range(len(CovaMatrix_list)):
                temp_dis = torch.transpose(query_sam, 0, 1)@CovaMatrix_list[j]@query_sam
                diag = temp_dis.diag()
                d = diag.sum() / len(diag)
                mea_sim[0, j] = d

            Cova_Sim.append(mea_sim.unsqueeze(0))

        Cova_Sim = torch.cat(Cova_Sim, 0) # get Batch*1*(h*w*num_classes)
        return Cova_Sim 
    
    def forward(self, input, support_covas, ml):
        return self.cal_similarity(input, support_covas, ml)

    
def ResNet18():
    return CovaResNet(BasicBlock, [2, 2, 2, 2], with_cova=False)

def CovaResNet18():
    return CovaResNet(BasicBlock, [2, 2, 2, 2], with_cova=True)


# def ResNet34():
#     return ResNet(BasicBlock, [3, 4, 6, 3])


def ResNet50():
    return ResNet(Bottleneck, [3, 4, 6, 3], with_cova=False)


# def ResNet101():
#     return ResNet(Bottleneck, [3, 4, 23, 3])


# def ResNet152():
#     return ResNet(Bottleneck, [3, 8, 36, 3])


def test():
    net = ResNet18()
    y = net(torch.randn(1, 3, 32, 32))
    print(y.size())

# test()
