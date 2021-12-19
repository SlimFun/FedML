import torch
from torch import nn
from torch.autograd import Variable


class CovaMLoss(nn.Module):
    def __init__(self):
        super(CovaMLoss, self).__init__()
#         self.ranking_loss = nn.MarginRankingLoss(margin=margin)

    # calculate the similarity  
    def cal_similarity(self, input, CovaMatrix_list):

        B, C, h, w = input.size()
        Cova_Sim = []

        for i in range(B):
            query_sam = input[i]
            query_sam = query_sam.view(C, -1)
            query_sam_norm = torch.norm(query_sam, 2, 1, True)    
            query_sam = query_sam/query_sam_norm

            if torch.cuda.is_available():
                mea_sim = torch.zeros(1, len(CovaMatrix_list)*h*w).cuda()

            for j in range(len(CovaMatrix_list)):
                temp_dis = torch.transpose(query_sam, 0, 1)@CovaMatrix_list[j]@query_sam
                mea_sim[0, j*h*w:(j+1)*h*w] = temp_dis.diag()

            Cova_Sim.append(mea_sim.unsqueeze(0))

        Cova_Sim = torch.cat(Cova_Sim, 0) # get Batch*1*(h*w*num_classes)
        return Cova_Sim 

    def forward(self, input, support_covas):
        cova_sim = self.cal_similarity(input, support_covas)
        
        return self.cal_similarity(input, support_covas)