from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F


class EdgeSaliencyLoss(nn.Module):
    def __init__(self, device, alpha_sal=0.7):
        super(EdgeSaliencyLoss, self).__init__()

        self.alpha_sal = alpha_sal

        self.laplacian_kernel = torch.tensor([[[-1., -1., -1.], [-1., 8., -1.], [-1., -1., -1.]],
                                             [[-1., -1., -1.], [-1., 8., -1.], [-1., -1., -1.]],
                                             [[-1., -1., -1.], [-1., 8., -1.], [-1., -1., -1.]],
                                             [[-1., -1., -1.], [-1., 8., -1.], [-1., -1., -1.]],
                                             [[-1., -1., -1.], [-1., 8., -1.], [-1., -1., -1.]],
                                             [[-1., -1., -1.], [-1., 8., -1.], [-1., -1., -1.]],
                                             [[-1., -1., -1.], [-1., 8., -1.], [-1., -1., -1.]],
                                             [[-1., -1., -1.], [-1., 8., -1.], [-1., -1., -1.]],
                                             [[-1., -1., -1.], [-1., 8., -1.], [-1., -1., -1.]]],
                                             dtype=torch.float, requires_grad=False)
        self.laplacian_kernel = self.laplacian_kernel.view((1, 9, 3, 3))  # Shape format of weight for convolution  1028
        self.laplacian_kernel = self.laplacian_kernel.to(device)

    @staticmethod
    def weighted_bce(input_, target, weight_0=1.0, weight_1=1.0, eps=1e-15):
        wbce_loss = -weight_1 * target * torch.log(input_ + eps) - weight_0 * (1 - target) * torch.log(
            1 - input_ + eps)
        return torch.mean(wbce_loss)

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(9):
            temp_prob = input_tensor == i  # * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def forward(self, y_pred, y_gt):
        # Generate edge maps
        # y_pred = y_pred.unsqueeze(1)
        # y_gt = y_gt.unsqueeze(1)
        # y_pred = torch.argmax(torch.softmax(y_pred, dim=1), dim=1).squeeze(0)
        # print(y_gt)
        y_gt = self._one_hot_encoder(y_gt)
        # print(y_gt)
        y_gt_edges = F.relu(torch.tanh(F.conv2d(y_gt, self.laplacian_kernel, padding=(1, 1))))    #1028
        y_pred_edges = F.relu(torch.tanh(F.conv2d(y_pred, self.laplacian_kernel, padding=(1, 1))))

        # sal_loss = F.binary_cross_entropy(input=y_pred, target=y_gt)
        # sal_loss = self.weighted_bce(input_=y_pred, target=y_gt, weight_0=1.0, weight_1=1.12)
        edge_loss = F.binary_cross_entropy(input=y_pred_edges, target=y_gt_edges)

        total_loss = edge_loss
        return total_loss/9


if __name__ == '__main__':
    if torch.cuda.is_available():
        device = torch.device(device='cuda')
    else:
        device = torch.device(device='cpu')

    dummy_input = torch.autograd.Variable(torch.sigmoid(torch.randn(2, 1, 8, 16)), requires_grad=True).to(device)
    dummy_gt = torch.autograd.Variable(torch.ones_like(dummy_input)).to(device)
    print('Input Size :', dummy_input.size())

    criteria = EdgeSaliencyLoss(device=device)
    loss = criteria(dummy_input, dummy_gt)
    print('Loss Value :', loss)
