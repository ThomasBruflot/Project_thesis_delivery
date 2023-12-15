import math
import subprocess
import sys

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

install("einops")
install("syops")
install("progress")
install("thop")
install("vprof")
install("torchvision")
install("opencv-python")
install("scikit-learn")
install("matplotlib")
install("numpy")
install("tqdm")

import torch
from torch import nn
from torch.nn import functional as F




def heaviside(x):
    return (x >= 0.).to(x.dtype)


class SurrogateFunctionBase(nn.Module):
    """
    Surrogate Function 的基类
    :param alpha: 为一些能够调控函数形状的代理函数提供参数.
    :param requires_grad: 参数 ``alpha`` 是否需要计算梯度, 默认为 ``False``
    """

    def __init__(self, alpha, requires_grad=True):
        super().__init__()
        self.alpha = nn.Parameter(
            torch.tensor(alpha, dtype=torch.float),
            requires_grad=requires_grad)

    @staticmethod
    def act_fun(x, alpha):
        """
        :param x: 膜电位的输入
        :param alpha: 控制代理梯度形状的变量, 可以为 ``NoneType``
        :return: 激发之后的spike, 取值为 ``[0, 1]``
        """
        raise NotImplementedError

    def forward(self, x):
        """
        :param x: 膜电位输入
        :return: 激发之后的spike
        """
        return self.act_fun(x, self.alpha)


'''
    sigmoid surrogate function.
'''


class sigmoid(torch.autograd.Function):
    """
    使用 sigmoid 作为代理梯度函数
    对应的原函数为:

    .. math::
            g(x) = \\mathrm{sigmoid}(\\alpha x) = \\frac{1}{1+e^{-\\alpha x}}
    反向传播的函数为:

    .. math::
            g'(x) = \\alpha * (1 - \\mathrm{sigmoid} (\\alpha x)) \\mathrm{sigmoid} (\\alpha x)

    """

    @staticmethod
    def forward(ctx, x, alpha):
        if x.requires_grad:
            ctx.save_for_backward(x)
            ctx.alpha = alpha
        return heaviside(x)

    @staticmethod
    def backward(ctx, grad_output):
        grad_x = None
        if ctx.needs_input_grad[0]:
            s_x = torch.sigmoid(ctx.alpha * ctx.saved_tensors[0])
            grad_x = grad_output * s_x * (1 - s_x) * ctx.alpha
        return grad_x, None


class SigmoidGrad(SurrogateFunctionBase):
    def __init__(self, alpha=1., requires_grad=False):
        super().__init__(alpha, requires_grad)

    @staticmethod
    def act_fun(x, alpha):
        return sigmoid.apply(x, alpha)


'''
    atan surrogate function.
'''


class atan(torch.autograd.Function):
    """
    使用 Atan 作为代理梯度函数
    对应的原函数为:

    .. math::
            g(x) = \\frac{1}{\\pi} \\arctan(\\frac{\\pi}{2}\\alpha x) + \\frac{1}{2}
    反向传播的函数为:

    .. math::
            g'(x) = \\frac{\\alpha}{2(1 + (\\frac{\\pi}{2}\\alpha x)^2)}

    """

    @staticmethod
    def forward(ctx, inputs, alpha):
        ctx.save_for_backward(inputs, alpha)
        return inputs.gt(0.).float()

    @staticmethod
    def backward(ctx, grad_output):
        grad_x = None
        grad_alpha = None

        shared_c = grad_output / \
                   (1 + (ctx.saved_tensors[1] * math.pi /
                         2 * ctx.saved_tensors[0]).square())
        if ctx.needs_input_grad[0]:
            grad_x = ctx.saved_tensors[1] / 2 * shared_c
        if ctx.needs_input_grad[1]:
            grad_alpha = (ctx.saved_tensors[0] / 2 * shared_c).sum()

        return grad_x, grad_alpha


class AtanGrad(SurrogateFunctionBase):
    def __init__(self, alpha=2., requires_grad=True):
        super().__init__(alpha, requires_grad)

    @staticmethod
    def act_fun(x, alpha):
        return atan.apply(x, alpha)


'''
    gate surrogate fucntion.
'''


class gate(torch.autograd.Function):
    """
    使用 gate 作为代理梯度函数
    对应的原函数为:

    .. math::
            g(x) = \\mathrm{NonzeroSign}(x) \\log (|\\alpha x| + 1)
    反向传播的函数为:

    .. math::
            g'(x) = \\frac{\\alpha}{1 + |\\alpha x|} = \\frac{1}{\\frac{1}{\\alpha} + |x|}

    """

    @staticmethod
    def forward(ctx, x, alpha):
        if x.requires_grad:
            grad_x = torch.where(x.abs() < 1. / alpha, torch.ones_like(x), torch.zeros_like(x))
            ctx.save_for_backward(grad_x)
        return x.gt(0).float()

    @staticmethod
    def backward(ctx, grad_output):
        grad_x = None
        if ctx.needs_input_grad[0]:
            grad_x = grad_output * ctx.saved_tensors[0]
        return grad_x, None


class GateGrad(SurrogateFunctionBase):
    def __init__(self, alpha=2., requires_grad=False):
        super().__init__(alpha, requires_grad)

    @staticmethod
    def act_fun(x, alpha):
        return gate.apply(x, alpha)


'''
    gatquadratic_gate surrogate function.
'''


class quadratic_gate(torch.autograd.Function):
    """
    使用 quadratic_gate 作为代理梯度函数
    对应的原函数为:

    .. math::
        g(x) =
        \\begin{cases}
        0, & x < -\\frac{1}{\\alpha} \\\\
        -\\frac{1}{2}\\alpha^2|x|x + \\alpha x + \\frac{1}{2}, & |x| \\leq \\frac{1}{\\alpha}  \\\\
        1, & x > \\frac{1}{\\alpha} \\\\
        \\end{cases}

    反向传播的函数为:

    .. math::
        g'(x) =
        \\begin{cases}
        0, & |x| > \\frac{1}{\\alpha} \\\\
        -\\alpha^2|x|+\\alpha, & |x| \\leq \\frac{1}{\\alpha}
        \\end{cases}

    """

    @staticmethod
    def forward(ctx, x, alpha):
        if x.requires_grad:
            mask_zero = (x.abs() > 1 / alpha)
            grad_x = -alpha * alpha * x.abs() + alpha
            grad_x.masked_fill_(mask_zero, 0)
            ctx.save_for_backward(grad_x)
        return x.gt(0.).float()

    @staticmethod
    def backward(ctx, grad_output):
        grad_x = None
        if ctx.needs_input_grad[0]:
            grad_x = grad_output * ctx.saved_tensors[0]
        return grad_x, None


class QGateGrad(SurrogateFunctionBase):
    def __init__(self, alpha=2., requires_grad=False):
        super().__init__(alpha, requires_grad)

    @staticmethod
    def act_fun(x, alpha):
        return quadratic_gate.apply(x, alpha)


class relu_like(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        if x.requires_grad:
            ctx.save_for_backward(x, alpha)
        return heaviside(x)

    @staticmethod
    def backward(ctx, grad_output):
        grad_x, grad_alpha = None, None
        x, alpha = ctx.saved_tensors
        if ctx.needs_input_grad[0]:
            grad_x = grad_output * x.gt(0.).float() * alpha
        if ctx.needs_input_grad[1]:
            grad_alpha = (grad_output * F.relu(x)).sum()
        return grad_x, grad_alpha

class RoundGrad(nn.Module):
    def __init__(self, **kwargs):
        super(RoundGrad, self).__init__()
        self.act = nn.Hardtanh(-.5, 4.5)

    def forward(self, x):
        x = self.act(x)
        return x.ceil() + x - x.detach()

class ReLUGrad(SurrogateFunctionBase):
    """
    使用ReLU作为代替梯度函数, 主要用为相同结构的ANN的测试
    """

    def __init__(self, alpha=2., requires_grad=False):
        super().__init__(alpha, requires_grad)

    @staticmethod
    def act_fun(x, alpha):
        return relu_like.apply(x, alpha)


'''
    Straight-Through (ST) Estimator
'''


class straight_through_estimator(torch.autograd.Function):
    """
    使用直通估计器作为代理梯度函数
    http://arxiv.org/abs/1308.3432
    """

    @staticmethod
    def forward(ctx, inputs):
        outputs = heaviside(inputs)
        ctx.save_for_backward(outputs)
        return outputs

    @staticmethod
    def backward(ctx, grad_output):
        grad_x = None
        if ctx.needs_input_grad[0]:
            grad_x = grad_output
        return grad_x


class stdp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs):
        outputs = inputs.gt(0.).float()
        ctx.save_for_backward(outputs)
        return outputs

    @staticmethod
    def backward(ctx, grad_output):
        inputs, = ctx.saved_tensors
        return inputs * grad_output


class STDPGrad(SurrogateFunctionBase):
    def __init__(self, alpha=2., requires_grad=False):
        super().__init__(alpha, requires_grad)

    @staticmethod
    def act_fun(x, alpha):
        return stdp.apply(x)





class backeigate(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return input.gt(0.).float()

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        temp = abs(input) < 0.5
        return grad_input * temp.float()


class BackEIGateGrad(SurrogateFunctionBase):
    def __init__(self, alpha=2., requires_grad=False):
        super().__init__(alpha, requires_grad)

    @staticmethod
    def act_fun(x, alpha):
        return backeigate.apply(x)

class ei(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return torch.sign(input).float()

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        temp = abs(input) < 0.5
        return grad_input * temp.float()


class EIGrad(SurrogateFunctionBase):
    def __init__(self, alpha=2., requires_grad=False):
        super().__init__(alpha, requires_grad)

    @staticmethod
    def act_fun(x, alpha):
        return ei.apply(x)
import warnings
import torch
from torch import nn
import torch.nn.functional as F


class LateralInhibition(nn.Module):
    """
    侧抑制 用于发放脉冲的神经元抑制其他同层神经元 在膜电位上作用
    """
    def __init__(self, node, inh, mode="constant"):
        super().__init__()
        self.inh = inh
        self.node = node
        self.mode = mode

    def forward(self, x: torch.Tensor, xori=None):
        # x.shape = [N, C,W,H]
        # ret.shape = [N, C,W,H]
        if self.mode == "constant":

            self.node.mem = self.node.mem - self.inh * (x.max(1, True)[0] - x)

        elif self.mode == "max":
            self.node.mem = self.node.mem - self.inh * xori.max(1, True)[0] .detach() * (x.max(1, True)[0] - x)
        elif self.mode == "threshold":
            self.node.mem = self.node.mem - self.inh * self.node.threshold * (x.max(1, True)[0] - x)
        else:
            pass
        return x
import warnings
import math
import numpy as np
import torch
from torch import nn
from torch import einsum
from torch.nn.modules.batchnorm import _BatchNorm
import torch.nn.functional as F
from torch.nn import Parameter
from einops import rearrange


class VotingLayer(nn.Module):
    """
    用于SNNs的输出层, 几个神经元投票选出最终的类
    :param voter_num: 投票的神经元的数量, 例如 ``voter_num = 10``, 则表明会对这10个神经元取平均
    """

    def __init__(self, voter_num: int):
        super().__init__()
        self.voting = nn.AvgPool1d(voter_num, voter_num)

    def forward(self, x: torch.Tensor):
        # x.shape = [N, voter_num * C]
        # ret.shape = [N, C]
        return self.voting(x.unsqueeze(1)).squeeze(1)


class WTALayer(nn.Module):
    """
    winner take all用于SNNs的每层后，将随机选取一个或者多个输出
    :param k: X选取的输出数目 k默认等于1
    """
    def __init__(self, k=1):
        super().__init__()
        self.k = k

    def forward(self, x: torch.Tensor):
        # x.shape = [N, C,W,H]
        # ret.shape = [N, C,W,H]
        pos = x * torch.rand(x.shape, device=x.device)
        if self.k > 1:
            x = x * (pos >= pos.topk(self.k, dim=1)[0][:, -1:]).float()
        else:
            x = x * (pos >= pos.max(1, True)[0]).float()

        return x


class NDropout(nn.Module):
    """
    与Drop功能相同, 但是会保证同一个样本不同时刻的mask相同.
    """

    def __init__(self, p):
        super(NDropout, self).__init__()
        self.p = p
        self.mask = None

    def n_reset(self):
        """
        重置, 能够生成新的mask
        :return:
        """
        self.mask = None

    def create_mask(self, x):
        """
        生成新的mask
        :param x: 输入Tensor, 生成与之形状相同的mask
        :return:
        """
        self.mask = F.dropout(torch.ones_like(x.data), self.p, training=True)

    def forward(self, x):
        if self.training:
            if self.mask is None:
                self.create_mask(x)

            return self.mask * x
        else:
            return x


class WSConv2d(nn.Conv2d):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, gain=True):
        super(WSConv2d, self).__init__(in_channels, out_channels, kernel_size, stride,
                                       padding, dilation, groups, bias)

        if gain:
            self.gain = nn.Parameter(torch.ones(self.out_channels, 1, 1, 1))
        else:
            self.gain = 1.

    def forward(self, x):
        weight = self.weight
        weight_mean = weight.mean(dim=1, keepdim=True).mean(dim=2,
                                                            keepdim=True).mean(dim=3, keepdim=True)
        weight = weight - weight_mean
        std = weight.view(weight.size(0), -1).std(dim=1).view(-1, 1, 1, 1) + 1e-5
        weight = self.gain * weight / std.expand_as(weight)
        return F.conv2d(x, weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)


class ThresholdDependentBatchNorm2d(_BatchNorm):
    """
    tdBN
    https://ojs.aaai.org/index.php/AAAI/article/view/17320
    """

    def __init__(self, num_features, alpha: float, threshold: float = .5, layer_by_layer: bool = True, affine: bool = True,**kwargs):
        self.alpha = alpha
        self.threshold = threshold

        super().__init__(num_features=num_features, affine=affine)

        assert layer_by_layer, \
            'tdBN may works in step-by-step mode, which will not take temporal dimension into batch norm'
        assert self.affine, 'ThresholdDependentBatchNorm needs to set `affine = True`!'

        torch.nn.init.constant_(self.weight, alpha * threshold)

    def _check_input_dim(self, input):
        if input.dim() != 4:
            raise ValueError("expected 4D input (got {}D input)".format(input.dim()))

    def forward(self, input):
        # input = rearrange(input, '(t b) c w h -> b (t c) w h', t=self.step)
        output = super().forward(input)
        return output
        # return rearrange(output, 'b (t c) w h -> (t b) c w h', t=self.step)

class TEBN(nn.Module):
    def __init__(self, num_features,step, eps=1e-5, momentum=0.1,**kwargs):
        super(TEBN, self).__init__()
        self.bn = nn.BatchNorm3d(num_features)
        self.p = nn.Parameter(torch.ones(4, 1, 1, 1, 1))
        self.step=step
    def forward(self, input):
        #y = input.transpose(1, 2).contiguous()  # N T C H W ,  N C T H W
        y = rearrange(input,"(t b) c w h -> t c b w h",t=self.step)
        y = self.bn(y)
        # y = y.contiguous().transpose(1, 2)
        # y = y.transpose(0, 1).contiguous()  # NTCHW  TNCHW
        y = rearrange(y,"t c b w h -> t b c w h")
        y = y * self.p
        #y = y.contiguous().transpose(0, 1)  # TNCHW  NTCHW
        y = rearrange(y, "t b c w h -> (t b) c w h")
        return y
class LayerNorm(nn.Module):
    """ LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class SMaxPool(nn.Module):
    """用于转换方法的最大池化层的常规替换
    选用具有最大脉冲发放率的神经元的脉冲通过，能够满足一般性最大池化层的需要

    Reference:
    https://arxiv.org/abs/1612.04052
    """

    def __init__(self, child):
        super(SMaxPool, self).__init__()
        self.opration = child
        self.sumspike = 0

    def forward(self, x):
        self.sumspike += x
        single = self.opration(self.sumspike * 1000)
        sum_plus_spike = self.opration(x + self.sumspike * 1000)

        return sum_plus_spike - single

    def reset(self):
        self.sumspike = 0


class LIPool(nn.Module):
    r"""用于转换方法的最大池化层的精准替换
    LIPooling通过引入侧向抑制机制保证在转换后的SNN中输出的最大值与期望值相同。

    Reference:
    https://arxiv.org/abs/2204.13271
    """

    def __init__(self, child=None):
        super(LIPool, self).__init__()
        if child is None:
            raise NotImplementedError("child should be Pooling operation with torch.")

        self.opration = child
        self.sumspike = 0

    def forward(self, x):
        self.sumspike += x
        out = self.opration(self.sumspike)
        self.sumspike -= F.interpolate(out, scale_factor=2, mode='nearest')
        return out

    def reset(self):
        self.sumspike = 0
# encoding: utf-8
# Author    : Floyed<Floyed_Shen@outlook.com>
# Datetime  : 2022/4/10 18:46
# User      : Floyed
# Product   : PyCharm
# Project   : braincog
# File      : node.py
# explain   : 神经元节点类型

import abc
import math
from abc import ABC
import numpy as np
import random
import torch
from torch import nn
from torch.nn import Parameter
import torch.nn.functional as F
from einops import rearrange, repeat


class BaseNode(nn.Module, abc.ABC):
    """
    神经元模型的基类
    :param threshold: 神经元发放脉冲需要达到的阈值
    :param v_reset: 静息电位
    :param dt: 时间步长
    :param step: 仿真步
    :param requires_thres_grad: 是否需要计算对于threshold的梯度, 默认为 ``False``
    :param sigmoid_thres: 是否使用sigmoid约束threshold的范围搭到 [0, 1], 默认为 ``False``
    :param requires_fp: 是否需要在推理过程中保存feature map, 需要消耗额外的内存和时间, 默认为 ``False``
    :param layer_by_layer: 是否以一次性计算所有step的输出, 在网络模型较大的情况下, 一般会缩短单次推理的时间, 默认为 ``False``
    :param n_groups: 在不同的时间步, 是否使用不同的权重, 默认为 ``1``, 即不分组
    :param mem_detach: 是否将上一时刻的膜电位在计算图中截断
    :param args: 其他的参数
    :param kwargs: 其他的参数
    """

    def __init__(self,
                 threshold=.5,
                 v_reset=0.,
                 dt=1.,
                 step=8,
                 requires_thres_grad=False,
                 sigmoid_thres=False,
                 requires_fp=False,
                 layer_by_layer=False,
                 n_groups=1,
                 *args,
                 **kwargs):

        super(BaseNode, self).__init__()
        self.threshold = Parameter(torch.tensor(threshold), requires_grad=requires_thres_grad)
        self.sigmoid_thres = sigmoid_thres
        self.mem = 0.
        self.spike = 0.
        self.dt = dt
        self.feature_map = []
        self.mem_collect = []
        self.requires_fp = requires_fp
        self.v_reset = v_reset
        self.step = step
        self.layer_by_layer = layer_by_layer
        self.groups = n_groups
        self.mem_detach = kwargs['mem_detach'] if 'mem_detach' in kwargs else False
        self.requires_mem = kwargs['requires_mem'] if 'requires_mem' in kwargs else False

    @abc.abstractmethod
    def calc_spike(self):
        """
        通过当前的mem计算是否发放脉冲，并reset
        :return: None
        """

        pass

    def integral(self, inputs):
        """
        计算由当前inputs对于膜电势的累积
        :param inputs: 当前突触输入电流
        :type inputs: torch.tensor
        :return: None
        """

        pass

    def get_thres(self):
        return self.threshold if not self.sigmoid_thres else self.threshold.sigmoid()

    def rearrange2node(self, inputs):
        if self.groups != 1:
            if len(inputs.shape) == 4:
                outputs = rearrange(inputs, 'b (c t) w h -> t b c w h', t=self.step)
            elif len(inputs.shape) == 2:
                outputs = rearrange(inputs, 'b (c t) -> t b c', t=self.step)
            else:
                raise NotImplementedError

        elif self.layer_by_layer:
            if len(inputs.shape) == 4:
                outputs = rearrange(inputs, '(t b) c w h -> t b c w h', t=self.step)
            elif len(inputs.shape) == 3:
                outputs = rearrange(inputs, '(t b) n c -> t b n c', t=self.step)
            elif len(inputs.shape) == 2:
                outputs = rearrange(inputs, '(t b) c -> t b c', t=self.step)
            else:
                raise NotImplementedError


        else:
            outputs = inputs

        return outputs

    def rearrange2op(self, inputs):
        if self.groups != 1:
            if len(inputs.shape) == 5:
                outputs = rearrange(inputs, 't b c w h -> b (c t) w h')
            elif len(inputs.shape) == 3:
                outputs = rearrange(inputs, ' t b c -> b (c t)')
            else:
                raise NotImplementedError
        elif self.layer_by_layer:
            if len(inputs.shape) == 5:
                outputs = rearrange(inputs, 't b c w h -> (t b) c w h')
            elif len(inputs.shape) == 4:
                outputs = rearrange(inputs, ' t b n c -> (t b) n c')
            elif len(inputs.shape) == 3:
                outputs = rearrange(inputs, ' t b c -> (t b) c')
            else:
                raise NotImplementedError

        else:
            outputs = inputs

        return outputs

    def forward(self, inputs):
        """
        torch.nn.Module 默认调用的函数，用于计算膜电位的输入和脉冲的输出
        在```self.requires_fp is True``` 的情况下，可以使得```self.feature_map```用于记录trace
        :param inputs: 当前输入的膜电位
        :return: 输出的脉冲
        """

        if self.layer_by_layer or self.groups != 1:
            inputs = self.rearrange2node(inputs)

            outputs = []
            for i in range(self.step):

                if self.mem_detach and hasattr(self.mem, 'detach'):
                    self.mem = self.mem.detach()
                    self.spike = self.spike.detach()
                self.integral(inputs[i])

                self.calc_spike()

                if self.requires_fp is True:
                    self.feature_map.append(self.spike)
                if self.requires_mem is True:
                    self.mem_collect.append(self.mem)
                outputs.append(self.spike)
            outputs = torch.stack(outputs)

            outputs = self.rearrange2op(outputs)
            return outputs
        else:
            if self.mem_detach and hasattr(self.mem, 'detach'):
                self.mem = self.mem.detach()
                self.spike = self.spike.detach()
            self.integral(inputs)
            self.calc_spike()
            if self.requires_fp is True:
                self.feature_map.append(self.spike)
            if self.requires_mem is True:
                self.mem_collect.append(self.mem)
            return self.spike

    def n_reset(self):
        """
        神经元重置，用于模型接受两个不相关输入之间，重置神经元所有的状态
        :return: None
        """
        self.mem = self.v_reset
        self.spike = 0.
        self.feature_map = []
        self.mem_collect = []
    def get_n_attr(self, attr):

        if hasattr(self, attr):
            return getattr(self, attr)
        else:
            return None

    def set_n_warm_up(self, flag):
        """
        一些训练策略会在初始的一些epoch，将神经元视作ANN的激活函数训练，此为设置是否使用该方法训练
        :param flag: True：神经元变为激活函数， False：不变
        :return: None
        """
        self.warm_up = flag

    def set_n_threshold(self, thresh):
        """
        动态设置神经元的阈值
        :param thresh: 阈值
        :return:
        """
        self.threshold = Parameter(torch.tensor(thresh, dtype=torch.float), requires_grad=False)

    def set_n_tau(self, tau):
        """
        动态设置神经元的衰减系数，用于带Leaky的神经元
        :param tau: 衰减系数
        :return:
        """
        if hasattr(self, 'tau'):
            self.tau = Parameter(torch.tensor(tau, dtype=torch.float), requires_grad=False)
        else:
            raise NotImplementedError

#============================================================================
# node的基类
class BaseMCNode(nn.Module, abc.ABC):
    """
    多房室神经元模型的基类
    :param threshold: 神经元发放脉冲需要达到的阈值
    :param v_reset: 静息电位
    :param comps: 神经元不同房室, 例如["apical", "basal", "soma"]
    """
    def __init__(self,
                 threshold=1.0,
                 v_reset=0.,
                 comps=[]):
        super().__init__()
        self.threshold = Parameter(torch.tensor(threshold), requires_grad=False)
        # self.decay = Parameter(torch.tensor(decay), requires_grad=False)
        self.v_reset = v_reset
        assert len(comps) != 0
        self.mems = dict()
        for c in comps:
            self.mems[c] = None
        self.spike = None
        self.warm_up = False

    @abc.abstractmethod
    def calc_spike(self):
        pass
    @abc.abstractmethod
    def integral(self, inputs):
        pass

    def forward(self, inputs: dict):
        '''
        Params:
            inputs dict: Inputs for every compartments of neuron
        '''
        if self.warm_up:
            return inputs
        else:
            self.integral(**inputs)
            self.calc_spike()
            return self.spike

    def n_reset(self):
        for c in self.mems.keys():
            self.mems[c] = self.v_reset
        self.spike = 0.0

    def get_n_fire_rate(self):
        if self.spike is None:
            return 0.
        return float((self.spike.detach() >= self.threshold).sum()) / float(np.product(self.spike.shape))

    def set_n_warm_up(self, flag):
        self.warm_up = flag

    def set_n_threshold(self, thresh):
        self.threshold = Parameter(torch.tensor(thresh, dtype=torch.float), requires_grad=False)


class ThreeCompNode(BaseMCNode):
    """
    三房室神经元模型
    :param threshold: 神经元发放脉冲需要达到的阈值
    :param v_reset: 静息电位
    :param tau: 胞体膜电位时间常数, 用于控制胞体膜电位衰减
    :param tau_basal: 基底树突膜电位时间常数, 用于控制基地树突胞体膜电位衰减
    :param tau_apical: 远端树突膜电位时间常数, 用于控制远端树突胞体膜电位衰减
    :param comps: 神经元不同房室, 例如["apical", "basal", "soma"]
    :param act_fun: 脉冲梯度代理函数
    """
    def __init__(self,
                 threshold=1.0,
                 tau=2.0,
                 tau_basal=2.0,
                 tau_apical=2.0,
                 v_reset=0.0,
                 comps=['basal', 'apical', 'soma'],
                 act_fun=AtanGrad):
        g_B = 0.6
        g_L = 0.05
        super().__init__(threshold, v_reset, comps)
        self.tau = tau
        self.tau_basal = tau_basal
        self.tau_apical = tau_apical
        self.act_fun = act_fun(alpha=tau, requires_grad=False)

    def integral(self, basal_inputs, apical_inputs):
        '''
        Params:
            inputs torch.Tensor: Inputs for basal dendrite
        '''

        self.mems['basal'] =  (self.mems['basal'] + basal_inputs) / self.tau_basal
        self.mems['apical'] =  (self.mems['apical'] + apical_inputs) / self.tau_apical

        self.mems['soma'] = self.mems['soma'] + (self.mems['apical'] + self.mems['basal'] - self.mems['soma']) / self.tau


    def calc_spike(self):
        self.spike = self.act_fun(self.mems['soma'] - self.threshold)
        self.mems['soma'] = self.mems['soma']  * (1. - self.spike.detach())
        self.mems['basal'] = self.mems['basal'] * (1. - self.spike.detach())
        self.mems['apical'] = self.mems['apical']  * (1. - self.spike.detach())


#============================================================================

# 用于静态测试 使用ANN的情况 不累积电位
class ReLUNode(BaseNode):
    """
    用于相同连接的ANN的测试
    """

    def __init__(self,
                 *args,
                 **kwargs):
        super().__init__(requires_fp=False, *args, **kwargs)
        self.act_fun = nn.ReLU()

    def forward(self, x):
        """
        参考```BaseNode```
        :param x:
        :return:
        """
        self.spike = self.act_fun(x)
        if self.requires_fp is True:
            self.feature_map.append(self.spike)
        if self.requires_mem is True:
            self.mem_collect.append(self.mem)
        return self.spike

    def calc_spike(self):
        pass


class BiasReLUNode(BaseNode):
    """
    用于相同连接的ANN的测试, 会在每个时刻注入恒定电流, 使得神经元更容易激发
    """

    def __init__(self,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.act_fun = nn.ReLU()

    def forward(self, x):
        self.spike = self.act_fun(x + 0.1)
        if self.requires_fp is True:
            self.feature_map += self.spike
        return self.spike

    def calc_spike(self):
        pass


# ============================================================================
# 用于SNN的node
class IFNode(BaseNode):
    """
    Integrate and Fire Neuron
    :param threshold: 神经元发放脉冲需要达到的阈值
    :param v_reset: 静息电位
    :param dt: 时间步长
    :param step: 仿真步
    :param act_fun: 使用surrogate gradient 对梯度进行近似, 默认为 ``surrogate.AtanGrad``
    :param requires_thres_grad: 是否需要计算对于threshold的梯度, 默认为 ``False``
    :param sigmoid_thres: 是否使用sigmoid约束threshold的范围搭到 [0, 1], 默认为 ``False``
    :param requires_fp: 是否需要在推理过程中保存feature map, 需要消耗额外的内存和时间, 默认为 ``False``
    :param layer_by_layer: 是否以一次性计算所有step的输出, 在网络模型较大的情况下, 一般会缩短单次推理的时间, 默认为 ``False``
    :param n_groups: 在不同的时间步, 是否使用不同的权重, 默认为 ``1``, 即不分组
    :param args: 其他的参数
    :param kwargs: 其他的参数
    """

    def __init__(self, threshold=.5, act_fun=AtanGrad, *args, **kwargs):
        """
        :param threshold:
        :param act_fun:
        :param args:
        :param kwargs:
        """
        super().__init__(threshold, *args, **kwargs)
        if isinstance(act_fun, str):
            act_fun = eval(act_fun)
        self.act_fun = act_fun(alpha=2., requires_grad=False)

    def integral(self, inputs):
        self.mem = self.mem + inputs * self.dt

    def calc_spike(self):
        self.spike = self.act_fun(self.mem - self.get_thres())
        self.mem = self.mem * (1 - self.spike.detach())


class LIFNode(BaseNode):
    """
    Leaky Integrate and Fire
    :param threshold: 神经元发放脉冲需要达到的阈值
    :param v_reset: 静息电位
    :param dt: 时间步长
    :param step: 仿真步
    :param tau: 膜电位时间常数, 用于控制膜电位衰减
    :param act_fun: 使用surrogate gradient 对梯度进行近似, 默认为 ``surrogate.AtanGrad``
    :param requires_thres_grad: 是否需要计算对于threshold的梯度, 默认为 ``False``
    :param sigmoid_thres: 是否使用sigmoid约束threshold的范围搭到 [0, 1], 默认为 ``False``
    :param requires_fp: 是否需要在推理过程中保存feature map, 需要消耗额外的内存和时间, 默认为 ``False``
    :param layer_by_layer: 是否以一次性计算所有step的输出, 在网络模型较大的情况下, 一般会缩短单次推理的时间, 默认为 ``False``
    :param n_groups: 在不同的时间步, 是否使用不同的权重, 默认为 ``1``, 即不分组
    :param args: 其他的参数
    :param kwargs: 其他的参数
    """

    def __init__(self, threshold=0.5, tau=2., act_fun=QGateGrad, *args, **kwargs):
        super().__init__(threshold, *args, **kwargs)
        self.tau = tau
        if isinstance(act_fun, str):
            act_fun = eval(act_fun)
        self.act_fun = act_fun(alpha=2., requires_grad=False)
        # self.threshold = threshold
        # print(threshold)
        # print(tau)

    def integral(self, inputs):
        self.mem = self.mem + (inputs - self.mem) / self.tau

    def calc_spike(self):
        self.spike = self.act_fun(self.mem - self.threshold)
        self.mem = self.mem * (1 - self.spike.detach())


class BurstLIFNode(LIFNode):
    def __init__(self, threshold=.5, tau=2., act_fun=RoundGrad, *args, **kwargs):
        super().__init__(threshold=threshold, tau=tau, act_fun=act_fun, *args, **kwargs)
        self.burst_factor = 1.5

    def calc_spike(self):
        LIFNode.calc_spike(self)
        self.spike = torch.where(self.spike > 1., self.burst_factor * self.spike, self.spike)



class BackEINode(BaseNode):
    """
    BackEINode with self feedback connection and excitatory and inhibitory neurons
    Reference：https://www.sciencedirect.com/science/article/pii/S0893608022002520
    :param threshold: 神经元发放脉冲需要达到的阈值
    :param if_back whether to use self feedback
    :param if_ei whether to use excitotory and inhibitory neurons
    :param args: 其他的参数
    :param kwargs: 其他的参数
    """
    def __init__(self, threshold=0.5, decay=0.2, act_fun=BackEIGateGrad, th_fun=EIGrad, channel=40, if_back=True,
                 if_ei=True, cfg_backei=2, *args, **kwargs):
        super().__init__(threshold, *args, **kwargs)
        self.decay = decay
        if isinstance(act_fun, str):
            act_fun = eval(act_fun)
        if isinstance(th_fun, str):
            th_fun = eval(th_fun)
        self.act_fun = act_fun()
        self.th_fun = th_fun()
        self.channel = channel
        self.if_back = if_back

        if self.if_back:
            self.back = nn.Conv2d(channel, channel, kernel_size=2 * cfg_backei+1, stride=1, padding=cfg_backei)
        self.if_ei = if_ei
        if self.if_ei:
            self.ei = nn.Conv2d(channel, channel, kernel_size=2 * cfg_backei+1, stride=1, padding=cfg_backei)

    def integral(self, inputs):
        if self.mem is None:
            self.mem = torch.zeros_like(inputs)
            self.spike = torch.zeros_like(inputs)
        self.mem = self.decay * self.mem
        if self.if_back:
            self.mem += F.sigmoid(self.back(self.spike)) * inputs
        else:
            self.mem += inputs

    def calc_spike(self):
        if self.if_ei:
            ei_gate = self.th_fun(self.ei(self.mem))
            self.spike = self.act_fun(self.mem-self.threshold)
            self.mem = self.mem * (1 - self.spike)
            self.spike = ei_gate * self.spike
        else:
            self.spike = self.act_fun(self.mem-self.threshold)
            self.mem = self.mem * (1 - self.spike)

    def n_reset(self):
        self.mem = None
        self.spike = None
        self.feature_map = []
        self.mem_collect = []


class NoiseLIFNode(LIFNode):
    """
    Noisy Leaky Integrate and Fire
    在神经元中注入噪声, 默认的噪声分布为 ``Beta(log(2), log(6))``
    :param threshold: 神经元发放脉冲需要达到的阈值
    :param v_reset: 静息电位
    :param dt: 时间步长
    :param step: 仿真步
    :param tau: 膜电位时间常数, 用于控制膜电位衰减
    :param act_fun: 使用surrogate gradient 对梯度进行近似, 默认为 ``surrogate.AtanGrad``
    :param requires_thres_grad: 是否需要计算对于threshold的梯度, 默认为 ``False``
    :param sigmoid_thres: 是否使用sigmoid约束threshold的范围搭到 [0, 1], 默认为 ``False``
    :param requires_fp: 是否需要在推理过程中保存feature map, 需要消耗额外的内存和时间, 默认为 ``False``
    :param layer_by_layer: 是否以一次性计算所有step的输出, 在网络模型较大的情况下, 一般会缩短单次推理的时间, 默认为 ``False``
    :param n_groups: 在不同的时间步, 是否使用不同的权重, 默认为 ``1``, 即不分组
    :param log_alpha: 控制 beta 分布的参数 ``a``
    :param log_beta: 控制 beta 分布的参数 ``b``
    :param args: 其他的参数
    :param kwargs: 其他的参数
    """

    def __init__(self,
                 threshold=1,
                 tau=2.,
                 act_fun=GateGrad,
                 log_alpha=np.log(2),
                 log_beta=np.log(6),
                 *args,
                 **kwargs):
        super().__init__(threshold=threshold, tau=tau, act_fun=act_fun, *args, **kwargs)
        self.log_alpha = Parameter(torch.as_tensor(log_alpha), requires_grad=True)
        self.log_beta = Parameter(torch.as_tensor(log_beta), requires_grad=True)

        # self.fc = nn.Sequential(
        #     nn.Linear(1, 5),
        #     nn.ReLU(),
        #     nn.Linear(5, 5),
        #     nn.ReLU(),
        #     nn.Linear(5, 2)
        # )

    def integral(self, inputs):  # b, c, w, h / b, c
        # self.mu, self.log_var = self.fc(inputs.mean().unsqueeze(0)).split(1)
        alpha, beta = torch.exp(self.log_alpha), torch.exp(self.log_beta)
        mu = alpha / (alpha + beta)
        var = ((alpha + 1) * alpha) / ((alpha + beta + 1) * (alpha + beta))
        noise = torch.distributions.beta.Beta(alpha, beta).sample(inputs.shape) * self.get_thres()
        noise = noise * var / var.detach() + mu - mu.detach()

        self.mem = self.mem + ((inputs - self.mem) / self.tau + noise) * self.dt


class BiasLIFNode(BaseNode):
    """
    带有恒定电流输入Bias的LIF神经元，用于带有抑制性/反馈链接的网络的测试
    Noisy Leaky Integrate and Fire
    在神经元中注入噪声, 默认的噪声分布为 ``Beta(log(2), log(6))``
    :param threshold: 神经元发放脉冲需要达到的阈值
    :param v_reset: 静息电位
    :param dt: 时间步长
    :param step: 仿真步
    :param tau: 膜电位时间常数, 用于控制膜电位衰减
    :param act_fun: 使用surrogate gradient 对梯度进行近似, 默认为 ``surrogate.AtanGrad``
    :param requires_thres_grad: 是否需要计算对于threshold的梯度, 默认为 ``False``
    :param sigmoid_thres: 是否使用sigmoid约束threshold的范围搭到 [0, 1], 默认为 ``False``
    :param requires_fp: 是否需要在推理过程中保存feature map, 需要消耗额外的内存和时间, 默认为 ``False``
    :param layer_by_layer: 是否以一次性计算所有step的输出, 在网络模型较大的情况下, 一般会缩短单次推理的时间, 默认为 ``False``
    :param n_groups: 在不同的时间步, 是否使用不同的权重, 默认为 ``1``, 即不分组
    :param args: 其他的参数
    :param kwargs: 其他的参数
    """

    def __init__(self, threshold=1., tau=2., act_fun=AtanGrad, *args, **kwargs):
        super().__init__(threshold, *args, **kwargs)
        self.tau = tau
        if isinstance(act_fun, str):
            act_fun = eval(act_fun)
        self.act_fun = act_fun(alpha=2., requires_grad=False)

    def integral(self, inputs):
        self.mem = self.mem + ((inputs - self.mem) / self.tau) * self.dt + 0.1

    def calc_spike(self):
        self.spike = self.act_fun(self.mem - self.get_thres())
        self.mem = self.mem * (1 - self.spike.detach())


class LIFSTDPNode(BaseNode):
    """
    用于执行STDP运算时使用的节点 decay的方式是膜电位乘以decay并直接加上输入电流
    """

    def __init__(self, threshold=1., tau=2., act_fun=AtanGrad, *args, **kwargs):
        super().__init__(threshold, *args, **kwargs)
        self.tau = tau
        if isinstance(act_fun, str):
            act_fun = eval(act_fun)
        self.act_fun = act_fun(alpha=2., requires_grad=False)

    def integral(self, inputs):
        self.mem = self.mem * self.tau + inputs

    def calc_spike(self):
        self.spike = self.act_fun(self.mem - self.threshold)
        # print(( self.threshold).max())
        self.mem = self.mem * (1 - self.spike.detach())

    def requires_activation(self):
        return False


class PLIFNode(BaseNode):
    """
    Parametric LIF， 其中的 ```tau``` 会被backward过程影响
    Reference：https://arxiv.org/abs/2007.05785
    :param threshold: 神经元发放脉冲需要达到的阈值
    :param v_reset: 静息电位
    :param dt: 时间步长
    :param step: 仿真步
    :param tau: 膜电位时间常数, 用于控制膜电位衰减
    :param act_fun: 使用surrogate gradient 对梯度进行近似, 默认为 ``surrogate.AtanGrad``
    :param requires_thres_grad: 是否需要计算对于threshold的梯度, 默认为 ``False``
    :param sigmoid_thres: 是否使用sigmoid约束threshold的范围搭到 [0, 1], 默认为 ``False``
    :param requires_fp: 是否需要在推理过程中保存feature map, 需要消耗额外的内存和时间, 默认为 ``False``
    :param layer_by_layer: 是否以一次性计算所有step的输出, 在网络模型较大的情况下, 一般会缩短单次推理的时间, 默认为 ``False``
    :param n_groups: 在不同的时间步, 是否使用不同的权重, 默认为 ``1``, 即不分组
    :param args: 其他的参数
    :param kwargs: 其他的参数
    """

    def __init__(self, threshold=1., tau=2., act_fun=AtanGrad, *args, **kwargs):
        super().__init__(threshold, *args, **kwargs)
        init_w = -math.log(tau - 1.)
        if isinstance(act_fun, str):
            act_fun = eval(act_fun)
        self.act_fun = act_fun(alpha=2., requires_grad=True)
        self.w = nn.Parameter(torch.as_tensor(init_w))

    def integral(self, inputs):
        self.mem = self.mem + ((inputs - self.mem) * self.w.sigmoid()) * self.dt

    def calc_spike(self):
        self.spike = self.act_fun(self.mem - self.get_thres())
        self.mem = self.mem * (1 - self.spike.detach())


class NoisePLIFNode(PLIFNode):
    """
    Noisy Parametric Leaky Integrate and Fire
    :param threshold: 神经元发放脉冲需要达到的阈值
    :param v_reset: 静息电位
    :param dt: 时间步长
    :param step: 仿真步
    :param tau: 膜电位时间常数, 用于控制膜电位衰减
    :param act_fun: 使用surrogate gradient 对梯度进行近似, 默认为 ``surrogate.AtanGrad``
    :param requires_thres_grad: 是否需要计算对于threshold的梯度, 默认为 ``False``
    :param sigmoid_thres: 是否使用sigmoid约束threshold的范围搭到 [0, 1], 默认为 ``False``
    :param requires_fp: 是否需要在推理过程中保存feature map, 需要消耗额外的内存和时间, 默认为 ``False``
    :param layer_by_layer: 是否以一次性计算所有step的输出, 在网络模型较大的情况下, 一般会缩短单次推理的时间, 默认为 ``False``
    :param n_groups: 在不同的时间步, 是否使用不同的权重, 默认为 ``1``, 即不分组
    :param args: 其他的参数
    :param kwargs: 其他的参数
    """

    def __init__(self,
                 threshold=1,
                 tau=2.,
                 act_fun=GateGrad,
                 *args,
                 **kwargs):
        super().__init__(threshold=threshold, tau=tau, act_fun=act_fun, *args, **kwargs)
        log_alpha = kwargs['log_alpha'] if 'log_alpha' in kwargs else np.log(2)
        log_beta = kwargs['log_beta'] if 'log_beta' in kwargs else np.log(6)
        self.log_alpha = Parameter(torch.as_tensor(log_alpha), requires_grad=True)
        self.log_beta = Parameter(torch.as_tensor(log_beta), requires_grad=True)

        # self.fc = nn.Sequential(
        #     nn.Linear(1, 5),
        #     nn.ReLU(),
        #     nn.Linear(5, 5),
        #     nn.ReLU(),
        #     nn.Linear(5, 2)
        # )

    def integral(self, inputs):  # b, c, w, h / b, c
        # self.mu, self.log_var = self.fc(inputs.mean().unsqueeze(0)).split(1)
        alpha, beta = torch.exp(self.log_alpha), torch.exp(self.log_beta)
        mu = alpha / (alpha + beta)
        var = ((alpha + 1) * alpha) / ((alpha + beta + 1) * (alpha + beta))
        noise = torch.distributions.beta.Beta(alpha, beta).sample(inputs.shape) * self.get_thres()
        noise = noise * var / var.detach() + mu - mu.detach()
        self.mem = self.mem + ((inputs - self.mem) * self.w.sigmoid() + noise) * self.dt


class BiasPLIFNode(BaseNode):
    """
    Parametric LIF with bias
    :param threshold: 神经元发放脉冲需要达到的阈值
    :param v_reset: 静息电位
    :param dt: 时间步长
    :param step: 仿真步
    :param tau: 膜电位时间常数, 用于控制膜电位衰减
    :param act_fun: 使用surrogate gradient 对梯度进行近似, 默认为 ``surrogate.AtanGrad``
    :param requires_thres_grad: 是否需要计算对于threshold的梯度, 默认为 ``False``
    :param sigmoid_thres: 是否使用sigmoid约束threshold的范围搭到 [0, 1], 默认为 ``False``
    :param requires_fp: 是否需要在推理过程中保存feature map, 需要消耗额外的内存和时间, 默认为 ``False``
    :param layer_by_layer: 是否以一次性计算所有step的输出, 在网络模型较大的情况下, 一般会缩短单次推理的时间, 默认为 ``False``
    :param n_groups: 在不同的时间步, 是否使用不同的权重, 默认为 ``1``, 即不分组
    :param args: 其他的参数
    :param kwargs: 其他的参数
    """

    def __init__(self, threshold=1., tau=2., act_fun=AtanGrad, *args, **kwargs):
        super().__init__(threshold, *args, **kwargs)
        init_w = -math.log(tau - 1.)
        if isinstance(act_fun, str):
            act_fun = eval(act_fun)
        self.act_fun = act_fun(alpha=2., requires_grad=True)
        self.w = nn.Parameter(torch.as_tensor(init_w))

    def integral(self, inputs):
        self.mem = self.mem + ((inputs - self.mem) * self.w.sigmoid() + 0.1) * self.dt

    def calc_spike(self):
        self.spike = self.act_fun(self.mem - self.get_thres())
        self.mem = self.mem * (1 - self.spike.detach())


class DoubleSidePLIFNode(LIFNode):
    """
    能够输入正负脉冲的 PLIF
    :param threshold: 神经元发放脉冲需要达到的阈值
    :param v_reset: 静息电位
    :param dt: 时间步长
    :param step: 仿真步
    :param tau: 膜电位时间常数, 用于控制膜电位衰减
    :param act_fun: 使用surrogate gradient 对梯度进行近似, 默认为 ``surrogate.AtanGrad``
    :param requires_thres_grad: 是否需要计算对于threshold的梯度, 默认为 ``False``
    :param sigmoid_thres: 是否使用sigmoid约束threshold的范围搭到 [0, 1], 默认为 ``False``
    :param requires_fp: 是否需要在推理过程中保存feature map, 需要消耗额外的内存和时间, 默认为 ``False``
    :param layer_by_layer: 是否以一次性计算所有step的输出, 在网络模型较大的情况下, 一般会缩短单次推理的时间, 默认为 ``False``
    :param n_groups: 在不同的时间步, 是否使用不同的权重, 默认为 ``1``, 即不分组
    :param args: 其他的参数
    :param kwargs: 其他的参数
    """

    def __init__(self,
                 threshold=.5,
                 tau=2.,
                 act_fun=AtanGrad,
                 *args,
                 **kwargs):
        super().__init__(threshold, tau, act_fun, *args, **kwargs)
        if isinstance(act_fun, str):
            act_fun = eval(act_fun)
        self.act_fun = act_fun(alpha=2., requires_grad=True)

    def calc_spike(self):
        self.spike = self.act_fun(self.mem - self.get_thres()) - self.act_fun(self.get_thres - self.mem)
        self.mem = self.mem * (1. - torch.abs(self.spike.detach()))


class IzhNode(BaseNode):
    """
    Izhikevich 脉冲神经元
    :param threshold: 神经元发放脉冲需要达到的阈值
    :param v_reset: 静息电位
    :param dt: 时间步长
    :param step: 仿真步
    :param tau: 膜电位时间常数, 用于控制膜电位衰减
    :param act_fun: 使用surrogate gradient 对梯度进行近似, 默认为 ``surrogate.AtanGrad``
    :param args: 其他的参数
    :param kwargs: 其他的参数
    """

    def __init__(self, threshold=1., tau=2., act_fun=AtanGrad, *args, **kwargs):
        super().__init__(threshold, *args, **kwargs)
        self.tau = tau
        if isinstance(act_fun, str):
            act_fun = eval(act_fun)
        self.act_fun = act_fun(alpha=2., requires_grad=False)
        self.a = kwargs['a'] if 'a' in kwargs else 0.02
        self.b = kwargs['b'] if 'b' in kwargs else 0.2
        self.c = kwargs['c'] if 'c' in kwargs else -55.
        self.d = kwargs['d'] if 'd' in kwargs else -2.
        '''
        v' = 0.04v^2 + 5v + 140 -u + I
        u' = a(bv-u)
        下面是将Izh离散化的写法
        if v>= thresh:
            v = c
            u = u + d
        '''
        # 初始化膜电势 以及 对应的U
        self.mem = 0.
        self.u = 0.
        self.dt = kwargs['dt'] if 'dt' in kwargs else 1.

    def integral(self, inputs):
        self.mem = self.mem + self.dt * (0.04 * self.mem * self.mem + 5 * self.mem - self.u + 140 + inputs)
        self.u = self.u + self.dt * (self.a * self.b * self.mem - self.a * self.u)

    def calc_spike(self):
        self.spike = self.act_fun(self.mem - self.get_thres())  # 大于阈值释放脉冲
        self.mem = self.mem * (1 - self.spike.detach()) + self.spike.detach() * self.c
        self.u = self.u + self.spike.detach() * self.d

    def n_reset(self):
        self.mem = 0.
        self.u = 0.
        self.spike = 0.


class IzhNodeMU(BaseNode):
    """
    Izhikevich 脉冲神经元多参数版
    :param threshold: 神经元发放脉冲需要达到的阈值
    :param v_reset: 静息电位
    :param dt: 时间步长
    :param step: 仿真步
    :param tau: 膜电位时间常数, 用于控制膜电位衰减
    :param act_fun: 使用surrogate gradient 对梯度进行近似, 默认为 ``surrogate.AtanGrad``
    :param args: 其他的参数
    :param kwargs: 其他的参数
    """

    def __init__(self, threshold=1., tau=2., act_fun=AtanGrad, *args, **kwargs):
        super().__init__(threshold, *args, **kwargs)
        self.tau = tau
        if isinstance(act_fun, str):
            act_fun = eval(act_fun)
        self.act_fun = act_fun(alpha=2., requires_grad=False)
        self.a = kwargs['a'] if 'a' in kwargs else 0.02
        self.b = kwargs['b'] if 'b' in kwargs else 0.2
        self.c = kwargs['c'] if 'c' in kwargs else -55.
        self.d = kwargs['d'] if 'd' in kwargs else -2.
        self.mem = kwargs['mem'] if 'mem' in kwargs else 0.
        self.u = kwargs['u'] if 'u' in kwargs else 0.
        self.dt = kwargs['dt'] if 'dt' in kwargs else 1.

    def integral(self, inputs):
        self.mem = self.mem + self.dt * (0.04 * self.mem * self.mem + 5 * self.mem - self.u + 140 + inputs)
        self.u = self.u + self.dt * (self.a * self.b * self.mem - self.a * self.u)

    def calc_spike(self):
        self.spike = self.act_fun(self.mem - self.threshold)
        self.mem = self.mem * (1 - self.spike.detach()) + self.spike.detach() * self.c
        self.u = self.u + self.spike.detach() * self.d

    def n_reset(self):
        self.mem = -70.
        self.u = 0.
        self.spike = 0.

    def requires_activation(self):
        return False


class DGLIFNode(BaseNode):
    """
    Reference: https://arxiv.org/abs/2110.08858
    :param threshold: 神经元的脉冲发放阈值
    :param tau: 神经元的膜常数, 控制膜电位衰减
    """

    def __init__(self, threshold=.5, tau=2., *args, **kwargs):
        super().__init__(threshold, tau, *args, **kwargs)
        self.act = nn.ReLU()
        self.tau = tau

    def integral(self, inputs):
        inputs = self.act(inputs)
        self.mem = self.mem + ((inputs - self.mem) / self.tau) * self.dt

    def calc_spike(self):
        spike = self.mem.clone()
        spike[(spike < self.get_thres())] = 0.
        # self.spike = spike / (self.mem.detach().clone() + 1e-12)
        self.spike = spike - spike.detach() + \
                     torch.where(spike.detach() > self.get_thres(), torch.ones_like(spike), torch.zeros_like(spike))
        self.spike = spike
        self.mem = torch.where(self.mem >= self.get_thres(), torch.zeros_like(self.mem), self.mem)


class HTDGLIFNode(IFNode):
    """
    Reference: https://arxiv.org/abs/2110.08858
    :param threshold: 神经元的脉冲发放阈值
    :param tau: 神经元的膜常数, 控制膜电位衰减
    """

    def __init__(self, threshold=.5, tau=2., *args, **kwargs):
        super().__init__(threshold, *args, **kwargs)
        self.warm_up = False

    def calc_spike(self):
        spike = self.mem.clone()
        spike[(spike < self.get_thres())] = 0.
        # self.spike = spike / (self.mem.detach().clone() + 1e-12)
        self.spike = spike - spike.detach() + \
                     torch.where(spike.detach() > self.get_thres(), torch.ones_like(spike), torch.zeros_like(spike))
        self.spike = spike
        self.mem = torch.where(self.mem >= self.get_thres(), torch.zeros_like(self.mem), self.mem)
        # self.mem[[(spike > self.get_thres())]] = self.mem[[(spike > self.get_thres())]] - self.get_thres()

        self.mem = (self.mem + 0.2 * self.spike - 0.2 * self.spike.detach()) * self.dt

    def forward(self, inputs):
        if self.warm_up:
            return F.relu(inputs)
        else:
            return super(IFNode, self).forward(F.relu(inputs))


class SimHHNode(BaseNode):
    """
    简单版本的HH模型
    :param threshold: 神经元发放脉冲需要达到的阈值
    :param v_reset: 静息电位
    :param dt: 时间步长
    :param step: 仿真步
    :param tau: 膜电位时间常数, 用于控制膜电位衰减
    :param act_fun: 使用surrogate gradient 对梯度进行近似, 默认为 ``surrogate.AtanGrad``
    :param args: 其他的参数
    :param kwargs: 其他的参数
    """

    def __init__(self, threshold=50., tau=2., act_fun=AtanGrad, *args, **kwargs):
        super().__init__(threshold, *args, **kwargs)
        self.tau = tau
        if isinstance(act_fun, str):
            act_fun = eval(act_fun)
        '''
        I = Cm dV/dt + g_k*n^4*(V_m-V_k) + g_Na*m^3*h*(V_m-V_Na) + g_l*(V_m - V_L)
        '''
        self.act_fun = act_fun(alpha=2., requires_grad=False)
        self.g_Na, self.g_K, self.g_l = torch.tensor(120.), torch.tensor(120), torch.tensor(0.3)  # k 36
        self.V_Na, self.V_K, self.V_l = torch.tensor(120.), torch.tensor(-120.), torch.tensor(10.6)  # k -12
        self.m, self.n, self.h = torch.tensor(0), torch.tensor(0), torch.tensor(0)
        self.mem = 0
        self.dt = 0.01

    def integral(self, inputs):
        self.I_Na = torch.pow(self.m, 3) * self.g_Na * self.h * (self.mem - self.V_Na)
        self.I_K = torch.pow(self.n, 4) * self.g_K * (self.mem - self.V_K)
        self.I_L = self.g_l * (self.mem - self.V_l)
        self.mem = self.mem + self.dt * (inputs - self.I_Na - self.I_K - self.I_L) / 0.02
        # non Na
        # self.mem = self.mem + 0.01 * (inputs -  self.I_K - self.I_L) / 0.02  #decayed
        # NON k
        # self.mem = self.mem + 0.01 * (inputs - self.I_Na - self.I_L) / 0.02  #increase

        self.alpha_n = 0.01 * (self.mem + 10.0) / (1 - torch.exp(-(self.mem + 10.0) / 10))
        self.beta_n = 0.125 * torch.exp(-(self.mem) / 80)

        self.alpha_m = 0.1 * (self.mem + 25) / (1 - torch.exp(-(self.mem + 25) / 10))
        self.beta_m = 4 * torch.exp(-(self.mem) / 18)

        self.alpha_h = 0.07 * torch.exp(-(self.mem) / 20)
        self.beta_h = 1 / (1 + torch.exp(-(self.mem + 30) / 10))

        self.n = self.n + self.dt * (self.alpha_n * (1 - self.n) - self.beta_n * self.n)
        self.m = self.m + self.dt * (self.alpha_m * (1 - self.m) - self.beta_m * self.m)
        self.h = self.h + self.dt * (self.alpha_h * (1 - self.h) - self.beta_h * self.h)

    def calc_spike(self):
        self.spike = self.act_fun(self.mem - self.threshold)
        self.mem = self.mem * (1 - self.spike.detach())

    def forward(self, inputs):
        self.integral(inputs)
        self.calc_spike()
        return self.spike

    def n_reset(self):
        self.mem = 0.
        self.spike = 0.
        self.m, self.n, self.h = torch.tensor(0), torch.tensor(0), torch.tensor(0)

    def requires_activation(self):
        return False


class CTIzhNode(IzhNode):
    def __init__(self, threshold=1., tau=2., act_fun=AtanGrad, *args, **kwargs):
        super().__init__(threshold, tau, act_fun, *args, **kwargs)

        self.name = kwargs['name'] if 'name' in kwargs else ''
        self.excitability = kwargs['excitability'] if 'excitability' in kwargs else 'TRUE'
        self.spikepattern = kwargs['spikepattern'] if 'spikepattern' in kwargs else 'RS'
        self.synnum = kwargs['synnum'] if 'synnum' in kwargs else 0
        self.locationlayer = kwargs['locationlayer'] if 'locationlayer' in kwargs else ''
        self.adjneuronlist = {}
        self.proximal_dendrites = []
        self.distal_dendrites = []
        self.totalindex = kwargs['totalindex'] if 'totalindex' in kwargs else 0
        self.colindex = 0
        self.state = 'inactive'

        self.Gup = kwargs['Gup'] if 'Gup' in kwargs else 0.0
        self.Gdown = kwargs['Gdown'] if 'Gdown' in kwargs else 0.0
        self.Vr = kwargs['Vr'] if 'Vr' in kwargs else 0.0
        self.Vt = kwargs['Vt'] if 'Vt' in kwargs else 0.0
        self.Vpeak = kwargs['Vpeak'] if 'Vpeak' in kwargs else 0.0
        self.capicitance = kwargs['capacitance'] if 'capacitance' in kwargs else 0.0
        self.k = kwargs['k'] if 'k' in kwargs else 0.0
        self.mem = -65
        self.vtmp = -65
        self.u = -13.0
        self.spike = 0
        self.dc = 0

    def integral(self, inputs):
        self.mem += self.dt * (
                self.k * (self.mem - self.Vr) * (self.mem - self.Vt) - self.u + inputs) / self.capicitance
        self.u += self.dt * (self.a * (self.b * (self.mem - self.Vr) - self.u))

    def calc_spike(self):
        if self.mem >= self.Vpeak:
            self.mem = self.c
            self.u = self.u + self.d
            self.spike = 1
            self.spreadMarkPostNeurons()

    def spreadMarkPostNeurons(self):
        for post, list in self.adjneuronlist.items():
            if self.excitability == "TRUE":
                post.dc = random.randint(140, 160)
            else:
                post.dc = random.randint(-160, -140)


class adth(BaseNode):
    """
        The adaptive Exponential Integrate-and-Fire model (aEIF)
        :param args: Other parameters
        :param kwargs: Other parameters
    """

    def __init__(self, *args, **kwargs):
        super().__init__(requires_fp=False, *args, **kwargs)

    def adthNode(self, v, dt, c_m, g_m, alpha_w, ad, Ieff, Ichem, Igap, tau_ad, beta_ad, vt, vm1):
        """
                Calculate the neurons that discharge after the current threshold is reached
                :param v: Current neuron voltage
                :param dt: time step
                :param ad:Adaptive variable
                :param vv:Spike, if the voltage exceeds the threshold from below
        """
        v = v + dt / c_m * (-g_m * v + alpha_w * ad + Ieff + Ichem + Igap)
        ad = ad + dt / tau_ad * (-ad + beta_ad * v)
        vv = (v >= vt).astype(int) * (vm1 < vt).astype(int)
        vm1 = v
        return v, ad, vv, vm1

    def calc_spike(self):
        pass


class HHNode(BaseNode):
    """
    用于脑模拟的HH模型
    p: [threshold, g_Na, g_K, g_l, V_Na, V_K, V_l, C]

    """

    def __init__(self, p, dt, device, act_fun=AtanGrad, *args, **kwargs):
        super().__init__(threshold=p[0], *args, **kwargs)
        if isinstance(act_fun, str):
            act_fun = eval(act_fun)
        '''
        I = Cm dV/dt + g_k*n^4*(V_m-V_k) + g_Na*m^3*h*(V_m-V_Na) + g_l*(V_m - V_L)
        '''
        self.neuron_num = len(p[0])
        self.act_fun = act_fun(alpha=2., requires_grad=False)
        self.tau_I = 3
        self.g_Na = torch.tensor(p[1])
        self.g_K = torch.tensor(p[2])
        self.g_l = torch.tensor(p[3])
        self.V_Na = torch.tensor(p[4])
        self.V_K = torch.tensor(p[5])
        self.V_l = torch.tensor(p[6])
        self.C = torch.tensor(p[7])
        self.m = 0.05 * torch.ones(self.neuron_num, device=device, requires_grad=False)
        self.n = 0.31 * torch.ones(self.neuron_num, device=device, requires_grad=False)
        self.h = 0.59 * torch.ones(self.neuron_num, device=device, requires_grad=False)
        self.v_reset = 0
        self.dt = dt
        self.dt_over_tau = self.dt / self.tau_I
        self.sqrt_coeff = math.sqrt(1 / (2 * (1 / self.dt_over_tau)))
        self.mu = 10
        self.sig = 12

        self.mem = torch.tensor(self.v_reset, device=device, requires_grad=False)
        self.mem_p = self.mem
        self.spike = torch.zeros(self.neuron_num, device=device, requires_grad=False)
        self.Iback = torch.zeros(self.neuron_num, device=device, requires_grad=False)
        self.Ieff = torch.zeros(self.neuron_num, device=device, requires_grad=False)

    def integral(self, inputs):
        self.alpha_n = (0.1 - 0.01 * self.mem) / (torch.exp(1 - 0.1 * self.mem) - 1)
        self.alpha_m = (2.5 - 0.1 * self.mem) / (torch.exp(2.5 - 0.1 * self.mem) - 1)
        self.alpha_h = 0.07 * torch.exp(-self.mem / 20.0)

        self.beta_n = 0.125 * torch.exp(-self.mem / 80.0)
        self.beta_m = 4.0 * torch.exp(-self.mem / 18.0)
        self.beta_h = 1 / (torch.exp(3 - 0.1 * self.mem) + 1)

        self.n = self.n + self.dt * (self.alpha_n * (1 - self.n) - self.beta_n * self.n)
        self.m = self.m + self.dt * (self.alpha_m * (1 - self.m) - self.beta_m * self.m)
        self.h = self.h + self.dt * (self.alpha_h * (1 - self.h) - self.beta_h * self.h)

        self.I_Na = torch.pow(self.m, 3) * self.g_Na * self.h * (self.mem - self.V_Na)
        self.I_K = torch.pow(self.n, 4) * self.g_K * (self.mem - self.V_K)
        self.I_L = self.g_l * (self.mem - self.V_l)

        self.mem_p = self.mem
        self.mem = self.mem + self.dt * (inputs - self.I_Na - self.I_K - self.I_L) / self.C

    def calc_spike(self):
        self.spike = (self.threshold > self.mem_p).float() * (self.mem > self.threshold).float()

    def forward(self, inputs):
        self.integral(inputs)
        self.calc_spike()
        return self.spike, self.mem

    def requires_activation(self):
        return False


class aEIF(BaseNode):
    """
        The adaptive Exponential Integrate-and-Fire model (aEIF)
        This class define the membrane, spike, current and parameters of a neuron group of a specific type
        :param args: Other parameters
        :param kwargs: Other parameters
    """

    def __init__(self, p, dt, device, *args, **kwargs):
        """
            p:[threshold, v_reset, c_m, tao_w, alpha_ad, beta_ad]

        """
        super().__init__(threshold=p[0], requires_fp=False, *args, **kwargs)
        self.neuron_num = len(p[0])
        self.g_m = 0.1  # neuron conduction
        self.dt = dt
        self.tau_I = 3  # Time constant to filter the synaptic inputs
        self.Delta_T = 0.5  # parameter
        self.v_reset = p[1]  # membrane potential reset to v_reset after fire spike
        self.c_m = p[2]
        self.tau_w = p[3]  # Time constant of adaption coupling
        self.alpha_ad = p[4]
        self.beta_ad = p[5]
        self.refrac = 5 / self.dt  # refractory period
        self.dt_over_tau = self.dt / self.tau_I
        self.sqrt_coeff = math.sqrt(1 / (2 * (1 / self.dt_over_tau)))
        self.mem = self.v_reset
        self.spike = torch.zeros(self.neuron_num, device=device, requires_grad=False)
        self.ad = torch.zeros(self.neuron_num, device=device, requires_grad=False)
        self.ref = torch.randint(0, int(self.refrac + 1), (1, self.neuron_num), device=device, requires_grad=False).squeeze(
            0)  # refractory counter
        self.ref = self.ref.float()
        self.mu = 10
        self.sig = 12
        self.Iback = torch.zeros(self.neuron_num, device=device, requires_grad=False)
        self.Ieff = torch.zeros(self.neuron_num, device=device, requires_grad=False)

    def integral(self, inputs):

        self.mem = self.mem + (self.ref > self.refrac) * self.dt / self.c_m * \
                   (-self.g_m * (self.mem - self.v_reset) + self.g_m * self.Delta_T *
                    torch.exp((self.mem - self.threshold) / self.Delta_T) +
                    self.alpha_ad * self.ad + inputs)

        self.ad = self.ad + (self.ref > self.refrac) * self.dt / self.tau_w * \
                  (-self.ad + self.beta_ad * (self.mem - self.v_reset))

    def calc_spike(self):
        self.spike = (self.mem > self.threshold).float()
        self.ref = self.ref * (1 - self.spike) + 1
        self.ad = self.ad + self.spike * 30
        self.mem = self.spike * self.v_reset + (1 - self.spike.detach()) * self.mem

    def forward(self, inputs):

        # aeifnode_cuda.forward(self.threshold, self.c_m, self.alpha_w, self.beta_ad, inputs, self.ref, self.ad, self.mem, self.spike)
        self.integral(inputs)
        self.calc_spike()

        return self.spike, self.mem

class LIAFNode(BaseNode):
    """
    Leaky Integrate and Analog Fire (LIAF), Reference: https://ieeexplore.ieee.org/abstract/document/9429228
    与LIF相同, 但前传的是膜电势, 更新沿用阈值和膜电势
    :param act_fun: 前传使用的激活函数 [ReLU, SeLU, LeakyReLU]
    :param threshold_related: 阈值依赖模式，若为"True"则 self.spike = act_fun(mem-threshold)
    :note that BaseNode return self.spike, and here self.spike is analog value.
    """
    def __init__(self, spike_act=BackEIGateGrad(), act_fun="SELU", threshold=0.5, tau=2., threshold_related=True, *args, **kwargs):
        super().__init__(threshold, *args, **kwargs)
        if isinstance(act_fun, str):
            act_fun = eval("nn." + act_fun + "()")
        self.tau = tau
        self.act_fun = act_fun
        self.spike_act = spike_act
        self.threshold_related = threshold_related

    def integral(self, inputs):
        self.mem = self.mem + (inputs - self.mem) / self.tau

    def calc_spike(self):
        if self.threshold_related:
            spike_tmp = self.act_fun(self.mem - self.threshold)
        else:
            spike_tmp = self.act_fun(self.mem)
        self.spike = self.spike_act(self.mem - self.threshold)
        self.mem = self.mem * (1 - self.spike)
        self.spike = spike_tmp



class OnlineLIFNode(BaseNode):
    """
    Online-update Leaky Integrate and Fire
    与LIF模型相同，但是时序信息在反传时从计算图剥离，因此可以实现在线的更新；模型占用显存固定，不随仿真步step线性提升。
    使用此神经元需要修改:  1. 将模型中t次forward从model_zoo写到main.py中
                       2. 在Conv层与OnelineLIFNode层中加入Replace函数，即时序前传都是detach的，但仍计算该层空间梯度信息。
                       3. 网络结构不适用BN层，使用weight standardization
    注意该神经元不同于OTTT，而是将时序信息全部扔弃。对应这篇文章：https://arxiv.org/abs/2302.14311
    若需保留时序，需要对self.rate_tracking进行计算。实现可参考https://github.com/pkuxmq/OTTT-SNN
    """

    def __init__(self, threshold=0.5, tau=2., act_fun=QGateGrad, init=False, *args, **kwargs):
        super().__init__(threshold, *args, **kwargs)
        self.tau = tau
        if isinstance(act_fun, str):
            act_fun = eval(act_fun)
        self.act_fun = act_fun(alpha=2., requires_grad=False)
        self.rate_tracking = None
        self.init = True


    def integral(self, inputs):
        if self.init is True:
            self.mem = torch.zeros_like(inputs)
            self.init = False
        self.mem = self.mem.detach() + (inputs - self.mem.detach()) / self.tau

    def calc_spike(self):
        self.spike = self.act_fun(self.mem - self.threshold)
        self.mem = self.mem * (1 - self.spike.detach())
        with torch.no_grad():
            if self.rate_tracking == None:
                self.rate_tracking = self.spike.clone().detach()
        self.spike = torch.cat((self.spike, self.rate_tracking), dim=0)

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cv2
import numpy as np
from copy import deepcopy
import os, time, math,random
from sklearn.metrics import confusion_matrix
from torch.profiler import profile, record_function, ProfilerActivity
from thop import clever_format
from vprof import runner




seed = 0
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)

torch.cuda.manual_seed(seed) #GPU随机种子确定

torch.backends.cudnn.benchmark = False #模型卷积层预先优化关闭
torch.backends.cudnn.deterministic = True #确定为默认卷积算法

random.seed(seed)

os.environ["PYTHONHASHSEED"] = str(seed)

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
dev = "cuda"
device = "cpu"#torch.device(dev) if torch.cuda.is_available() else 'cpu'
torch.set_printoptions(precision=4, sci_mode=False)


# ===========================================================================================================

convoff = 0.3


# avgscale = 5


class STDPConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride, padding,groups,
                 tau_decay=torch.exp(-1.0 / torch.tensor(100.0)), offset=convoff, static=True, inh=6.5, avgscale=5):
        super().__init__()
        self.tau_decay = tau_decay
        self.offset = offset
        self.static = static
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,groups=groups,
                              bias=False)
        self.avgpool = nn.AvgPool2d(kernel_size=kernel_size, stride=stride, padding=padding)
        self.mem = self.spike = self.refrac_count = None
        self.normweight()
        self.inh = inh
        self.avgscale = avgscale
        self.onespike=True
        self.node=LIFSTDPNode(act_fun=STDPGrad,tau=tau_decay,mem_detach=True)
        self.WTA=WTALayer( )
        self.lateralinh=LateralInhibition(self.node,self.inh,mode="threshold")

    def mem_update(self, x, onespike=True):  # b,c,h,w

        x=self.node( x)

        if x.max() > 0:
            x=self.WTA(x)

            self.lateralinh(x)

        self.spike= x
        return self.spike

    def forward(self, x, T=None, onespike=True):

        if not self.static:
            batch, T, c, h, w = x.shape
            x = x.reshape(-1, c, h, w)

        x = self.conv(  x)

        n = self.getthresh(x)
        self.node.threshold.data = n

        x=x.clamp(min=0)
        x = n / (1 + torch.exp(-(x - 4 * n / 10) * (8 / n)))

        if not self.static:
            x = x.reshape(batch, T, c, h, w)
            xsum = None
            for i in range(T):
                tmp = self.mem_update(x[:, i], onespike).unsqueeze(1)
                if xsum is not None:
                    xsum = torch.cat([xsum, tmp], 1)
                else:
                    xsum = tmp
        else:
            xsum = 0
            for i in range(T):
                xsum += self.mem_update(x, onespike)

        return xsum

    def reset(self):
        #self.mem = self.spike = self.refrac_count = None
        self.node.n_reset()
    def normgrad(self, force=False):
        if force:
            min = self.conv.weight.grad.data.min(1, True)[0].min(2, True)[0].min(3, True)[0]
            max = self.conv.weight.grad.data.min(1, True)[0].max(2, True)[0].max(3, True)[0]
            self.conv.weight.grad.data -= min
            tmp = self.offset * max
        else:
            tmp = self.offset * self.spike.mean(0, True).mean(2, True).mean(3, True).permute(1, 0, 2, 3)
        self.conv.weight.grad.data -= tmp
        self.conv.weight.grad.data = -self.conv.weight.grad.data

    def normweight(self, clip=False):
        if clip:
            self.conv.weight.data = torch. \
                clamp(self.conv.weight.data, min=-3, max=1.0)
        else:
            c, i, w, h = self.conv.weight.data.shape

            avg=self.conv.weight.data.mean(1, True).mean(2, True).mean(3, True)
            self.conv.weight.data -=avg

            tmp = self.conv.weight.data.reshape(c, 1, -1, 1)

            self.conv.weight.data /= tmp.std(2, unbiased=False, keepdim=True)


    def getthresh(self, scale):

        tmp2= scale.max(0, True)[0].max(2, True)[0].max(3, True)[0]+0.0001

        return tmp2


class STDPLinear(nn.Module):
    def __init__(self, in_planes, out_planes,
                 tau_decay=0.99, offset=0.05, static=True,inh=10):
        super().__init__()
        self.tau_decay = tau_decay
        self.offset = offset
        self.static = static
        self.linear = nn.Linear(in_planes, out_planes, bias=False)
        self.mem = self.spike = self.refrac_count = None
        # torch.nn.init.xavier_uniform_(self.linear.weight, gain=1)
        self.normweight(False)
        self.threshold = torch.ones(out_planes, device=device) *20

        self.inh=inh
        self.node=LIFSTDPNode(act_fun=STDPGrad,tau=tau_decay  ,mem_detach=True)
        self.WTA=WTALayer( )
        self.lateralinh=LateralInhibition(self.node,self.inh,mode="max")
        self.init=False
    def mem_update(self, x, onespike=True):  # b,c,h,w
        if not self.init:
            self.node.threshold.data= (x.max(0)[0].detach()*3).to(device)
            self.init=True

        xori=x
        x=self.node( x)
        if x.max() > 0:
            x=self.WTA(x)

            self.lateralinh(x,xori)

        self.spike=x
        return self.spike

    def forward(self, x, T, onespike=True):

        if not self.static:
            batch, T, w = x.shape
            x = x.reshape(-1, w)
        x = x.detach()



        x = self.linear(x)
        self.x=x.detach()

        if not self.static:
            x = x.reshape(batch, T, w)
            xsum = None
            for i in range(T):
                tmp = self.mem_update(x[:, i], onespike).unsqueeze(1)
                if xsum is not None:
                    xsum = torch.cat([xsum, tmp], 1)
                else:
                    xsum = tmp
        else:
            xsum = 0
            for i in range(T):
                xsum += self.mem_update(x, onespike)
        #print(xsum.mean())
        return xsum

    def reset(self):

        self.node.n_reset()
    def normgrad(self, force=False):
        if force:

            pass
        else:
            tmp = self.offset * self.spike.mean(0, True).permute(1, 0)


        self.linear.weight.grad.data = -self.linear.weight.grad.data


    def normweight(self, clip=False):

        if clip:
            self.linear.weight.data = torch. \
                clamp(self.linear.weight.data, min=0, max=1.0)
        else:
            self.linear.weight.data = torch. \
                clamp(self.linear.weight.data, min=0, max=1.0)
            sumweight = self.linear.weight.data.sum(1, True)
            sumweight += (~(sumweight.bool())).float()
            # self.linear.weight.data *= 11.76  / sumweight
            self.linear.weight.data /= self.linear.weight.data.max(1, True)[0] / 0.1

    def getthresh(self, scale):
        tmp = self.linear.weight.clamp(min=0) * scale
        tmp2 = tmp.sum(1, True).reshape(1, -1)
        return tmp2

    def updatethresh(self, plus=0.05):

        self.node.threshold += (plus*self.x * self.spike.detach()).sum(0)
        tmp=self.node.threshold.max()-350
        if tmp>0:
            self.node.threshold-=tmp

class STDPFlatten(nn.Module):
    def __init__(self, start_dim=0, end_dim=-1):
        super().__init__()
        self.flatten = nn.Flatten(start_dim=start_dim, end_dim=end_dim)

    def forward(self, x, T):  # [batch,T,c,w,h]

        return self.flatten(x)


class STDPMaxPool(nn.Module):
    def __init__(self, kernel_size, stride, padding, static=True):
        super().__init__()
        self.static = static
        self.pool = nn.MaxPool2d(kernel_size, stride, padding)

    def forward(self, x, T):  # [batch,T,c,w,h]

        if not self.static:
            batch, T, c, h, w = x.shape
            x = x.reshape(-1, c, h, w)
        x = self.pool(x)
        if not self.static:
            x = x.reshape(batch, T, c, h, w)

        return x


alpha = 1.0


class Normliaze(nn.Module):
    def __init__(self, static=True):
        super().__init__()
        self.static = static

    def forward(self, x, T):  # [batch,T,c,w,h]
        # print(x.shape)
        x /= x.max(1, True)[0].max(2, True)[0].max(3, True)[0]
        # x/=x.mean()/0.13

        return x


class voting(nn.Module):

    def __init__(self, shape):
        super().__init__()
        self.label = torch.zeros(shape) - 1
        self.assignments=0
    def assign_labels(self, spikes, labels, rates=None, n_labels=10, alpha=alpha):
        # 根据最后一层的spikes 以及 label 对于最后一层的神经元赋予不同的label
        # spikes 是 batch * time * in_size
        # print(spikes.size())
        n_neurons = spikes.size(2)
        if rates is None:
            rates = torch.zeros(n_neurons, n_labels, device=device)
        self.n_labels = n_labels
        spikes = spikes.cpu().sum(1).to(device)

        for i in range(n_labels):
            n_labeled = torch.sum(labels == i).float()
            # 就是说上一次assign label计算的rates 拿过来滑动平均一下   #这里似乎可以改
            if n_labeled > 0:
                indices = torch.nonzero(labels == i).view(-1)
                tmp = torch.sum(spikes[indices], 0) / n_labeled  # 平均脉冲数
                rates[:, i] = alpha * rates[:, i] + tmp

        # 此时的rates是 in_size * n_label, 对应哪个label的rates最高 该神经元就对应着该label
        self.assignments = torch.max(rates, 1)[1]
        return self.assignments, rates

    def get_label(self, spikes):
        # 根据最后一层的spike 计算得到label
        n_samples = spikes.size(0)
        spikes = spikes.cpu().sum(1).to(device)
        rates = torch.zeros(n_samples, self.n_labels, device=device)

        for i in range(self.n_labels):
            n_assigns = torch.sum(self.assignments == i).float()  # 共有多少个该类别节点
            if n_assigns > 0:
                indices = torch.nonzero(self.assignments == i).view(-1)  # 找到该类别节点位置
                rates[:, i] = torch.sum(spikes[:, indices], 1) / n_assigns  # 该类别平均所有该类别节点发放脉冲数

        return torch.sort(rates, dim=1, descending=True)[1][:, 0]

inh=25
inh2=1.625
channel=12
neuron=6400
class Conv_Net(nn.Module):
    def __init__(self):
        super(Conv_Net, self).__init__()
        self.conv = nn.ModuleList([
            STDPConv(1, channel, 3, 1, 1,1, static=True, inh=1.625, avgscale=5 ),
            STDPMaxPool(2, 2, 0, static=True),
            Normliaze(),
            #STDPConv(12, 48, 3, 1, 1,1, static=True, inh=inh2, avgscale=10 ),
            #STDPMaxPool(2, 2, 0, static=True),
            #Normliaze(),

            STDPFlatten(start_dim=1),
            STDPLinear(196*channel, neuron, static=True,inh=inh)


        ])

        self.voting = voting(10)

    def forward(self, x, inlayer, outlayer, T, onespike=True):  # [b,t,w,h]

        for i in range(inlayer, outlayer + 1):
            x = self.conv[i](x, T)
        return x

    def normgrad(self, layer, force=False):
        self.conv[layer].normgrad(force)

    def normweight(self, layer, clip=False):
        self.conv[layer].normweight(clip)

    def updatethresh(self, layer, plus=0.05):
        self.conv[layer].updatethresh(plus)

    def reset(self, layer):
        if isinstance(layer, list):
            for i in layer:
                self.conv[i].reset()
        else:
            self.conv[layer].reset()




def plot_confusion_matrix(cm, classes, normalize=True, title='Test Confusion matrix', cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        #print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
    plt.figure()
    #print(cm)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    # plt.xticks(tick_marks, classes, rotation=45)
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.

    for i in range(cm.shape[0]):
        plt.text(i, i, format(cm[i, i], fmt), horizontalalignment="center",
                 color="white" if cm[i, i] > thresh else "black")
    plt.tight_layout()
    #plt.savefig('confusestpf2'+str(channel)+"_n"+str(neuron)+".pdf")
    #plt.show()
#if __name__ == '__main__':


def trainAndTest():
    batch_size = 1024
    T = 100
    transform = transforms.Compose(
        [transforms.Resize((28, 28)), transforms.Grayscale(num_output_channels=1), transforms.ToTensor()])
    transform = transforms.Compose([transforms.ToTensor()])
    # mnist_train = datasets.CIFAR10(root='/data/datasets/CIFAR10/', train=True, download=False, transform=transform )
    # mnist_test = datasets.CIFAR10(root='/data/datasets/CIFAR10/', train=False, download=False, transform=transform )
    #mnist_train = datasets.FashionMNIST(root='/data/dyt//', train=True, download=True, transform=transform )
    #mnist_test = datasets.FashionMNIST(root='/data/dyt/', train=False, download=False, transform=transform )
    mnist_train = datasets.MNIST(root='./', train=True, download=True, transform=transform)
    mnist_test = datasets.MNIST(root='./', train=False, download=False, transform=transform)
    train_iter = torch.utils.data.DataLoader(mnist_train, batch_size=batch_size, shuffle=True, num_workers=1)
    test_iter = torch.utils.data.DataLoader(mnist_test, batch_size=batch_size, shuffle=False, num_workers=1)

    print("data loaded")
    
    model = Conv_Net().to(device)
    convlist = [index for index, i in enumerate(model.conv) if isinstance(i, (STDPConv, STDPLinear))]

    print("model instance made")
    #cap = torch.ones([100000, 1000, 30], device=device)

    print("Init training")
    for layer in range(len(convlist) - 1):
        optimizer = torch.optim.SGD(list(model.parameters())[layer:layer + 1], lr=0.1)
        print("optimizer chosen for layer: ", layer)
        for epoch in range(3):
            print("init image loading")
            for step, (x, y) in enumerate(train_iter):
                #print("image loaded")
                x = x.to(device)
                y = y.to(device)
                #print("Will now load spikes to model")
                spikes = model(x, 0, convlist[layer], T)
                #print("spikes loaded")
                optimizer.zero_grad()
                #print("optimizer used")
                spikes.sum().backward(torch.tensor(1/  (spikes.shape[0] * spikes.shape[2] * spikes.shape[3])))
                #print("spikes summed")
                # spikes.sum().backward(  )
                model.conv[convlist[layer]].spike = spikes.detach()
                #print("model spikes updated")
                #print("count spikes attempt: ", torch.count_nonzero(model.conv[convlist[layer]].spike))
                model.normgrad(convlist[layer], force=True)
                #print("normalization of gradient done")
                optimizer.step()
                #print("Took a step with optimizer")
                model.normweight(convlist[layer], clip=False)
                #print("done with normalizing weights")
                # print(model.conv[convlist[layer]].conv.weight.data )
                model.reset(convlist)
                #print("model has been reset")


            print("layer", layer, "epoch", epoch, 'Done')
        #model.conv[convlist[layer]].onespike=False
    # ===========================================================================================================
    # linear
    #model.conv[convlist[-2]].onespike=True

    cap = None
    batch_size = 1024 #512
    
    T = 200
    layer = len(convlist) - 1
    plus = 0.002
    lr = 0.0001
    train_iter = torch.utils.data.DataLoader(mnist_train, batch_size=batch_size, shuffle=True, num_workers=1)
    test_iter = torch.utils.data.DataLoader(mnist_test, batch_size=batch_size, shuffle=False, num_workers=1)
    optimizer = torch.optim.SGD(list(model.parameters())[layer:], lr=lr)

    rates = None
    earlyStopping = 15
    counter = 0
    best = 0
    accrecord=[]
    for epoch in range(1000):
        spikefull = None
        labelfull = None
        for step, (x, y) in enumerate(train_iter):
            x = x.to(device)
            y = y.to(device)

            spiketime = 0

            spikes = model(x, 0, convlist[layer], T)
            # print(spikes.mean())
            optimizer.zero_grad()
            spikes.sum().backward()
            model.conv[convlist[layer]].spike = spikes.detach()
            model.normgrad(convlist[layer], force=False)
            optimizer.step()
            model.updatethresh(convlist[layer], plus=plus)
            model.normweight(convlist[layer], clip=False)

            spikes = spikes.reshape(spikes.shape[0], 1, -1).detach()
            if spikefull is None:
                spikefull = spikes
                #print("spikefull: ", spikefull)
                labelfull = y
            else:
                spikefull = torch.cat([spikefull, spikes], 0)
                labelfull = torch.cat([labelfull, y], 0)

            model.reset(convlist)

        _, rates = model.voting.assign_labels(spikefull, labelfull, rates)
        rates = rates.detach() * 0.5
        result = model.voting.get_label(spikefull)
        acc = (result == labelfull).float().mean()

        print(epoch, acc, 'channel', channel, "n", neuron)
        print(model.conv[-1].node.threshold.max(),model.conv[-1].node.threshold.mean(),model.conv[-1].node.threshold.min())

        # model.conv[-1].threshold*=0.98
        spikefull = None
        labelfull = None
        result = None

        for step2, (x, y) in enumerate(test_iter):
            x = x.to(device) # Data
            y = y.to(device) # Labels

            """
            -> Iterate through the test set

            -> Fetch all spikes from the model given the input image

            -> add spikes to an array called spikefull which hold which spikes fired at the output layer for each image in the
               test set

            -> Results are from comparing the voting method compared to the labels.
            """
            #with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
            #  with record_function("model_inference"):
            #      model(x, 0, convlist[layer], T) #This gives the profiling for one inference operation on a picture x.
            #print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))

            #with profile(activities=[ProfilerActivity.CPU], profile_memory=True, record_shapes=True) as prof:
            #  model(x, 0, convlist[layer], T) #This gives the profiling for one inference operation on a picture x.

            #print(prof.key_averages().table(sort_by="self_cpu_memory_usage", row_limit=10))


            #macs, params = profile(model, inputs=(x, 0, convlist[layer], T))
            #macs, params = clever_format([macs, params], "%.3f")
            #print(macs,params)

            spiketime = 0
           
            with torch.profiler.profile(
                    activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
                    #schedule=torch.profiler.schedule(wait=1, warmup=1, active=1),
                    on_trace_ready=torch.profiler.tensorboard_trace_handler('./logs/BrainCogNetworkProfiled'),
                    record_shapes=True,
                    profile_memory=True,
                    #with_stack=True,
                    with_flops=True,
                    #with_modules=True
            ) as prof:
                with torch.no_grad():
                    spikes = model(x, 0, convlist[layer], T)

            print(prof.key_averages().table(sort_by="self_cpu_memory_usage", row_limit=10))

            spikes = spikes.reshape(spikes.shape[0], 1, -1).detach()

            with torch.no_grad():
                if spikefull is None:
                    spikefull = spikes
                    labelfull = y

                else:
                    spikefull = torch.cat([spikefull, spikes], 0)
                    labelfull = torch.cat([labelfull, y], 0)

            model.reset(convlist)

        result = model.voting.get_label(spikefull)
        acc = (result == labelfull).float().mean()
        if best < acc:
            best = acc
            torch.save( model, "modelftstp28_350_c"+str(channel)+"_n"+str(neuron)+"_p"+str(acc)+".pth")
            classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
            counter = 0
        else:
          counter += 1


        cm = confusion_matrix(labelfull.cpu(), result.cpu())
        plot_confusion_matrix(cm, classes)
        print("test", acc, "best", best)
        accrecord.append(acc)
        if(counter >= earlyStopping):

          return
        #torch.save(accrecord,"accfstp28_350_c"+str(channel)+"_n"+str(neuron)+".pth")
trainAndTest()

