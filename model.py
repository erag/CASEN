import os

import paddle
import paddle.nn as nn

from layers import AutoEncoderHandwritten


class CASEN(nn.Layer):
    def __init__(self, n_input, cluster):
        super(CASEN, self).__init__()
        self.ae = AutoEncoderHandwritten(n_input, cluster)

    # def _pretrain(self, path):
    #     """
    #     pretrain auto encoder
    #     if done, load pretrain model
    #     :return:
    #     """
    #     if os.path.exists(path):
    #         paddle.fluid.io.load_persistables(path)

    def forward(self, x, cluster, edges, graph=None):
        h, decoder = self.ae(x, cluster)
        pass
