import paddle
import paddle.nn as nn
from paddle.nn import Linear
import paddle.nn.functional as F


class AutoEncoderHandwritten(nn.Layer):
    def __init__(self, n_input, n_z, n_stack=2, delta=0.8, hidden=1500):
        """
        AutoEncoder for Handwritten dataset
        :param n_input:
        :param n_z:
        :param n_stack:
        :param delta:
        :param hidden:
        """
        super(AutoEncoderHandwritten, self).__init__()
        # encoder for view1-5
        self.enc0_1, self.enc0_2, self.enc0_3, self.z0_layer = Linear(0, 0), Linear(0, 0), Linear(0, 0), Linear(0, 0)
        view_encoder0 = [self.enc0_1, self.enc0_2, self.enc0_3, self.z0_layer]
        self.enc1_1, self.enc1_2, self.enc1_3, self.z1_layer = Linear(0, 0), Linear(0, 0), Linear(0, 0), Linear(0, 0)
        view_encoder1 = [self.enc1_1, self.enc1_2, self.enc1_3, self.z1_layer]
        self.enc2_1, self.enc2_2, self.enc2_3, self.z2_layer = Linear(0, 0), Linear(0, 0), Linear(0, 0), Linear(0, 0)
        view_encoder2 = [self.enc2_1, self.enc2_2, self.enc2_3, self.z2_layer]
        self.enc3_1, self.enc3_2, self.enc3_3, self.z3_layer = Linear(0, 0), Linear(0, 0), Linear(0, 0), Linear(0, 0)
        view_encoder3 = [self.enc3_1, self.enc3_2, self.enc3_3, self.z3_layer]
        self.enc4_1, self.enc4_2, self.enc4_3, self.z4_layer = Linear(0, 0), Linear(0, 0), Linear(0, 0), Linear(0, 0)
        view_encoder4 = [self.enc4_1, self.enc4_2, self.enc4_3, self.z4_layer]

        # decoder for view1-5
        self.dec0_1, self.dec0_2, self.dec0_3, self.dec0_4, self.x_0_bar_layer = Linear(0, 0), Linear(0, 0), Linear(0,
                                                                                                                    0), Linear(
            0, 0), Linear(0, 0)
        view_decoder0 = [self.dec0_1, self.dec0_2, self.dec0_3, self.dec0_4, self.x_0_bar_layer]
        self.dec1_1, self.dec1_2, self.dec1_3, self.dec1_4, self.x_1_bar_layer = Linear(0, 0), Linear(0, 0), Linear(0,
                                                                                                                    0), Linear(
            0, 0), Linear(0, 0)
        view_decoder1 = [self.dec1_1, self.dec1_2, self.dec1_3, self.dec1_4, self.x_1_bar_layer]
        self.dec2_1, self.dec2_2, self.dec2_3, self.dec2_4, self.x_2_bar_layer = Linear(0, 0), Linear(0, 0), Linear(0,
                                                                                                                    0), Linear(
            0, 0), Linear(0, 0)
        view_decoder2 = [self.dec2_1, self.dec2_2, self.dec2_3, self.dec2_4, self.x_2_bar_layer]
        self.dec3_1, self.dec3_2, self.dec3_3, self.dec3_4, self.x_3_bar_layer = Linear(0, 0), Linear(0, 0), Linear(0,
                                                                                                                    0), Linear(
            0, 0), Linear(0, 0)
        view_decoder3 = [self.dec3_1, self.dec3_2, self.dec3_3, self.dec3_4, self.x_3_bar_layer]
        self.dec4_1, self.dec4_2, self.dec4_3, self.dec4_4, self.x_4_bar_layer = Linear(0, 0), Linear(0, 0), Linear(0,
                                                                                                                    0), Linear(
            0, 0), Linear(0, 0)
        view_decoder4 = [self.dec4_1, self.dec4_2, self.dec4_3, self.dec4_4, self.x_4_bar_layer]

        view1, view2, view3, view4, view5 = [], [], [], [], []
        views = [view1, view2, view3, view4, view5]
        view_encoder = [view_encoder0, view_encoder1, view_encoder2, view_encoder3, view_encoder4]
        view_decoder = [view_decoder0, view_decoder1, view_decoder2, view_decoder3, view_decoder4]
        for i, view in enumerate(views):
            for j in range(n_stack):
                view.append(int(round(n_input[i] * delta)))
            view.append(int(hidden))
        for i, view in enumerate(views):
            view_encoder[i][0] = Linear(n_input[0], views[i][0])
            view_encoder[i][1] = Linear(views[i][0], views[i][1])
            view_encoder[i][2] = Linear(views[i][1], views[i][2])
            view_encoder[i][3] = Linear(views[i][2], n_z)
            view_decoder[i][0] = Linear(n_z, n_z)
            view_decoder[i][1] = Linear(n_z, views[i][2])
            view_decoder[i][2] = Linear(views[i][2], views[i][1])
            view_decoder[i][3] = Linear(views[i][1], views[i][0])
            view_decoder[i][4] = Linear(views[i][0], n_input[i])

    def forward(self, x, weight):
        x_0 = x[0]
        x_1 = x[1]
        x_2 = x[2]
        x_3 = x[3]
        x_4 = x[4]
        # encoder0
        enc0_h1 = F.relu(self.enc0_1(x_0))
        enc0_h2 = F.relu(self.enc0_2(enc0_h1))
        enc0_h3 = F.relu(self.enc0_3(enc0_h2))
        z0 = self.z0_layer(enc0_h3)
        # encoder1
        enc1_h1 = F.relu(self.enc1_1(x_1))
        enc1_h2 = F.relu(self.enc1_2(enc1_h1))
        enc1_h3 = F.relu(self.enc1_3(enc1_h2))
        z1 = self.z1_layer(enc1_h3)
        # encoder2
        enc2_h1 = F.relu(self.enc2_1(x_2))
        enc2_h2 = F.relu(self.enc2_2(enc2_h1))
        enc2_h3 = F.relu(self.enc2_3(enc2_h2))
        z2 = self.z2_layer(enc2_h3)
        # encoder3
        enc3_h1 = F.relu(self.enc3_1(x_3))
        enc3_h2 = F.relu(self.enc3_2(enc3_h1))
        enc3_h3 = F.relu(self.enc3_3(enc3_h2))
        z3 = self.z3_layer(enc3_h3)
        # encoder4
        enc4_h1 = F.relu(self.enc4_1(x_4))
        enc4_h2 = F.relu(self.enc4_2(enc4_h1))
        enc4_h3 = F.relu(self.enc4_3(enc4_h2))
        z4 = self.z4_layer(enc4_h3)
        aggr = paddle.diag(weight[:, 0]).mm(z0) + paddle.diag(weight[:, 1]).mm(z1) + paddle.diag(weight[:, 2]).mm(
            z2) + paddle.diag(weight[:, 3]).mm(z3) + paddle.diag(weight[:, 4]).mm(z4)
        wei = 1 / paddle.sum(weight, 1)
        z = paddle.diag(wei).mm(aggr)

        # decoder0
        r0 = F.relu(self.dec0_0(z))
        dec0_h1 = F.relu(self.dec0_1(r0))
        dec0_h2 = F.relu(self.dec0_2(dec0_h1))
        dec0_h3 = F.relu(self.dec0_3(dec0_h2))
        x_0_bar = self.x_0_bar_layer(dec0_h3)
        # decoder1
        r1 = F.relu(self.dec1_0(z))
        dec1_h1 = F.relu(self.dec1_1(r1))
        dec1_h2 = F.relu(self.dec1_2(dec1_h1))
        dec1_h3 = F.relu(self.dec1_3(dec1_h2))
        x_1_bar = self.x_1_bar_layer(dec1_h3)
        # decoder2
        r2 = F.relu(self.dec2_0(z))
        dec2_h1 = F.relu(self.dec2_1(r2))
        dec2_h2 = F.relu(self.dec2_2(dec2_h1))
        dec2_h3 = F.relu(self.dec2_3(dec2_h2))
        x_2_bar = self.x_2_bar_layer(dec2_h3)
        # decoder3
        r3 = F.relu(self.dec3_0(z))
        dec3_h1 = F.relu(self.dec3_1(r3))
        dec3_h2 = F.relu(self.dec3_2(dec3_h1))
        dec3_h3 = F.relu(self.dec3_3(dec3_h2))
        x_3_bar = self.x_3_bar_layer(dec3_h3)
        # decoder4
        r4 = F.relu(self.dec4_0(z))
        dec4_h1 = F.relu(self.dec4_1(r4))
        dec4_h2 = F.relu(self.dec4_2(dec4_h1))
        dec4_h3 = F.relu(self.dec4_3(dec4_h2))
        x_4_bar = self.x_4_bar_layer(dec4_h3)
        dec = [x_0_bar, x_1_bar, x_2_bar, x_3_bar, x_4_bar]
        return z, dec
