import torch
import torch.nn as nn


class FeaturePyramidNetwork(nn.Module):
    def __init__(self):
        super(FeaturePyramidNetwork, self).__init__()
        self.name = "AEFPNC"
        self.bn0 = nn.BatchNorm2d(3)
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)  # 3x256x256 -> 16x256x256
        self.bn1 = nn.BatchNorm2d(num_features=16)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)  # 16x256x256  -> 32x256x256
        self.bn2 = nn.BatchNorm2d(num_features=32)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)  # 32x256x256 -> 64x256x256
        self.bn3 = nn.BatchNorm2d(num_features=64)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=48, kernel_size=3, padding=1)  # 64x256x256 -> 48x256x256
        self.bn4 = nn.BatchNorm2d(num_features=48)
        self.conv5 = nn.Conv2d(in_channels=48, out_channels=32, kernel_size=3, padding=1)  # 48x256x256 -> 32x256x256
        self.bn5 = nn.BatchNorm2d(num_features=32)
        self.conv6 = nn.Conv2d(in_channels=32, out_channels=24, kernel_size=3, padding=1)  # 32x256x256 -> 24x256x256
        self.bn6 = nn.BatchNorm2d(num_features=24)
        self.conv7 = nn.Conv2d(in_channels=24, out_channels=16, kernel_size=3, padding=1)  # 24x256x256 -> 16x256x256
        self.bn7 = nn.BatchNorm2d(num_features=16)
        self.down = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, stride=2)
        self.conv_smooth1 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, padding=1)
        self.conv_smooth2 = nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, padding=1)
        self.conv_smooth3 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.convtrans1 = nn.ConvTranspose2d(in_channels=16, out_channels=24, kernel_size=3, padding=1)
        self.convtrans2 = nn.ConvTranspose2d(in_channels=24, out_channels=32, kernel_size=3, padding=1)
        self.convtrans3 = nn.ConvTranspose2d(in_channels=32, out_channels=48, kernel_size=3, padding=1)
        self.convtrans4 = nn.ConvTranspose2d(in_channels=48, out_channels=64, kernel_size=3, padding=1)
        self.convtrans5 = nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=3, padding=1)
        self.convtrans6 = nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=3, padding=1)
        self.convtrans7 = nn.ConvTranspose2d(in_channels=16, out_channels=3, kernel_size=3, padding=1)

    def forward(self, x):
        # ============ Encoder ===========
        # ====== Bottom Up Layers =====
        x = self.bn0(x)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.bn1(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.bn2(x)
        x = self.conv3(x)
        x = self.relu(x)
        x = self.bn3(x)
        _, _, H1, W1 = x.size()
        # ======= Branch network ======
        x_d1 = self.down(x)  # 128x128
        _, _, H2, W2 = x_d1.size()
        x_d2 = self.down(x_d1)  # 64x64
        _, _, H3, W3 = x_d2.size()
        x_d3 = self.down(x_d2)  # 32x32
        # ======= First Branch =======
        x = self.conv4(x)
        x = self.relu(x)
        x = self.bn4(x)
        x = self.conv5(x)
        x = self.relu(x)
        x = self.bn5(x)
        x = self.conv6(x)
        x = self.relu(x)
        x = self.bn6(x)
        x = self.conv7(x)
        x = self.relu(x)
        x = self.bn7(x)
        # ======= Second Branch ========
        x_d1 = self.conv4(x_d1)
        x_d1 = self.relu(x_d1)
        x_d1 = self.bn4(x_d1)
        x_d1 = self.conv5(x_d1)
        x_d1 = self.relu(x_d1)
        x_d1 = self.bn5(x_d1)
        x_d1 = self.conv6(x_d1)
        x_d1 = self.relu(x_d1)
        x_d1 = self.bn6(x_d1)
        x_d1 = self.conv7(x_d1)
        x_d1 = self.relu(x_d1)
        x_d1 = self.bn7(x_d1)
        x_d1 = self.upsample(x_d1, size=(H1, W1))
        # ======= Third Branch ========
        x_d2 = self.conv4(x_d2)
        x_d2 = self.relu(x_d2)
        x_d2 = self.bn4(x_d2)
        x_d2 = self.conv5(x_d2)
        x_d2 = self.relu(x_d2)
        x_d2 = self.bn5(x_d2)
        x_d2 = self.conv6(x_d2)
        x_d2 = self.relu(x_d2)
        x_d2 = self.bn6(x_d2)
        x_d2 = self.conv7(x_d2)
        x_d2 = self.relu(x_d2)
        x_d2 = self.bn7(x_d2)
        x_d2 = self.upsample(x_d2, size=(H2, W2))
        x_d2 = self.upsample(x_d2, size=(H1, W1))
        # ======= Fourth Branch ========
        x_d3 = self.conv4(x_d3)
        x_d3 = self.relu(x_d3)
        x_d3 = self.bn4(x_d3)
        x_d3 = self.conv5(x_d3)
        x_d3 = self.relu(x_d3)
        x_d3 = self.bn5(x_d3)
        x_d3 = self.conv6(x_d3)
        x_d3 = self.relu(x_d3)
        x_d3 = self.bn6(x_d3)
        x_d3 = self.conv7(x_d3)
        x_d3 = self.relu(x_d3)
        x_d3 = self.bn7(x_d3)
        x_d3 = self.upsample(x_d3, size=(H3, W3))
        x_d3 = self.upsample(x_d3, size=(H2, W2))
        x_d3 = self.upsample(x_d3, size=(H1, W1))
        # ======= Concat maps ==========
        x = torch.cat((x, x_d1, x_d2, x_d3), 1)
        x = self.conv_smooth1(x)
        x = self.conv_smooth2(x)
        x = self.conv_smooth3(x)
        # ============ Decoder ==========
        x = self.convtrans1(x)
        x = self.relu(x)
        x = self.convtrans2(x)
        x = self.relu(x)
        x = self.convtrans3(x)
        x = self.relu(x)
        x = self.convtrans4(x)
        x = self.relu(x)
        x = self.convtrans5(x)
        x = self.relu(x)
        x = self.convtrans6(x)
        x = self.relu(x)
        x = self.convtrans7(x)
        x = self.sigmoid(x)

        return x

    def upsample(self, x, size):
        up = nn.Upsample(size=size, mode="bilinear")
        return up(x)


class FPN_Gray(nn.Module):
    def __init__(self):
        super(FPN_Gray, self).__init__()
        self.name = "AEFPNC_Gray"
        self.bn0 = nn.BatchNorm2d(1)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1)  # 3x256x256 -> 16x256x256
        self.bn1 = nn.BatchNorm2d(num_features=16)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)  # 16x256x256  -> 32x256x256
        self.bn2 = nn.BatchNorm2d(num_features=32)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)  # 32x256x256 -> 64x256x256
        self.bn3 = nn.BatchNorm2d(num_features=64)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=48, kernel_size=3, padding=1)  # 64x256x256 -> 48x256x256
        self.bn4 = nn.BatchNorm2d(num_features=48)
        self.conv5 = nn.Conv2d(in_channels=48, out_channels=32, kernel_size=3, padding=1)  # 48x256x256 -> 32x256x256
        self.bn5 = nn.BatchNorm2d(num_features=32)
        self.conv6 = nn.Conv2d(in_channels=32, out_channels=24, kernel_size=3, padding=1)  # 32x256x256 -> 24x256x256
        self.bn6 = nn.BatchNorm2d(num_features=24)
        self.conv7 = nn.Conv2d(in_channels=24, out_channels=16, kernel_size=3, padding=1)  # 24x256x256 -> 16x256x256
        self.bn7 = nn.BatchNorm2d(num_features=16)
        self.down = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, stride=2)
        self.conv_smooth1 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, padding=1)
        self.conv_smooth2 = nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, padding=1)
        self.conv_smooth3 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.convtrans1 = nn.ConvTranspose2d(in_channels=16, out_channels=24, kernel_size=3, padding=1)
        self.convtrans2 = nn.ConvTranspose2d(in_channels=24, out_channels=32, kernel_size=3, padding=1)
        self.convtrans3 = nn.ConvTranspose2d(in_channels=32, out_channels=48, kernel_size=3, padding=1)
        self.convtrans4 = nn.ConvTranspose2d(in_channels=48, out_channels=64, kernel_size=3, padding=1)
        self.convtrans5 = nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=3, padding=1)
        self.convtrans6 = nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=3, padding=1)
        self.convtrans7 = nn.ConvTranspose2d(in_channels=16, out_channels=1, kernel_size=3, padding=1)

    def forward(self, x):
        # ============ Encoder ===========
        # ====== Bottom Up Layers =====
        x = self.bn0(x)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.bn1(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.bn2(x)
        x = self.conv3(x)
        x = self.relu(x)
        x = self.bn3(x)
        _, _, H1, W1 = x.size()
        # ======= Branch network ======
        x_d1 = self.down(x)  # 128x128
        _, _, H2, W2 = x_d1.size()
        x_d2 = self.down(x_d1)  # 64x64
        _, _, H3, W3 = x_d2.size()
        x_d3 = self.down(x_d2)  # 32x32
        # ======= First Branch =======
        x = self.conv4(x)
        x = self.relu(x)
        x = self.bn4(x)
        x = self.conv5(x)
        x = self.relu(x)
        x = self.bn5(x)
        x = self.conv6(x)
        x = self.relu(x)
        x = self.bn6(x)
        x = self.conv7(x)
        x = self.relu(x)
        x = self.bn7(x)
        # ======= Second Branch ========
        x_d1 = self.conv4(x_d1)
        x_d1 = self.relu(x_d1)
        x_d1 = self.bn4(x_d1)
        x_d1 = self.conv5(x_d1)
        x_d1 = self.relu(x_d1)
        x_d1 = self.bn5(x_d1)
        x_d1 = self.conv6(x_d1)
        x_d1 = self.relu(x_d1)
        x_d1 = self.bn6(x_d1)
        x_d1 = self.conv7(x_d1)
        x_d1 = self.relu(x_d1)
        x_d1 = self.bn7(x_d1)
        x_d1 = self.upsample(x_d1, size=(H1, W1))
        # ======= Third Branch ========
        x_d2 = self.conv4(x_d2)
        x_d2 = self.relu(x_d2)
        x_d2 = self.bn4(x_d2)
        x_d2 = self.conv5(x_d2)
        x_d2 = self.relu(x_d2)
        x_d2 = self.bn5(x_d2)
        x_d2 = self.conv6(x_d2)
        x_d2 = self.relu(x_d2)
        x_d2 = self.bn6(x_d2)
        x_d2 = self.conv7(x_d2)
        x_d2 = self.relu(x_d2)
        x_d2 = self.bn7(x_d2)
        x_d2 = self.upsample(x_d2, size=(H2, W2))
        x_d2 = self.upsample(x_d2, size=(H1, W1))
        # ======= Fourth Branch ========
        x_d3 = self.conv4(x_d3)
        x_d3 = self.relu(x_d3)
        x_d3 = self.bn4(x_d3)
        x_d3 = self.conv5(x_d3)
        x_d3 = self.relu(x_d3)
        x_d3 = self.bn5(x_d3)
        x_d3 = self.conv6(x_d3)
        x_d3 = self.relu(x_d3)
        x_d3 = self.bn6(x_d3)
        x_d3 = self.conv7(x_d3)
        x_d3 = self.relu(x_d3)
        x_d3 = self.bn7(x_d3)
        x_d3 = self.upsample(x_d3, size=(H3, W3))
        x_d3 = self.upsample(x_d3, size=(H2, W2))
        x_d3 = self.upsample(x_d3, size=(H1, W1))
        # ======= Concat maps ==========
        x = torch.cat((x, x_d1, x_d2, x_d3), 1)
        x = self.conv_smooth1(x)
        x = self.conv_smooth2(x)
        x = self.conv_smooth3(x)
        # ============ Decoder ==========
        x = self.convtrans1(x)
        x = self.relu(x)
        x = self.convtrans2(x)
        x = self.relu(x)
        x = self.convtrans3(x)
        x = self.relu(x)
        x = self.convtrans4(x)
        x = self.relu(x)
        x = self.convtrans5(x)
        x = self.relu(x)
        x = self.convtrans6(x)
        x = self.relu(x)
        x = self.convtrans7(x)
        x = self.sigmoid(x)

        return x

    def upsample(self, x, size):
        up = nn.Upsample(size=size, mode="bilinear")
        return up(x)
