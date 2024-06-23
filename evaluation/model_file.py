import torch.nn as nn
import torch


class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()

        self.list_instructions = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),

            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),

            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),

            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),

            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),

            nn.ConvTranspose2d(in_channels=32, out_channels=1, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.list_instructions(x)
        return x


class CNNEncDecModel(nn.Module):
    def __init__(self):
        super(CNNEncDecModel, self).__init__()

        self.encoder1 = self.apply_conv_layers(1, 64)
        self.encoder2 = self.apply_conv_layers(64, 128)
        self.encoder3 = self.apply_conv_layers(128, 256)
        self.bottleneck = self.apply_conv_layers(256, 512)
        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.decoder3 = self.apply_conv_layers(512, 256)
        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2, output_padding=(0, 1))
        self.decoder2 = self.apply_conv_layers(256, 128)
        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.decoder1 = self.apply_conv_layers(128, 64)
        self.final_conv = nn.Conv2d(64, 1, kernel_size=1)

    def apply_conv_layers(self, in_channels, out_channels):
        seq_instructions = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU()
        )
        return seq_instructions

    def forward(self, x):
        # (N, 1, 128, 170)
        e1 = self.encoder1(x)  # (N, 64, 128, 170)
        e2 = self.encoder2(nn.functional.max_pool2d(e1, 2))  # (N, 128, 64, 85)
        e3 = self.encoder3(nn.functional.max_pool2d(e2, 2))  # (N, 256, 32, 42)

        b = self.bottleneck(nn.functional.max_pool2d(e3, 2))  # (N, 512, 16, 21)

        d3 = self.upconv3(b)  # (N, 256, 32, 42)
        d3 = self.concatenate(d3, e3)
        d3 = self.decoder3(d3)  # (N, 256, 32, 42)

        d2 = self.upconv2(d3)  # (N, 128, 64, 85)
        d2 = self.concatenate(d2, e2)
        d2 = self.decoder2(d2)  # (N, 128, 64, 85)

        d1 = self.upconv1(d2)  # (N, 64, 128, 170)
        d1 = self.concatenate(d1, e1)
        d1 = self.decoder1(d1)  # (N, 64, 128, 170)

        output_layer = self.final_conv(d1)
        return output_layer

    def concatenate(self, upsampled, bypass):
        _, _, H, W = upsampled.size()
        bypass = self.prepare_position_crop(bypass, H, W)
        return torch.cat((upsampled, bypass), dim=1)

    def prepare_position_crop(self, layer, max_height, max_width):
        _, _, h, w = layer.size()
        diff_y = (h - max_height) // 2
        diff_x = (w - max_width) // 2
        return layer[:, :, diff_y:diff_y + max_height, diff_x:diff_x + max_width]
