import torch
from torch import nn
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class BotlenneckType1(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(3, 3), stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=(3, 3), stride=1, padding=1)
        self.relu2 = nn.ReLU()

    def forward(self, x: torch.Tensor):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        return x
    

class BotlenneckType2(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(3, 3), stride=1, padding=0)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=(3, 3), stride=1, padding=0)
        self.relu2 = nn.ReLU()

    def forward(self, x: torch.Tensor):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        return x


class DownSampleBlock(nn.Module):
    def __init__(self, in_channels=3, out_channels=16, device=None):
        super().__init__()
        self.c = BotlenneckType1(in_channels=in_channels, out_channels=out_channels)
        self.max_pool_1 = nn.MaxPool2d(kernel_size=(2, 2), stride=2, padding=0)

    def forward(self, x: torch.Tensor):
        x = self.c(x)
        x = self.max_pool_1(x)
        return x
    
class DownSample(nn.Module):
    def __init__(self, image_channels):
        super().__init__()
        self.down1 = DownSampleBlock(in_channels=image_channels, out_channels=16)
        self.down2 = DownSampleBlock(in_channels=16, out_channels=32)
        self.down3 = DownSampleBlock(in_channels=32, out_channels=64)
        self.down4 = DownSampleBlock(in_channels=64, out_channels=128)
        self.down5 = DownSampleBlock(in_channels=128, out_channels=256)
        self.down6 = DownSampleBlock(in_channels=256, out_channels=512)

    def forward(self, x: torch.Tensor):
        if x.shape[-1] <= 3:
            x = x.permute(0, 3, 1, 2)
        x_skips = [x]

        x = self.down1(x)
        x_skips.append(x)
        # print("down 1:",x.shape)

        x = self.down2(x)
        x_skips.append(x)
        # print("down 2:",x.shape)

        x = self.down3(x)
        x_skips.append(x)
        # print("down 3:",x.shape)

        x = self.down4(x)
        x_skips.append(x)
        # print("down 4:",x.shape)

        x = self.down5(x)
        x_skips.append(x)
        # print("down 5:",x.shape)

        x = self.down6(x)
        x_skips.append(x)
        # print("down 6:",x.shape)

        return x, x_skips
    

class UpSampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.t_conv = nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(2, 2), stride=2, padding=0)
        self.relu = nn.ReLU()
        self.botlenneck_block = BotlenneckType1(in_channels=out_channels*2, out_channels=out_channels)

    def forward(self, x: torch.Tensor, x_skip: torch.Tensor):
        x = self.relu(self.t_conv(x))
        x = torch.concat([x, x_skip], dim=1)
        x = self.botlenneck_block(x)
        return x
    

class UpSample(nn.Module):
    def __init__(self, image_channels=3):
        super().__init__()
        self.up1 = UpSampleBlock(1024, 512)
        self.up2 = UpSampleBlock(512, 256)
        self.up3 = UpSampleBlock(256, 128)
        self.up4 = UpSampleBlock(128, 64)
        self.up5 = UpSampleBlock(64, 32)
        self.up6 = UpSampleBlock(32, 16)
        self.up7 = UpSampleBlock(16, image_channels)

    def forward(self, x: torch.Tensor, x_skips: torch.Tensor):
        x = self.up1(x, x_skips[-1])
        x = self.up2(x, x_skips[-2])
        x = self.up3(x, x_skips[-3])
        x = self.up4(x, x_skips[-4])
        x = self.up5(x, x_skips[-5])
        x = self.up6(x, x_skips[-6])
        x = self.up7(x, x_skips[-7])
        return x
    

class Unet2015(nn.Module):
    def __init__(self, image_channels):
        super().__init__()
        self.down_sampler = DownSample(image_channels=image_channels)
        self.botlenneck = BotlenneckType2(in_channels=512, out_channels=1024)
        self.up_sampler = UpSample(image_channels=image_channels)

    def forward(self, x: torch.Tensor):
        if x.shape[2] != 512:
            raise ValueError("Model này được code để chạy trên chiều HxW = (512, 512)")
        x, x_skips = self.down_sampler(x)
        x = self.botlenneck(x)
        x = self.up_sampler(x, x_skips)
        return x
    

if __name__ == "__main__":
    model = Unet2015(3).to(device)
    x = torch.rand(1, 512, 512, 3).to(device)
    x = model(x)
    print(x)
