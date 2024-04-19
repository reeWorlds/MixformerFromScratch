import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class ResNetBlock(nn.Module):
    def __init__(self, config):
        super(ResNetBlock, self).__init__()

        self.channels = config["channels"]
        self.mid_channels = config["mid_channels"]

        self.conv1 = nn.Conv2d(self.channels, self.mid_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.mid_channels)
        self.conv2 = nn.Conv2d(self.mid_channels, self.channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(self.channels)

    def forward(self, x):
        y = F.relu(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        return F.relu(x + y)


class Encoder(nn.Module):
    def __init__(self, config):
        super(Encoder, self).__init__()

        self.part1_channels = config["part1_channels"]
        self.part2_channels = config["part2_channels"]
        self.part3_channels = config["part3_channels"]
        self.target_h = config["target_h"]
        self.target_w = config["target_w"]
        self.search_h = config["search_h"]
        self.search_w = config["search_w"]
        self.target_out_h = config["target_out_h"]
        self.target_out_w = config["target_out_w"]
        self.search_out_h = config["search_out_h"]
        self.search_out_w = config["search_out_w"]
        self.embeddings = config["embeddings"]
        self.reduce_size = config["reduce_size"]

        self.cnn1 = nn.Conv2d(3, self.part1_channels, kernel_size=5, stride=1, padding=2)
        self.block1 = nn.ModuleList([ResNetBlock(config["block1"]) for _ in range(2)])
        self.cnn2 = nn.Conv2d(self.part1_channels, self.part2_channels, kernel_size=3, stride=2, padding=1)
        self.block2 = nn.ModuleList([ResNetBlock(config["block2"]) for _ in range(2)])
        self.drop1 = nn.Dropout(0.5)
        self.cnn3 = nn.Conv2d(self.part2_channels, self.part3_channels, kernel_size=3, stride=2, padding=1)
        self.block3 = nn.ModuleList([ResNetBlock(config["block3"]) for _ in range(3)])
        self.drop2 = nn.Dropout(0.5)
        
        self.reduce1 = nn.Conv2d(self.part3_channels, self.reduce_size, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(self.reduce_size)
        self.drop3 = nn.Dropout(0.5)
        self.reduce2 = nn.Conv2d(16 * self.reduce_size, self.reduce_size, kernel_size=1, stride=1, padding=0)
        self.bn2 = nn.BatchNorm2d(self.reduce_size)
        self.linear = nn.Linear(self.reduce_size, self.embeddings // 16)

    def forward(self, x):
        B = x.shape[0]

        x = rearrange(x, 'b h w c -> b c h w')

        x = F.relu(self.cnn1(x))
        for block in self.block1:
            x = block(x)
        x = F.relu(self.cnn2(x))
        for block in self.block2:
            x = block(x)
        x = self.drop1(x)
        x = F.relu(self.cnn3(x))
        for block in self.block3:
            x = block(x)
        x = self.drop2(x)

        x = self.bn1(F.relu(self.reduce1(x)))
        x = rearrange(x, 'b c (h p1) (w p2) -> b (c p1 p2) h w', p1=4, p2=4).contiguous()
        x = self.drop3(x)
        x = self.bn2(F.relu(self.reduce2(x)))
        x = rearrange(x, "b c h w -> b h w c").contiguous()
        x = self.linear(x)
        x = x.view(B, self.embeddings)

        return x


class Decoder(nn.Module):
    def __init__(self, config):
        super(Decoder, self).__init__()

        self.target_h = config["target_h"]
        self.target_w = config["target_w"]
        self.search_h = config["search_h"]
        self.search_w = config["search_w"]
        self.target_out_h = config["target_out_h"]
        self.target_out_w = config["target_out_w"]
        self.search_out_h = config["search_out_h"]
        self.search_out_w = config["search_out_w"]
        self.embeddings = config["embeddings"]
        self.part3_channels = config["part3_channels"]
        self.part2_channels = config["part2_channels"]
        self.part1_channels = config["part1_channels"]
        self.reduce_size = config["reduce_size"]

        self.linear = nn.Linear(self.embeddings // 16, self.reduce_size)
        self.upscale2 = nn.Conv2d(self.reduce_size, 16 * self.reduce_size, kernel_size=1, stride=1, padding=0)
        self.bn2 = nn.BatchNorm2d(16 * self.reduce_size)
        self.drop3 = nn.Dropout(0.5)
        self.upscale1 = nn.Conv2d(self.reduce_size, self.part3_channels, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(self.part3_channels)
        self.drop2 = nn.Dropout(0.5)

        self.block3 = nn.ModuleList([ResNetBlock(config["block3"]) for _ in range(3)])
        self.cnn3 = nn.Conv2d(self.part3_channels, self.part2_channels * 4, kernel_size=1, stride=1, padding=0)
        self.upsampler3 = nn.PixelShuffle(2)
        self.drop1 = nn.Dropout(0.5)
        self.block2 = nn.ModuleList([ResNetBlock(config["block2"]) for _ in range(2)])
        self.cnn2 = nn.Conv2d(self.part2_channels, self.part1_channels * 4, kernel_size=1, stride=1, padding=0)
        self.upsampler2 = nn.PixelShuffle(2)
        self.block1 = nn.ModuleList([ResNetBlock(config["block1"]) for _ in range(2)])
        self.cnn1 = nn.Conv2d(self.part1_channels, 3, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        B = x.shape[0]
        
        x = x.view(B, 4, 4, self.embeddings // 16)
        x = self.linear(x)
        x = rearrange(x, "b h w c -> b c h w")
        x = self.bn2(F.relu(self.upscale2(x)))
        x = rearrange(x, 'b (c p1 p2) h w -> b c (h p1) (w p2)', p1=4, p2=4)
        x = self.drop3(x)
        x = self.bn1(F.relu(self.upscale1(x)))
        x = self.drop2(x)

        for block in self.block3:
            x = block(x)
        x = F.relu(self.upsampler3(self.cnn3(x)))
        x = self.drop1(x)
        for block in self.block2:
            x = block(x)
        x = F.relu(self.upsampler2(self.cnn2(x)))
        for block in self.block1:
            x = block(x)
        x = self.cnn1(x)
        x = rearrange(x, 'b c h w -> b h w c')

        return x


class Autoencoder(nn.Module):
    def __init__(self, config):
        super(Autoencoder, self).__init__()

        self.embeddings = config["embeddings"]

        self.encoder = Encoder(config['encoder'])
        self.decoder = Decoder(config['decoder'])

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    
    def forward_encoder(self, x):
        return self.encoder(x)
    
    def forward_decoder(self, x):
        return self.decoder(x)
