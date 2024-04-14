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
        self.pos_enc_channels = config["pos_enc_channels"]
        self.type_channels = config["type_channels"]
        self.embeddings = config["embeddings"]

        self.cnn1 = nn.Conv2d(3, self.part1_channels, kernel_size=5, stride=1, padding=2)
        self.block1 = nn.ModuleList([ResNetBlock(config["block1"]) for _ in range(3)])
        self.cnn2 = nn.Conv2d(self.part1_channels, self.part2_channels, kernel_size=3, stride=1, padding=1)
        self.block2 = nn.ModuleList([ResNetBlock(config["block2"]) for _ in range(3)])
        self.cnn3 = nn.Conv2d(self.part2_channels, self.part3_channels, kernel_size=3, stride=1, padding=1)
        self.block3 = nn.ModuleList([ResNetBlock(config["block3"]) for _ in range(3)])
        
        target_weights = torch.randn(self.pos_enc_channels, self.target_h, self.target_w)
        self.register_parameter('target_pos_embedding', nn.Parameter(target_weights))
        search_weights = torch.randn(self.pos_enc_channels, self.search_h, self.search_w)
        self.register_parameter('search_pos_embedding', nn.Parameter(search_weights))
        self.type_embedding = nn.Embedding(2, self.type_channels)

        self.drop1 = nn.Dropout(0.3)
        self.linear1 = nn.Linear(self.part3_channels + self.pos_enc_channels + self.type_channels,
                                 self.embeddings)
        self.linear2 = nn.Linear(self.embeddings, self.embeddings)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x, x_type):
        B = x.shape[0]

        x = rearrange(x, 'b h w c -> b c h w')

        x = F.relu(self.cnn1(x))
        for block in self.block1:
            x = block(x)
        x = F.relu(self.cnn2(x))
        for block in self.block2:
            x = block(x)
        x = F.relu(self.cnn3(x))
        for block in self.block3:
            x = block(x)

        pos_embedding = self.target_pos_embedding if x_type == 0 else self.search_pos_embedding
        pos_embedding = pos_embedding.repeat(B, 1, 1, 1)
        type_embedding = self.type_embedding(x_type).repeat(B, 1)
        type_embedding = type_embedding.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, x.shape[2], x.shape[3])

        x = torch.cat([x, pos_embedding, type_embedding], dim=1)
        x = rearrange(x, 'b c h w -> b h w c')

        x = self.drop1(x)
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        x = rearrange(x, 'b h w c -> b c h w')
        x = self.global_pool(x).squeeze(-1).squeeze(-1)

        return x


class Decoder(nn.Module):
    def __init__(self, config):
        super(Decoder, self).__init__()

        self.target_h = config["target_h"]
        self.target_w = config["target_w"]
        self.search_h = config["search_h"]
        self.search_w = config["search_w"]
        self.pos_enc_channels = config["pos_enc_channels"]
        self.type_channels = config["type_channels"]
        self.embeddings = config["embeddings"]
        self.part3_channels = config["part3_channels"]
        self.part2_channels = config["part2_channels"]
        self.part1_channels = config["part1_channels"]

        target_weights = torch.randn(self.pos_enc_channels, self.target_h, self.target_w)
        self.register_parameter('target_pos_embedding', nn.Parameter(target_weights))
        search_weights = torch.randn(self.pos_enc_channels, self.search_h, self.search_w)
        self.register_parameter('search_pos_embedding', nn.Parameter(search_weights))
        self.type_embedding = nn.Embedding(2, self.type_channels)

        self.linear1 = nn.Linear(self.embeddings + self.pos_enc_channels + self.type_channels,
                                 self.embeddings)
        self.drop1 = nn.Dropout(0.3)
        self.linear2 = nn.Linear(self.embeddings, self.part3_channels)

        self.block3 = nn.ModuleList([ResNetBlock(config["block3"]) for _ in range(3)])
        self.cnn3 = nn.Conv2d(self.part3_channels, self.part2_channels, kernel_size=3, stride=1, padding=1)
        self.block2 = nn.ModuleList([ResNetBlock(config["block2"]) for _ in range(3)])
        self.cnn2 = nn.Conv2d(self.part2_channels, self.part1_channels, kernel_size=3, stride=1, padding=1)
        self.block1 = nn.ModuleList([ResNetBlock(config["block1"]) for _ in range(3)])
        self.cnn1 = nn.Conv2d(self.part1_channels, 3, kernel_size=5, stride=1, padding=2)

    def forward(self, x, x_type):
        B = x.shape[0]
        image_h, image_w = (48, 48) if x_type == 0 else (64, 64)

        x = x.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, image_h, image_w)
        pos_embedding = self.target_pos_embedding if x_type == 0 else self.search_pos_embedding
        pos_embedding = pos_embedding.repeat(B, 1, 1, 1)
        type_embedding = self.type_embedding(x_type).repeat(B, 1)
        type_embedding = type_embedding.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, image_h, image_w)
        
        x = torch.cat([x, pos_embedding, type_embedding], dim=1)
        x = rearrange(x, 'b c h w -> b h w c')
        x = F.relu(self.linear1(x))
        x = self.drop1(x)
        x = F.relu(self.linear2(x))

        x = rearrange(x, 'b h w c -> b c h w')
        for block in self.block3:
            x = block(x)
        x = F.relu(self.cnn3(x))
        for block in self.block2:
            x = block(x)
        x = F.relu(self.cnn2(x))
        for block in self.block1:
            x = block(x)
        x = F.sigmoid(self.cnn1(x))
        x = rearrange(x, 'b c h w -> b h w c')

        return x


class Autoencoder(nn.Module):
    def __init__(self, config):
        super(Autoencoder, self).__init__()

        self.embeddings = config["embeddings"]

        self.encoder = Encoder(config['encoder'])
        self.decoder = Decoder(config['decoder'])

    def forward(self, x, x_type):
        x_type = x_type.to(x.device)
        x = self.encoder(x, x_type)
        x = self.decoder(x, x_type)
        return x
    
    def forward_encoder(self, x, x_type):
        x_type = x_type.to(x.device)
        return self.encoder(x, x_type)
    
    def forward_decoder(self, x, x_type):
        x_type = x_type.to(x.device)
        return self.decoder(x, x_type)
