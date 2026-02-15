import torch
import torch.nn as nn

n_class = 8 # assign no of classes in the dataset

patch_size = 16
projection_dim = 1024
mlp_hidden_dim = 4096
decoder_out_channels = 256
num_heads = 16
transformer_layers = 24

img_size = 512
num_patches = int((img_size*img_size)/(patch_size*patch_size))


class Gen_patches(nn.Module):
    def __init__(self, patch_size):
        super().__init__()
        self.proj = nn.Conv2d(3, projection_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, images):
        x = self.proj(images)
        x = x.flatten(2)  # (n_samples, embed_dim, n_patches)
        x = x.transpose(1, 2)  # (n_samples, n_patches, embed_dim)

        return x


class SelfAttention(nn.Module):
    def __init__(self, dim, heads=8, qkv_bias=False, qk_scale=None, dropout_rate=0.0):
        super().__init__()
        self.num_heads = heads
        head_dim = dim // heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(dropout_rate)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(dropout_rate)

    def forward(self, x):
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = (
            qkv[0],
            qkv[1],
            qkv[2],
        )

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        # print(attn.shape)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        # print(x.shape)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class MLP_block(nn.Module):
    def __init__(self, in_channel, out_channel, dropout_rate = 0.0):
        super().__init__()
        self.fc1 = nn.Linear(in_channel, out_channel)
        self.attn = nn.GELU()
        self.fc2 = nn.Linear(out_channel, in_channel)
        self.drop = nn.Dropout(p=dropout_rate)

    def forward(self, x):
        x = self.fc1(x)
        x = self.attn(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)

        return x

class Block(nn.Module):
    def __init__(self, projection_dim, num_heads, mlp_hidden_dim, qkv_bias=False, dropout_rate=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(projection_dim, eps=1e-6)
        self.attn = SelfAttention(
                projection_dim, 
                heads=num_heads, 
                qkv_bias=qkv_bias, 
                dropout_rate=dropout_rate
        )

        self.norm2 = nn.LayerNorm(projection_dim, eps=1e-6)

        self.mlp = MLP_block(
                in_channel=projection_dim,
                out_channel=mlp_hidden_dim,
                dropout_rate = dropout_rate
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))

        return x

class TSIF(nn.Module):
    def __init__(self, dim=1024, k=3, dropout_rate=0.0):
        super().__init__()
        self.position_conv = nn.Conv2d(dim, dim, kernel_size=k, stride=1, padding=1)
        self.drop = nn.Dropout(dropout_rate)

    def forward(self, x, H, W):
        print("TSIF x shape: ", x.shape)
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.position_conv(x) + x
        x = x.flatten(2).transpose(1, 2)

        x = self.drop(x)

        return x


class Model(nn.Module):
    def __init__(
            self,
            patch_size,
            n_class,
            projection_dim,
            depth,
            num_heads,
            decoder_out_channels,
            qkv_bias=True,
            dropout_rate=0.0,
    ):
        super().__init__()
        self.patch_embed = Gen_patches(patch_size)
        self.spatial_patch = TSIF(dropout_rate=dropout_rate)
        self.blocks = nn.ModuleList(
            [
                Block(
                    projection_dim,
                    num_heads,
                    mlp_hidden_dim,
                    qkv_bias=qkv_bias,
                    dropout_rate=dropout_rate
                )
                for _ in range(depth)
            ]
        )

        #Decoder layer initialization
        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(projection_dim, decoder_out_channels, kernel_size=3, stride=1, padding='same'),
            nn.GELU(),

            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(decoder_out_channels, decoder_out_channels, kernel_size=3, stride=1, padding='same'),
            nn.GELU(),

            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(decoder_out_channels, decoder_out_channels, kernel_size=3, stride=1, padding='same'),
            nn.GELU(),

            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(decoder_out_channels, decoder_out_channels, kernel_size=3, stride=1, padding='same'),
            nn.GELU(),

            #Output layer
            nn.Conv2d(decoder_out_channels, n_class, kernel_size=1, stride=1)
        )

    def forward(self, x):
        x = self.patch_embed(x)

        # print(x.shape)
        for i, block in enumerate(self.blocks):
            x = block(x)
            if i == 0:
                x = self.spatial_patch(x, 32, 32)

        # print(x.shape)
        x = x.transpose(1, 2)
        # print(x.shape)
        # img_2d = torch.reshape(x, (x.shape[0], x.shape[1], 64, 64))

        img_2d = torch.reshape(x, (x.shape[0], x.shape[1], 32, 32))
        # print(img_2d.shape)

        output = self.decoder(img_2d)

        return output

model = Model(patch_size, n_class, projection_dim, transformer_layers, num_heads, decoder_out_channels, dropout_rate=0.2)