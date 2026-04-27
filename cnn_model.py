import torch
from torch import nn


class ResidualBlock(nn.Module):

    def __init__(self, input_shape, output_shape):
        super().__init__()

        self.conv_block = self.make_block(input_shape, output_shape)

        self.skip = nn.Identity()
        self.relu = nn.ReLU(False)

        if input_shape != output_shape:
            self.skip = nn.Sequential(
                nn.Conv2d(input_shape, output_shape, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(output_shape),
            )

    def make_block(self, in_f, out_f):
          layers = [
                nn.Conv2d(in_f, out_f, kernel_size=3, padding=1, stride=1),
                nn.BatchNorm2d(out_f),
                nn.ReLU(),
                nn.Conv2d(out_f, out_f, kernel_size=3, padding=1, stride=1),
                nn.BatchNorm2d(out_f),
                nn.ReLU(),
                nn.Conv2d(out_f, out_f, kernel_size=3, padding=1, stride=1),
                nn.BatchNorm2d(out_f)
            ]

          return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv_block(x)
        out = out + self.skip(x)
        out = self.relu(out)
        return out

class FashionRecognitionModel(nn.Module):
    def __init__(self, input_shape: int, hidden_units: int, output_shape: int, image_size=224):
        super().__init__()

        # Definition layers
        self.block_1 = nn.Sequential(
            ResidualBlock(input_shape, hidden_units)
        )
        # 224 -> 112
        self.block_2 = nn.Sequential(
            ResidualBlock(hidden_units, hidden_units),
            nn.MaxPool2d(kernel_size=2, stride=2)
            )

        self.block_3 = nn.Sequential(
            ResidualBlock(hidden_units, 2 * hidden_units)
            )
        # 112 -> 56
        self.block_4 = nn.Sequential(
            ResidualBlock(2 * hidden_units, 2 * hidden_units),
            nn.MaxPool2d(kernel_size=2, stride=2)
            )

        self.block_5 = nn.Sequential(
            ResidualBlock(2 * hidden_units, 4 * hidden_units)
            )
        # 56 -> 28
        self.block_6 = nn.Sequential(
            ResidualBlock(4 * hidden_units, 4 * hidden_units),
             nn.MaxPool2d(kernel_size=2, stride=2)
            )

        self.block_7 = nn.Sequential(
            ResidualBlock(4 * hidden_units, 8 * hidden_units)
            )

        # 28 -> 14
        self.block_8 = nn.Sequential(
            ResidualBlock(8 * hidden_units, 16 * hidden_units),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

         # 28 -> 7
        self.block_9 = nn.Sequential(
            ResidualBlock(16 * hidden_units, 320),
            nn.MaxPool2d(kernel_size=2, stride=2)
            )


        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=1),
            nn.Flatten(),
            nn.Dropout(0.4),
            nn.Linear(320, output_shape)
        )

    def forward(self, x):
      x = self.block_1(x)
      x = self.block_2(x)
      x = self.block_3(x)
      x = self.block_4(x)
      x = self.block_5(x)
      x = self.block_6(x)
      x = self.block_7(x)
      x = self.block_8(x)
      x = self.block_9(x)

      x = self.classifier(x)

      return x
