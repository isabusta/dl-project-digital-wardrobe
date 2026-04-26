import torch
from torch import nn
class FashionRecognitionBaselineModel(nn.Module):

  def __init__(self,
               input_shape: int,
               hidden_units: int,
               output_shape: int,
               image_size = 224) -> None:

               super().__init__()
               conv_blocks = []

               self.conv_block_1 = nn.Sequential(

                nn.Conv2d(in_channels=input_shape,
                   out_channels=hidden_units,
                   kernel_size=3,
                   stride = 1,
                   padding=1),
                nn.BatchNorm2d(hidden_units),
                nn.ReLU(),

                nn.Conv2d(in_channels=hidden_units,
                   out_channels=hidden_units,
                   kernel_size=3,
                   stride = 1,
                   padding=1),
                nn.BatchNorm2d(hidden_units),
                nn.ReLU(),

                nn.Conv2d(in_channels=hidden_units,
                   out_channels = hidden_units,
                   kernel_size=3,
                   stride = 1,
                   padding=1),

                nn.BatchNorm2d(hidden_units),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2,
                      stride=2),

     )
               self.conv_block_2 = nn.Sequential(

                nn.Conv2d(in_channels = hidden_units,
                   out_channels = 2 * hidden_units,
                   kernel_size=3,
                   stride = 1,
                   padding=1),

                nn.BatchNorm2d(2 * hidden_units),

                nn.ReLU(),

                nn.Conv2d(in_channels= 2 * hidden_units,
                   out_channels= 2 * hidden_units,
                   kernel_size=3,
                   stride = 1,
                   padding=1),
                nn.BatchNorm2d(2 * hidden_units),
                nn.ReLU(),

                nn.Conv2d(in_channels=2 * hidden_units,
                   out_channels= 2 * hidden_units,
                   kernel_size=3,
                   stride = 1,
                   padding=1),

                nn.BatchNorm2d(2 * hidden_units),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2,
                      stride=2),

     )
               self.conv_block_3 = nn.Sequential(

                nn.Conv2d(in_channels= 2 * hidden_units,
                   out_channels = 4 * hidden_units,
                   kernel_size=3,
                   stride = 1,
                   padding=1),
                nn.BatchNorm2d(4 * hidden_units),
                nn.ReLU(),

                nn.Conv2d(in_channels = 4 * hidden_units,
                   out_channels = 4 * hidden_units,
                   kernel_size=3,
                   stride = 1,
                   padding=1),
                nn.BatchNorm2d(4  *hidden_units),
                nn.ReLU(),

                nn.Conv2d(in_channels = 4 * hidden_units,
                   out_channels = 4 * hidden_units,
                   kernel_size=3,
                   stride = 1,
                   padding=1),
              nn.BatchNorm2d(4 * hidden_units),

              nn.ReLU(),

              nn.MaxPool2d(kernel_size=2,
                      stride=2),
     )
               self.conv_block_4 = nn.Sequential(

                nn.Conv2d(in_channels= 4 * hidden_units,
                   out_channels = 8 * hidden_units,
                   kernel_size=3,
                   stride = 1,
                   padding=1),
              nn.BatchNorm2d(8 * hidden_units),
              nn.ReLU(),

              nn.Conv2d(in_channels = 8 * hidden_units,
                   out_channels= 8 * hidden_units,
                   kernel_size=3,
                   stride = 1,
                   padding=1),
              nn.BatchNorm2d(8 * hidden_units),
              nn.ReLU(),

              nn.Conv2d(in_channels = 8 * hidden_units,
                   out_channels = 8 * hidden_units,
                   kernel_size=3,
                   stride = 1,
                   padding=1),

                nn.BatchNorm2d(8 * hidden_units),
                nn.ReLU(),

              nn.MaxPool2d(kernel_size=2,
                      stride=2),
     )

               conv_blocks.append(self.conv_block_1)
               conv_blocks.append(self.conv_block_2)
               conv_blocks.append(self.conv_block_3)
               conv_blocks.append(self.conv_block_4)

               output_size = image_size // (16)

               self.classifier = nn.Sequential(
                  nn.Flatten(),
                  nn.Dropout(0.35),
                  nn.Linear(in_features=8*hidden_units * output_size * output_size,
                  out_features=output_shape)
     )


  def forward(self, X):

    return self.classifier(
      self.conv_block_4(
          self.conv_block_3(
            self.conv_block_2(
                self.conv_block_1(X)
                )
            )
          )
    )

