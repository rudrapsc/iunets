import torch
from torch import nn
from .utils import get_num_channels
from .layers import StandardBlock

class StandardUNet(nn.Module):
    def __init__(self,
                input_shape_or_channels,
                dim=None,
                architecture=[2,2,2,2],
                base_filters=32,
                skip_connection=False,
                block_type=StandardBlock,
                zero_init=False,
                *args,
                **kwargs):
        super(StandardUNet, self).__init__()
        self.input_channels = get_num_channels(input_shape_or_channels)
        self.base_filters = base_filters
        self.architecture = architecture
        self.n_levels = len(self.architecture)
        self.dim = dim
        self.skip_connection = skip_connection
        self.block_type = block_type
        
        pool_ops = [nn.MaxPool1d, 
                    nn.MaxPool2d, 
                    nn.MaxPool3d]
        pool_op = pool_ops[dim-1]

        upsampling_ops = [nn.ConvTranspose1d,
                          nn.ConvTranspose2d,
                          nn.ConvTranspose3d]
        upsampling_op = upsampling_ops[dim-1]


        filters = self.base_filters
        filters_list = [filters]
        
        self.module_L = nn.ModuleList()
        self.module_R = nn.ModuleList()
        self.downsampling_layers = nn.ModuleList()
        self.upsampling_layers = nn.ModuleList()
        
        
        # Left side of the U-Net
        for i in range(self.n_levels):
            self.module_L.append(nn.ModuleList())
            self.downsampling_layers.append(
                pool_op(kernel_size=2)
            )
            
            depth = architecture[i]    
                
            for j in range(depth):
                if i == 0 and j == 0:
                    in_channels = self.input_channels
                else:
                    in_channels = self.base_filters * (2**i)
                    
                if j == depth-1:
                    out_channels = self.base_filters * (2**(i+1))
                else:
                    out_channels = self.base_filters * (2**i)
                self.module_L[i].append(
                    self.block_type(self.dim, in_channels, out_channels, zero_init, *args, **kwargs)
                )
                
        
        # Right side of the U-Net
        for i in range(self.n_levels-1):
            self.module_R.append(nn.ModuleList())
            depth = architecture[i]    
            for j in range(depth):
                if j == 0:
                    in_channels = 3*self.base_filters * (2**(i+1))
                else:
                    in_channels = self.base_filters * (2**(i+1))
                out_channels = self.base_filters * (2**(i+1))
                self.module_R[i].append(
                    self.block_type(self.dim, in_channels, out_channels, zero_init, *args, **kwargs)
                )
            

            self.upsampling_layers.append(
                upsampling_op(self.base_filters * (2**(i+2)),
                              self.base_filters * (2**(i+2)),
                              kernel_size=2,
                              stride=2)
            )
        
        if self.skip_connection:
            # We have to convert back to the original number of channels if
            # we want a skip connection. We do this with an appropriate
            # convolution.
            conv_ops = [nn.Conv1d,
                        nn.Conv2d,
                        nn.Conv3d]
            conv_op = conv_ops[self.dim-1]
            self.output_layer = conv_op(self.base_filters*2,
                                   self.input_channels,
                                   3,
                                   padding=1)


    # def forward(self, input, *args, **kwargs):

    #     # FORWARD
    #     skip_inputs = []
        
    #     x = input
        
    #     # Left side
    #     for i in range(self.n_levels):
    #         depth = self.architecture[i]
    #         #  Left side
    #         for j in range(depth):
    #             x = self.module_L[i][j](x)

    #         # Downsampling L
    #         if i < self.n_levels - 1:
    #             skip_inputs.append(x)
    #             x = self.downsampling_layers[i](x)

    #     # Right side
    #     for i in range(self.n_levels - 2, -1, -1):
    #         depth = self.architecture[i]


    #         #implemention Soft attention by rudra 
    #         phi_x = self.upsampling_layers[i](x)
    #         phi_g = skip_inputs.pop()
    #         conv_layer = nn.Conv2d(in_channels=z, out_channels=z, kernel_size=3, stride=2, padding=1)
    #         phi_g = conv_layer(phi_g)
    #         conv_layer = nn.Conv2d(in_channels=z, out_channels=2*z, kernel_size=1, stride=1, padding=0)
    #         phi_x = conv_layer(phi_x)
            


    #         # Upsampling R
    #         x = self.upsampling_layers[i](x)
    #         y = skip_inputs.pop()
    #         x = torch.cat((x,y),dim=1)
            
    #         for j in range(depth):
    #             x = self.module_R[i][j](x)

    #     if self.skip_connection:
    #         x = self.output_layer(x) + input

    #     return x






    def forward(self, input, *args, **kwargs):
        # FORWARD
        skip_inputs = []
        
        x = input
        
        # Left side
        for i in range(self.n_levels):
            depth = self.architecture[i]
            # Left side
            for j in range(depth):
                x = self.module_L[i][j](x)

            # Downsampling L
            if i < self.n_levels - 1:
                skip_inputs.append(x)
                x = self.downsampling_layers[i](x)

        # Right side
        for i in range(self.n_levels - 2, -1, -1):
            depth = self.architecture[i]

            # Soft attention implementation by Rudra
            # Retrieve the feature maps from the left side (skip connection)
            phi_g = skip_inputs.pop()
            
            # Upsample the feature map from the current level on the right side
            phi_x = self.upsampling_layers[i](x)

            # Apply the first convolution to `phi_g` (from skip connection)
            conv_layer_g = nn.Conv2d(in_channels=phi_g.size(1), out_channels=phi_g.size(1), kernel_size=3, stride=2, padding=1)
            phi_g = conv_layer_g(phi_g)
            
            # Apply the second convolution to `phi_x` (upsampled right side)
            conv_layer_x = nn.Conv2d(in_channels=phi_x.size(1), out_channels=phi_x.size(1) * 2, kernel_size=1, stride=1, padding=0)
            phi_x = conv_layer_x(phi_x)

            # Element-wise addition of `phi_g` and `phi_x`
            concat_xg = phi_g + phi_x

            # Apply ReLU activation
            act_xg = F.relu(concat_xg)

            # Apply 2D convolution with 1 filter, 1x1 kernel, and 'same' padding
            psi = F.conv2d(act_xg, weight=torch.randn(1, act_xg.size(1), 1, 1), padding=0)

            # Apply sigmoid activation
            sigmoid_xg = torch.sigmoid(psi)

            # Get the shape for upsampling
            shape_sigmoid = sigmoid_xg.size()
            upsample_size = (phi_g.size(2), phi_g.size(3))  # Match `phi_g` dimensions

            # Upsample the psi tensor to match the spatial dimensions of `phi_g`
            upsample_psi = F.interpolate(sigmoid_xg, size=upsample_size, mode='nearest')

            # Multiply the upsampled psi with `phi_g`
            y = upsample_psi * phi_g

            # Final convolution on the attention output
            final_conv_layer = nn.Conv2d(in_channels=y.size(1), out_channels=phi_g.size(1), kernel_size=1, stride=1, padding=0)
            result = final_conv_layer(y)

            # Apply batch normalization
            result_bn = nn.BatchNorm2d(result.size(1))(result)

            # Combine the result of the attention mechanism with the upsampled `phi_x`
            x = torch.cat((result_bn, phi_x), dim=1)

            # Continue with the right side of the U-Net
            for j in range(depth):
                x = self.module_R[i][j](x)

        if self.skip_connection:
            x = self.output_layer(x) + input

        return x
