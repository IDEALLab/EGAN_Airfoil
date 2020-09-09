import torch
import numpy as np
import torch.nn as nn
import layers

class FeatureGenerator(nn.Module):
    """Regular fully connected network generating features.
    
    Args:
        in_features: The number of input features.
        out_feature: The number of output features.
        feature_gen_layers: The widths of the hidden layers.
        combo: The layer combination to be stacked up.
    
    Shape:
        - Input: `(N, H_in)` where H_in = in_features.
        - Output: `(N, H_out)` where H_out = out_features.
    """
    def __init__(
        self, in_features, out_features, 
        feature_gen_layers=[1024,],
        combo = layers.LinearCombo
        ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.generator = self.build_generator(feature_gen_layers, combo)
    
    def forward(self, input):
        return self.generator(input)

    def build_generator(self, feature_gen_layers, combo):
        generator = nn.Sequential()
        for idx, (in_ftr, out_ftr) in enumerate(zip(
            [self.in_features] + feature_gen_layers, 
            feature_gen_layers + [self.out_features]
            )):
            generator.add_module(str(idx), combo(in_ftr, out_ftr))
        return generator

class CPWGenerator(nn.Module):
    """Generate given number of control points and weights for Bezier Layer.

    Args:
        in_features: The number of input features.
        n_control_points: The number of control point and weights to be output.
        dense_layers: The widths of the hidden layers of the MLP connecting 
            input features and deconvolutional layers.
        deconv_channels: The number of channels deconvolutional layers have.
    
    Shape:
        - Input: `(N, H_in)` where H_in = in_features.
        - Output:
            - Control Points: `(N, 2, H_out)` where H_out = n_control_points.
            - Weights: `(N, 1, H_out)` where H_out = n_control_points.
    """
    def __init__(
        self, in_features, n_control_points,
        dense_layers = [1024,],
        deconv_channels = [32*8, 32*4, 32*2, 32],
        ):
        super().__init__()
        self.in_features = in_features
        self.n_control_points = n_control_points

        self.in_chnl, self.in_width = self.calculate_parameters(n_control_points, deconv_channels)

        self.dense = FeatureGenerator(in_features, self.in_chnl * 3 * self.in_width, dense_layers)
        self.deconv = self.build_deconv(deconv_channels)
        self.cp_gen = nn.Sequential(nn.Conv2d(deconv_channels[-1], 1, (2, 1)), nn.Tanh())
        self.w_gen = nn.Sequential(nn.Conv2d(deconv_channels[-1], 1, (3, 1)), nn.Sigmoid())
    
    def forward(self, input):
        x = self.deconv(self.dense(input).view(-1, self.in_chnl, 3, self.in_width))
        cp = self.cp_gen(x).squeeze(1)
        w = self.w_gen(x).squeeze(1)
        return cp, w
    
    def calculate_parameters(self, n_control_points, channels):
        n_l = len(channels) - 1
        in_chnl = channels[0]
        in_width = n_control_points // (2 ** n_l)
        assert in_width >= 4, 'Too many deconvolutional layers ({}) for the {} control points.'\
            .format(n_l, self.n_control_points)
        return in_chnl, in_width
    
    def build_deconv(self, channels):
        deconv = nn.Sequential()
        for idx, (in_chnl, out_chnl) in enumerate(zip(channels[:-1], channels[1:])):
            deconv.add_module(
                str(idx), layers.DeconvCombo(
                    in_chnl, out_chnl, 
                    kernel_size=(3,4), stride=(1,2), padding=1
                    )
                )
        return deconv

class BezierGenerator(nn.Module):
    """Generator for BezierGAN related projects.

    Args:
        in_features: The number of input features.
        n_control_points: The number of control point and weights to be output.
        dense_layers: The widths of the hidden layers of the MLP connecting 
            input features and deconvolutional layers.
        deconv_channels: The number of channels deconvolutional layers have.
    
    Shape:
        - Input: `(N, H_in)` where H_in = in_features.
        - Output:
            - Data Points: `(N, D, DP)` where D is the dimension and DP is the number of data points.
            - Control Points: `(N, 2, CP)` where CP = n_control_points.
            - Weights: `(N, 1, CP)` where CP = n_control_points.
            - Parameter Variables: `(N, 1, DP)` where DP is the number of data points.
            - Intervals: `(N, DP)` where DP is the number of data points.
    """
    def __init__(
        self, in_features, n_control_points, n_data_points, 
        m_features=256,
        feature_gen_layers=[1024,],
        dense_layers = [1024,],
        deconv_channels = [32*8, 32*4, 32*2, 32],
        ):
        super().__init__()
        self.in_features = in_features
        self.n_control_points = n_control_points
        self.n_data_points = n_data_points

        self.feature_generator = FeatureGenerator(in_features, m_features, feature_gen_layers)
        self.cpw_generator = CPWGenerator(in_features, n_control_points, dense_layers, deconv_channels)
        self.bezier_layer = layers.BezierLayer(m_features, n_control_points, n_data_points)
    
    def forward(self, input):
        features = self.feature_generator(input)
        cp, w = self.cpw_generator(input)
        dp, pv, intvls = self.bezier_layer(features, cp, w)
        return dp, cp, w, pv, intvls

class Critic(nn.Module):
    def __init__(self):
        super().__init__()
        self.discriminator

if __name__ == "__main__":
    b_gen = BezierGenerator(10, 40, 192)
    b = torch.rand([50, 10])
    dp, cp, w, _, _ = b_gen(b)
    print(dp, cp, w)