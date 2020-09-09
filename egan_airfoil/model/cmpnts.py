import torch
import numpy as np
import torch.nn as nn
import layers

class FeatureGenerator(nn.Module):
    """Generate features for Bezier Layer.
    
    Args:
        in_features: The number of input features.
        out_feature: The number of output features.
        feature_gen_layers: The widths of the intermediate layers of the feature generator.
    """
    def __init__(self, in_features, out_features, feature_gen_layers=[1024,]):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.generator = self.build_generator(feature_gen_layers)
    
    def forward(self, input):
        return self.generator(input)

    def build_generator(self, feature_gen_layers):
        generator = nn.Sequential()
        for idx, (in_ftr, out_ftr) in enumerate(zip(
            [self.in_features]+feature_gen_layers, 
            feature_gen_layers+[self.out_features]
            )):
            generator.add_module(str(idx), layers.LinearCombo(in_ftr, out_ftr))
        return generator

class CPWGenerator(nn.Module):
    """Generate control points and weights for Bezier Layer.
    """
    def __init__(self, in_features, n_control_points, n_data_points, configs):
        super().__init__()
        self.in_features = in_features
        self.n_control_points = n_control_points
        self.n_data_points = n_data_points

        self.chnl_cpw = 32 * 8
        self.n_cpw = n_control_points // 8
        self.kernel_size = (3, 4)

        self.dense = self.build_dense()
        self.deconv = self.build_deconv()
        self.cp_gen = self.build_cp_gen()
        self.w_gen = self.build_w_gen()
    
    def forward(self, input):
        x = self.deconv(
            self.dense(input).view(-1, self.chnl_cpw, 3, self.n_cpw)
        )
        cp = self.cp_gen(x).unsqueeze(1)
        w = self.w_gen(x).unsqueeze(1)
        return cp, w
    
    def build_dense(self):
        dense = nn.Sequential(
            layers.LinearCombo(self.in_features, 1024),
            layers.LinearCombo(1024, self.n_cpw * 3 * self.chnl_cpw)
        )
        return dense
    
    def build_deconv(self):
        deconv = nn.Sequential(
            layers.DeconvCombo(self.chnl_cpw, self.chnl_cpw//2, self.kernel_size, stride=(1,2), padding=1),
            layers.DeconvCombo(self.chnl_cpw//2, self.chnl_cpw//4, self.kernel_size, stride=(1,2), padding=1),
            layers.DeconvCombo(self.chnl_cpw//4, self.chnl_cpw//8, self.kernel_size, stride=(1,2), padding=1)
        )
        return deconv

    def build_cp_gen(self):
        cp_gen = nn.Sequential(
            nn.Conv2d(self.chnl_cpw//8, 1, (2, 1)),
            nn.Tanh()
        )
        return cp_gen

    def build_w_gen(self):
        w_gen = nn.Sequential(
            nn.Conv2d(self.chnl_cpw//8, 1, (3, 1)),
            nn.Sigmoid()
        )
        return w_gen

class BezierGenerator(nn.Module):
    def __init__(
        self, in_features, n_control_points, n_data_points, 
        m_features=256,
        feature_gen_layers=[1024,],
        cpw_gen_layers=3
        ):
        super().__init__()
        self.in_features = in_features
        self.m_features = m_features
        self.n_control_points = n_control_points
        self.n_data_points = n_data_points

        self.cpw_generator = CPWGenerator(
            in_features, n_control_points, n_data_points
            )
        self.feature_generator = FeatureGenerator(
            in_features, m_features, feature_gen_layers
            )
        self.bezier_layer = layers.BezierLayer(
            self.m_features, 
            self.n_control_points, 
            self.n_data_points
            )
    
    def forward(self, input):
        cp, w = self.cpw_generator(input)
        features = self.feature_generator(input)
        dp, pv, intvls = self.bezier_layer(features, cp, w)
        return dp, cp, w, pv, intvls

class BezierDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.discriminator

if __name__ == "__main__":
    a = FeatureGenerator(10, 20)
