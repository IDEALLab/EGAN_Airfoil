import torch
import torch.nn as nn
from .layers import BezierLayer

class FeatureGenerator(nn.Module):
    """Generator features for Bezier Layer.
    """
    def __init__(self, in_features, configs):
        super().__init__()
        self.in_features = in_features
        self.generator = self.build_generator(configs)
    
    def forward(self, input):
        return self.generator(input)

    def build_generator(self, configs):
        pre_width = self.in_features
        modules = []
        for layer in configs:
            modules.append(nn.Linear(pre_width, layer["width"]))
            modules.append(nn.BatchNorm1d(layer["width"]))
            modules.append(nn.LeakyReLU(layer["alpha"]))
            pre_width = layer["width"]
        return nn.Sequential(*modules)

class CPWGenerator(nn.Module):
    """Generate control points and weights for Bezier Layer.
    """
    def __init__(self, in_features, n_control_points, configs):
        super().__init__()
        self.in_features = in_features
        self.n_control_points = n_control_points
        self.trunk = self.build_trunk(configs['trunk'])
        self.cp_generator = self.build_cp_gen(configs['cp_gen'])
        self.w_generator = self.build_w_gen(configs['w_gen'])
    
    def forward(self, input):
        node = self.trunk(input)
        cp = self.cp_generator(node)
        w = self.w_generator(node)
        return cp, w
    
    def build_trunk(self, configs):
        pass

    def build_cp_gen(self, configs):
        pass

    def build_w_gen(self, configs):
        pass

class BezierGenerator(nn.Module):
    def __init__(self, in_features, n_control_points, n_data_points, configs):
        super().__init__()
        self.in_features = in_features
        self.n_control_points = n_control_points
        self.n_data_points = n_data_points
        self.cpw_generator = self.build_cpw_gen(configs['cpw'])
        self.feature_generator = FeatureGenerator(in_features, configs['feature'])
        self.bezier_layer = BezierLayer(20, 20, 192)
    
    def forward(self, input):
        cp, w = self.cpw_generator(input)
        features = self.feature_generator(input)
        dp, pv, intvls = self.bezier_layer(features, cp, w)
        return dp, cp, w, pv, intvls

class DiscriminatorEB(nn.Module):
    def __init__(self):
        super().__init__()
        self.discriminator