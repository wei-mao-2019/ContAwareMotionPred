import torch.nn as nn

from pvcnn.models.utils import create_pointnet2_sa_components, create_pointnet2_fp_modules, create_mlp_components

__all__ = ['PVCNN2']


class PVCNN2(nn.Module):

    def __init__(self, num_classes, extra_feature_channels=6, width_multiplier=1, voxel_resolution_multiplier=1,
                 sa_blocks=None, fp_blocks=None, with_classifier=True, is_bn=True):
        super().__init__()
        if sa_blocks is None:
            self.sa_blocks = [
                ((32, 2, 32), (1024, 0.1, 32, (32, 64))),
                ((64, 3, 16), (256, 0.2, 32, (64, 128))),
                ((128, 3, 8), (64, 0.4, 32, (128, 256))),
                (None, (16, 0.8, 32, (256, 256, 512))),
            ]
            self.fp_blocks = [
                ((256, 256), (256, 1, 8)),
                ((256, 256), (256, 1, 8)),
                ((256, 128), (128, 2, 16)),
                ((128, 128, 64), (64, 1, 32)),
            ]
        else:
            self.sa_blocks = sa_blocks
            self.fp_blocks = fp_blocks
        self.with_classifier = with_classifier
        self.in_channels = extra_feature_channels + 3

        sa_layers, sa_in_channels, channels_sa_features, _ = create_pointnet2_sa_components(
            sa_blocks=self.sa_blocks, extra_feature_channels=extra_feature_channels, with_se=True,
            width_multiplier=width_multiplier, voxel_resolution_multiplier=voxel_resolution_multiplier,
            is_bn=is_bn
        )
        self.sa_layers = nn.ModuleList(sa_layers)

        # only use extra features in the last fp module
        sa_in_channels[0] = extra_feature_channels
        fp_layers, channels_fp_features = create_pointnet2_fp_modules(
            fp_blocks=self.fp_blocks, in_channels=channels_sa_features, sa_in_channels=sa_in_channels, with_se=True,
            width_multiplier=width_multiplier, voxel_resolution_multiplier=voxel_resolution_multiplier,
            is_bn=is_bn
        )
        self.fp_layers = nn.ModuleList(fp_layers)
        if with_classifier:
            layers, _ = create_mlp_components(in_channels=channels_fp_features, out_channels=[128, 0.5, num_classes],
                                              classifier=True, dim=2, width_multiplier=width_multiplier,
                                              is_bn=is_bn)
            self.classifier = nn.Sequential(*layers)

    def forward(self, inputs):
        if isinstance(inputs, dict):
            inputs = inputs['features']

        coords, features = inputs[:, :3, :].contiguous(), inputs
        coords_list, in_features_list = [], []
        for sa_blocks in self.sa_layers:
            in_features_list.append(features)
            coords_list.append(coords)
            features, coords = sa_blocks((features, coords))
        in_features_list[0] = inputs[:, 3:, :].contiguous()

        for fp_idx, fp_blocks in enumerate(self.fp_layers):
            features, coords = fp_blocks((coords_list[-1-fp_idx], coords, features, in_features_list[-1-fp_idx]))
        if self.with_classifier:
            return self.classifier(features)
        else:
            return features


class PVCNN2_v1(nn.Module):

    def __init__(self, extra_feature_channels=3, width_multiplier=1, voxel_resolution_multiplier=1, sa_blocks=None):
        super().__init__()
        if sa_blocks is None:
            self.sa_blocks = [
                ((32, 2, 32), (1024, 0.1, 32, (32, 64))),
                ((64, 3, 16), (256, 0.2, 32, (64, 128))),
                ((128, 3, 8), (64, 0.4, 32, (128, 128))),
                (None, (16, 0.8, 32, (128, 128, 128)))
            ]
        else:
            self.sa_blocks = sa_blocks

        self.in_channels = extra_feature_channels + 3

        sa_layers, sa_in_channels, channels_sa_features, _ = create_pointnet2_sa_components(
            sa_blocks=self.sa_blocks, extra_feature_channels=extra_feature_channels, with_se=True,
            width_multiplier=width_multiplier, voxel_resolution_multiplier=voxel_resolution_multiplier
        )
        self.sa_layers = nn.ModuleList(sa_layers)
        self.sa_in_channels = sa_in_channels
        self.channels_sa_features = channels_sa_features


    def forward(self, inputs):
        if isinstance(inputs, dict):
            inputs = inputs['features']

        coords, features = inputs[:, :3, :].contiguous(), inputs
        for sa_blocks in self.sa_layers:
            features, coords = sa_blocks((features, coords))

        return features, coords

class PVCNN2_SA(nn.Module):

    def __init__(self, num_classes, extra_feature_channels=6, width_multiplier=1, voxel_resolution_multiplier=1,
                 sa_blocks=None, fp_blocks=None, with_classifier=True, is_bn=True):
        super().__init__()
        if sa_blocks is None:
            self.sa_blocks = [
                ((32, 2, 32), (1024, 0.1, 32, (32, 64))),
                ((64, 3, 16), (256, 0.2, 32, (64, 128))),
                ((128, 3, 8), (64, 0.4, 32, (128, 256))),
                (None, (16, 0.8, 32, (256, 256, 512))),
            ]
            self.fp_blocks = [
                ((256, 256), (256, 1, 8)),
                ((256, 256), (256, 1, 8)),
                ((256, 128), (128, 2, 16)),
                ((128, 128, 64), (64, 1, 32)),
            ]
        else:
            self.sa_blocks = sa_blocks
            self.fp_blocks = fp_blocks
        self.with_classifier = with_classifier
        self.in_channels = extra_feature_channels + 3

        sa_layers, sa_in_channels, channels_sa_features, _ = create_pointnet2_sa_components(
            sa_blocks=self.sa_blocks, extra_feature_channels=extra_feature_channels, with_se=True,
            width_multiplier=width_multiplier, voxel_resolution_multiplier=voxel_resolution_multiplier,
            is_bn=is_bn
        )
        self.sa_layers = nn.ModuleList(sa_layers)

        # only use extra features in the last fp module
        sa_in_channels[0] = extra_feature_channels

        self.sa_in_channels = sa_in_channels
        self.channels_sa_features = channels_sa_features

    def forward(self, inputs):
        if isinstance(inputs, dict):
            inputs = inputs['features']

        coords, features = inputs[:, :3, :].contiguous(), inputs
        coords_list, in_features_list = [], []
        for sa_blocks in self.sa_layers:
            in_features_list.append(features)
            coords_list.append(coords)
            features, coords = sa_blocks((features, coords))
        in_features_list[0] = inputs[:, 3:, :].contiguous()

        return coords_list,coords,features,in_features_list

class PVCNN2_FP(nn.Module):

    def __init__(self, num_classes, extra_feature_channels=6, width_multiplier=1, voxel_resolution_multiplier=1,
                 sa_blocks=None, fp_blocks=None, with_classifier=True, is_bn=True,
                 channels_sa_features=256, sa_in_channels=None):
        super().__init__()
        if sa_blocks is None:
            self.sa_blocks = [
                ((32, 2, 32), (1024, 0.1, 32, (32, 64))),
                ((64, 3, 16), (256, 0.2, 32, (64, 128))),
                ((128, 3, 8), (64, 0.4, 32, (128, 256))),
                (None, (16, 0.8, 32, (256, 256, 512))),
            ]
            self.fp_blocks = [
                ((256, 256), (256, 1, 8)),
                ((256, 256), (256, 1, 8)),
                ((256, 128), (128, 2, 16)),
                ((128, 128, 64), (64, 1, 32)),
            ]
        else:
            self.sa_blocks = sa_blocks
            self.fp_blocks = fp_blocks
        self.in_channels = extra_feature_channels + 3

        # only use extra features in the last fp module
        sa_in_channels[0] = extra_feature_channels
        fp_layers, channels_fp_features = create_pointnet2_fp_modules(
            fp_blocks=self.fp_blocks, in_channels=channels_sa_features, sa_in_channels=sa_in_channels, with_se=True,
            width_multiplier=width_multiplier, voxel_resolution_multiplier=voxel_resolution_multiplier,
            is_bn=is_bn
        )
        self.fp_layers = nn.ModuleList(fp_layers)

    def forward(self, coords_list,coords,features,in_features_list):
        # if isinstance(inputs, dict):
        #     inputs = inputs['features']

        # coords, features = inputs[:, :3, :].contiguous(), inputs
        # coords_list, in_features_list = [], []
        # for sa_blocks in self.sa_layers:
        #     in_features_list.append(features)
        #     coords_list.append(coords)
        #     features, coords = sa_blocks((features, coords))
        # in_features_list[0] = inputs[:, 3:, :].contiguous()

        for fp_idx, fp_blocks in enumerate(self.fp_layers):
            features, coords = fp_blocks((coords_list[-1-fp_idx], coords, features, in_features_list[-1-fp_idx]))
        # if self.with_classifier:
        #     return self.classifier(features)
        # else:
        return features