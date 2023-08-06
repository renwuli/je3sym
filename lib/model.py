from functools import partial

import numpy as np
import jittor as jt
import jittor.nn as nn

from .layers import SoftTopK, generate_8_rotations, generate_24_rotations, group
from .ops import get_clusters, extract_cluster_centriod


def symgroup(plane, thre, nsample, min_cluster_size):
    # pos:      [b, n, 3]
    # plane:    [b, n, 4]
    b, n, _ = plane.size()
    flatten_plane = plane.reshape(b * n, -1)
    flatten_batch = jt.arange(b).unsqueeze(-1).expand(-1, n).flatten()
    plane_clusters, plane_edges = get_clusters(
        flatten_plane, flatten_batch, thre, nsample, min_cluster_size
    )
    cluster_indices = extract_cluster_centriod(
        flatten_plane, plane_edges, flatten_batch, plane_clusters
    )
    cluster_plane = flatten_plane[cluster_indices]
    cluster_batch = flatten_batch[cluster_indices]
    return cluster_plane, cluster_batch


class PointEncoder(nn.Module):
    def __init__(self, mlps=[], k=64, radius=0.15, rotations=24) -> None:
        super(PointEncoder, self).__init__()
        self.k = k
        self.radius = radius

        mlps = [3] + mlps
        self.net = []
        for i in range(len(mlps) - 1):
            self.net.append(nn.Conv3d(mlps[i], mlps[i + 1], 1))
            self.net.append(nn.ReLU())
        self.net = nn.Sequential(*self.net)

        if rotations == 24:
            self.register_buffer("Rs", generate_24_rotations())
        elif rotations == 8:
            self.register_buffer("Rs", generate_8_rotations())

    def execute(self, x, y=None):
        # x: [b, n, 3]
        if y is None:
            y = x
        group_points = group(x, y, self.k, self.radius)  # [b, n, k, 3]
        group_points = jt.einsum(
            "bnkj,ijj->bnikj", group_points, self.Rs
        )  # [b, n, m, k, 3]
        group_points = group_points.permute(
            0, 4, 1, 2, 3
        ).contiguous()  # [b, 3, n, m, k]
        f = self.net(group_points)  # type: ignore # [b, 32, n, m, k]
        f = jt.max(f, dim=-1)  # [b, 32, n, m]
        f = f.mean(-1)  # [b, 32, n]
        return f


class SymmetryNet(nn.Module):
    def __init__(
        self, mlps, ks, radius, rotations, thre, nsample, min_cluster_size, topk=1
    ):
        super(SymmetryNet, self).__init__()

        self.mlps = mlps
        self.ks = ks
        self.radius = radius
        self.rotations = rotations

        self.encoder_num = 0
        self.encoders = nn.ModuleList()
        for k, r in zip(ks, radius):
            self.encoders.append(PointEncoder(mlps, k, r, rotations=rotations))
            self.encoder_num += 1

        self.dim = mlps[-1] * self.encoder_num

        self.topk = SoftTopK(topk + 1)

        self.cluster = partial(
            symgroup, thre=thre, nsample=nsample, min_cluster_size=min_cluster_size
        )

    def dense_predict(self, pos):
        batch_size, n, _ = pos.size()
        plane = []
        for i in range(self.encoder_num):
            f = self.encoders[i](pos).transpose(1, 2).contiguous()

            dists = f.unsqueeze(1).expand(-1, n, -1, -1) - f.unsqueeze(2).expand(
                -1, -1, n, -1
            )
            dists = jt.sum(jt.sqrt(jt.abs(dists)), -1) ** 2

            correspondence = self.topk(dists)[:, :, 1:, :]  # [b, n, k, n]
            correspondence = (
                correspondence.transpose(1, 2).contiguous().reshape(batch_size, -1, n)
            )  # [b, k, n, n]
            pos_prime = jt.matmul(
                correspondence, pos
            )  # [b,  k * n, n] * [b, n, 3] -> [b, k * n, 3]

            plane_normal = jt.normalize(
                pos_prime - pos.repeat(1, pos_prime.size(1) // n, 1), dim=-1
            )
            plane_footpoint = (
                pos.repeat(1, pos_prime.size(1) // n, 1) + pos_prime
            ) * 0.5
            plane_d = -jt.mul(plane_normal, plane_footpoint).sum(-1, keepdim=True)
            plane.append(jt.cat([plane_normal, plane_d], -1))

        # plane: [b, 2n, 4]
        plane = jt.cat(plane, dim=1)
        return plane

    def execute(self, pos):
        plane = self.dense_predict(pos)

        # cluster
        return self.cluster(plane)
