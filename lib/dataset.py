import os
import os.path as osp
import open3d as o3d
import progressbar
import numpy as np
import pickle
import scipy
import trimesh

import jittor as jt
from jittor.dataset import Dataset

from .util import read_point_cloud


def scene_as_mesh(scene):
    if isinstance(scene, trimesh.Scene):
        mesh = trimesh.util.concatenate(
            [
                trimesh.Trimesh(vertices=m.vertices, faces=m.faces)
                for m in scene.geometry.values()
            ]
        )
    else:
        mesh = scene

    return mesh


id2cat = {
    "02691156": "airplane",
    "02747177": "trash bin",
    "02773838": "bag",
    "02801938": "basket",
    "02808440": "bathtub",
    "02818832": "bed",
    "02828884": "bench",
    "02843684": "birdhouse",
    "02871439": "bookshelf",
    "02876657": "bottle",
    "02880940": "bowl",
    "02924116": "bus",
    "02933112": "cabinet",
    "02942699": "camera",
    "02946921": "can",
    "02954340": "cap",
    "02958343": "car",
    "02992529": "cellphone",
    "03001627": "chair",
    "03046257": "clock",
    "03085013": "keyboard",
    "03207941": "dishwasher",
    "03211117": "display",
    "03261776": "earphone",
    "03325088": "faucet",
    "03337140": "file cabinet",
    "03467517": "guitar",
    "03513137": "helmet",
    "03593526": "jar",
    "03624134": "knife",
    "03636649": "lamp",
    "03642806": "laptop",
    "03691459": "loudspeaker",
    "03710193": "mailbox",
    "03759954": "microphone",
    "03761084": "microwaves",
    "03790512": "motorbike",
    "03797390": "mug",
    "03928116": "piano",
    "03938244": "pillow",
    "03948459": "pistol",
    "03991062": "flowerpot",
    "04004475": "printer",
    "04074963": "remote",
    "04090263": "rifle",
    "04099429": "rocket",
    "04225987": "skateboard",
    "04256520": "sofa",
    "04330267": "stove",
    "04379243": "table",
    "04401088": "telephone",
    "04460130": "tower",
    "04468005": "train",
    "04530566": "watercraft",
    "04554684": "washer",
}

cat2id = {key: value for value, key in id2cat.items()}


def voxel_downsample(point_cloud, size=0.02, npoints=-1, max_iters=int(1e2)):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud[:, :3])
    normals = point_cloud.shape[1] == 6
    if normals:
        pcd.normals = o3d.utility.Vector3dVector(point_cloud[:, 3:])
    if npoints == -1:
        pcd = pcd.voxel_down_sample(size)
        return np.array(pcd.points).astype(np.float32)

    upper = 0.0001
    lower = 0.5
    for i in range(max_iters):
        mid = (upper + lower) / 2
        tmp = pcd.voxel_down_sample(mid)

        # minimal grid quantization, maximal resolution
        if np.asanyarray(tmp.points).shape[0] > npoints:
            upper = mid
        else:
            lower = mid
    if normals:
        pts = np.array(tmp.points).astype(np.float32)
        nrms = np.array(tmp.normals).astype(np.float32)
        return np.concatenate([pts, nrms], axis=-1)
    else:
        pts = np.array(tmp.points)
        if pts.shape[0] >= npoints:
            pts = pts[:npoints]
        else:
            pad_indices = np.random.choice(
                pts.shape[0], npoints - pts.shape[0], replace=False
            )
            pad_pts = pts[pad_indices]
            pts = np.concatenate([pts, pad_pts])
        return pts.astype(np.float32)


class ShapeNet(Dataset):
    def __init__(self, root, data_split, phase="train", cat="all", npoints=512):
        super(ShapeNet, self).__init__()
        self.root = root
        self.phase = phase
        self.cat = cat
        self.npoints = npoints

        self.data_pathes_archive = osp.join(data_split, f"{cat}_{self.phase}.pkl")
        if osp.exists(self.data_pathes_archive):
            with open(self.data_pathes_archive, "rb") as f:
                self.data_pathes = pickle.load(f)
        else:
            self.data_pathes = []
            if self.cat == "all":
                for txt in os.listdir(data_split):
                    if txt.count(self.phase):
                        with open(osp.join(data_split, txt), "r") as f:
                            lines = f.readlines()
                            for line in lines:
                                path = osp.join(
                                    root,
                                    txt.split("_")[0],
                                    line.strip(),
                                    "models",
                                    "model_normalized.obj",
                                )
                                if osp.exists(path):
                                    self.data_pathes.append(path)
            else:
                for txt in os.listdir(data_split):
                    if txt.count(cat2id[self.cat]) and txt.count(self.phase):
                        with open(osp.join(data_split, txt), "r") as f:
                            lines = f.readlines()
                            for line in lines:
                                path = osp.join(
                                    root,
                                    txt.split("_")[0],
                                    line.strip(),
                                    "models",
                                    "model_normalized.obj",
                                )
                                if osp.exists(path):
                                    self.data_pathes.append(path)
            with open(self.data_pathes_archive, "wb") as f:
                pickle.dump(self.data_pathes, f)

    def __len__(self):
        return len(self.data_pathes)

    def __getitem__(self, index):
        model_path = self.data_pathes[index]
        mesh = trimesh.load_mesh(model_path, force="mesh")
        mesh = scene_as_mesh(mesh)
        dense_points = np.array(trimesh.sample.sample_surface(mesh, 10000)[0])

        R = scipy.spatial.transform.Rotation.random().as_matrix()
        dense_points = dense_points @ R
        points = voxel_downsample(dense_points, npoints=self.npoints)
        return points


class ShapeNetEval(Dataset):
    def __init__(self, root, npoints):
        super(ShapeNetEval, self).__init__()
        self.root = root
        self.npoints = npoints

        self.datas = []
        self.data_pathes = []
        txt = osp.join(osp.dirname(root), "1000.txt")
        with open(txt, "r") as f:
            lines = f.readlines()
            for line in lines:
                model_id, plane_num = line.strip().split(" ")
                self.datas.append((model_id, plane_num))
                self.data_pathes.append(osp.join(root, model_id + ".obj"))

        self.sample()

    def __len__(self):
        return len(self.data_pathes)

    def sample(self):
        for index in progressbar.progressbar(range(len(self.data_pathes))):
            model_path = self.data_pathes[index]
            pointcloud_path = model_path.replace(
                ".obj", ".ply" if self.npoints == 512 else f"_{self.npoints}.ply"
            )
            if osp.exists(pointcloud_path):
                continue

            mesh = trimesh.load_mesh(model_path, force="mesh")
            mesh = scene_as_mesh(mesh)
            dense_points = np.array(trimesh.sample.sample_surface(mesh, 10000)[0])
            points = voxel_downsample(dense_points, npoints=self.npoints, size=0.01)
            points = points.cpu().numpy()
            pointcloud = o3d.geometry.PointCloud()
            pointcloud.points = o3d.utility.Vector3dVector(points)

            o3d.io.write_point_cloud(pointcloud_path, pointcloud)

    def __getitem__(self, index):
        model_path = self.data_pathes[index]
        pointcloud_path = model_path.replace(
            ".obj", ".ply" if self.npoints == 512 else f"_{self.npoints}.ply"
        )
        points = read_point_cloud(pointcloud_path)
        return points
