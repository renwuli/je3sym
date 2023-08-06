import math
import numpy as np
import jittor as jt
import jittor.nn as nn


def optimal_block(batch_size):
    return 2 ** int(math.log(batch_size))


class BallQuery(jt.Function):
    cuda_src = """
        __global__ void query_ball_point_kernel(int b, int n, int m, float radius,
                                                int nsample,
                                                const float *__restrict__ new_xyz,
                                                const float *__restrict__ xyz,
                                                int *__restrict__ idx,
                                                int *__restrict__ cnt) {
            int batch_index = blockIdx.x;
            xyz += batch_index * n * 3;
            new_xyz += batch_index * m * 3;
            idx += m * nsample * batch_index;
            cnt += batch_index * m;

            int index = threadIdx.x;
            int stride = blockDim.x;

            float radius2 = radius * radius;
            for (int j = index; j < m; j += stride) {
                float new_x = new_xyz[j * 3 + 0];
                float new_y = new_xyz[j * 3 + 1];
                float new_z = new_xyz[j * 3 + 2];
                cnt[j] = 0;

                for (int k = 0; k < n && cnt[j] < nsample; ++k) {
                    float x = xyz[k * 3 + 0];
                    float y = xyz[k * 3 + 1];
                    float z = xyz[k * 3 + 2];
                    float d2 = (new_x - x) * (new_x - x) + (new_y - y) * (new_y - y) +
                                (new_z - z) * (new_z - z);

                    if (d2 < radius2) {
                        if (cnt[j] == 0) {
                            for (int l = 0; l < nsample; ++l)
                                idx[j * nsample + l] = k;
                        }
                        idx[j * nsample + cnt[j]] = k;
                        ++cnt[j];
                    }
                }
            }
        }

        int block_size = #block_size;

        query_ball_point_kernel<<<in0_shape0, block_size>>>(
            in0_shape0, in1_shape1, in0_shape1, #radius, #nsample,
            in0_p, in1_p, out0_p, out1_p
        );
    """

    def __init__(self):
        super().__init__()

    def execute(self, new_xyz, pointset, radius, nsamples):
        """
        Parameters
        ----------
        xyz: jt.Var, (B, N, 3)

        Returns
        -------
        new_points: jt.Var, (B, N, nsamples, 3)
        """
        batch_size_x, n_input, n_coords = new_xyz.shape
        assert n_coords == 3

        batch_size_p, n_points, n_coords = pointset.shape
        assert n_coords == 3
        assert batch_size_x == batch_size_p

        block_size = optimal_block(batch_size_x)

        cuda_src = (
            self.cuda_src.replace("#block_size", str(block_size))
            .replace("#radius", str(radius))
            .replace("#nsample", str(nsamples))
        )

        idxs_shape = [batch_size_x, n_input, nsamples]
        cnts_shape = [batch_size_x, n_input]
        idxs, cnts = jt.code(
            [idxs_shape, cnts_shape],
            ["int32", "int32"],
            [new_xyz, pointset],
            cuda_src=cuda_src,
        )

        pc_shape = [batch_size_x, n_input, nsamples, 3]
        new_pointset = pointset.reindex(
            pc_shape,
            [
                "i0",
                "@e0(i0, i1, i2)",
                "i3",
            ],
            extras=[idxs],
        )

        return new_pointset


ball_query = BallQuery.apply


def batched_index_select(input, dim, index):
    for i in range(1, len(input.shape)):
        if i != dim:
            index = index.unsqueeze(i)
    expanse = list(input.shape)
    expanse[0] = -1
    expanse[dim] = -1
    index = index.expand(expanse)
    return jt.gather(input, dim, index)


def generate_24_rotations():
    res = []
    for id in [[0, 1, 2], [1, 2, 0], [2, 0, 1]]:
        R = np.identity(3)[:, id].astype(np.float32)
        R1 = np.asarray([R[:, 0], R[:, 1], R[:, 2]]).T
        R2 = np.asarray([-R[:, 0], -R[:, 1], R[:, 2]]).T
        R3 = np.asarray([-R[:, 0], R[:, 1], -R[:, 2]]).T
        R4 = np.asarray([R[:, 0], -R[:, 1], -R[:, 2]]).T
        res += [R1, R2, R3, R4]
    for id in [[0, 2, 1], [1, 0, 2], [2, 1, 0]]:
        R = np.identity(3)[:, id].astype(np.float32)
        R1 = np.asarray([-R[:, 0], -R[:, 1], -R[:, 2]]).T
        R2 = np.asarray([-R[:, 0], R[:, 1], R[:, 2]]).T
        R3 = np.asarray([R[:, 0], -R[:, 1], R[:, 2]]).T
        R4 = np.asarray([R[:, 0], R[:, 1], -R[:, 2]]).T
        res += [R1, R2, R3, R4]
    res = np.stack(res, axis=0)
    res = jt.array(res).float()
    return res


def generate_8_rotations():
    res = []
    R = np.identity(3).astype(np.float32)
    R1 = np.asarray([R[:, 0], R[:, 1], R[:, 2]]).T
    R2 = np.asarray([-R[:, 0], -R[:, 1], R[:, 2]]).T
    R3 = np.asarray([-R[:, 0], R[:, 1], -R[:, 2]]).T
    R4 = np.asarray([R[:, 0], -R[:, 1], -R[:, 2]]).T
    res += [R1, R2, R3, R4]
    R = np.identity(3).astype(np.float32)
    R1 = np.asarray([-R[:, 0], -R[:, 1], -R[:, 2]]).T
    R2 = np.asarray([-R[:, 0], R[:, 1], R[:, 2]]).T
    R3 = np.asarray([R[:, 0], -R[:, 1], R[:, 2]]).T
    R4 = np.asarray([R[:, 0], R[:, 1], -R[:, 2]]).T
    res += [R1, R2, R3, R4]
    res = np.stack(res, axis=0)
    res = jt.array(res).float()
    return res


def pca(points):
    # points: [b, m, k, 3]
    cov = jt.matmul(points.transpose(2, 3), points)
    cov = cov / points.shape[2]
    # cov: [b, m, 3, 3]
    e, v = jt.linalg.eigh(cov)
    # e: [b, m, 3], v: [b, m, 3, 3]
    can_points = jt.matmul(points, v)
    # can_points: [b, m, k, 3]
    return can_points


def group(points, dense_points, K=64, radius=0.15):
    # points: [b, n, 3]
    nn_points = ball_query(points, dense_points, radius, K)
    # nn_points: [b, n, k, 3]
    nn_points = nn_points - points[:, :, None, :]  # [b, n, k, 3]
    nn_points = pca(nn_points)
    return nn_points


class SoftTopK(nn.Module):
    def __init__(self, k: int, num_samples: int = 1, sigma: float = 0.0001):
        super(SoftTopK, self).__init__()
        self.k = k
        self.num_samples = num_samples
        self.sigma = sigma

    def __call__(self, x):
        return SoftTopKFunction.apply(x, self.k, self.num_samples, self.sigma)


class SoftTopKFunction(jt.Function):
    def execute(self, x, k: int, num_samples: int = 1, sigma: float = 0.0001):
        """
        the sigma lower, the topk harder
        """
        b, n, m = x.size()
        noise = jt.normal(mean=0.0, std=1.0, size=(b, n, num_samples, m), dtype=x.dtype)
        perturbed_x = x.unsqueeze(2) + noise * sigma  # [b, n, nsample, m]
        _, indices = jt.topk(perturbed_x, k=k, dim=-1, sorted=False, largest=False)
        perturbed_output = jt.nn.one_hot(
            indices, num_classes=m
        ).float()  # [b, n, nsample, k, m]
        indicators = perturbed_output.mean(dim=2)  # [b, n, k, m]

        self.k = k
        self.num_samples = num_samples
        self.sigma = sigma

        self.perturbed_output = perturbed_output
        self.noise = noise

        return indicators

    def grad(self, grad_output):
        if grad_output is None:
            return tuple([None] * 5)

        noise_gradient = self.noise
        expected_gradient = (
            jt.einsum("bnxkm,bnxm->bnkm", self.perturbed_output, noise_gradient)
            / self.num_samples
            / self.sigma
        )
        grad_input = jt.einsum("bnkm,bnkm->bnm", grad_output, expected_gradient)
        return (grad_input,) + tuple([None] * 5)
