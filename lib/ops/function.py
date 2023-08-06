import os
import os.path as osp
import math
import glob
import numpy as np
import jittor as jt


def optimal_block(batch_size):
    return 2 ** int(math.log(batch_size))


class BuildGraphBatch(jt.Function):
    cuda_header = """
    #include <math.h>
    """
    cuda_src = """
        __global__ void build_graph_batch_kernel(@ARGS_DEF) {
            @PRECALC
            const ptrdiff_t batch_idx = blockIdx.x;
            const ptrdiff_t start_idx_x = @in1(batch_idx);
            const ptrdiff_t start_idx_y = @in1(batch_idx);
            const ptrdiff_t end_idx_x = @in1(batch_idx + 1);
            const ptrdiff_t end_idx_y = @in1(batch_idx + 1);

            for (ptrdiff_t n_y = start_idx_y + threadIdx.x; n_y < end_idx_y; n_y += blockDim.x)
            {
                int32_t cnt = 0;
                float yx = @in0(n_y, 0);
                float yy = @in0(n_y, 1);
                float yz = @in0(n_y, 2);
                float yd = @in0(n_y, 3);


                for (ptrdiff_t n_x = start_idx_x; n_x < end_idx_x; n_x++)
                {
                    float xx = @in0(n_x, 0);
                    float xy = @in0(n_x, 1);
                    float xz = @in0(n_x, 2);
                    float xd = @in0(n_x, 3);


                    float plane_distance1 = sqrt((xx - yx) * (xx - yx) + (xy - yy) * (xy - yy) + (xz - yz) * (xz - yz) + (xd - yd) * (xd - yd));

                    float plane_distance2 = sqrt((xx + yx) * (xx + yx) + (xy + yy) * (xy + yy) + (xz + yz) * (xz + yz)+ (xd + yd) * (xd + yd));

                    float plane_distance = min(plane_distance1, plane_distance2);

                    if (plane_distance < #thre) 
                    {
                        @out0(n_y, cnt) = n_x;
                        @out1(n_y, cnt) = plane_distance;
                        @out2(n_y, cnt) = 1.0 - plane_distance;
                        cnt++;
                    }
                    if (cnt >= #nsample)
                    {
                        break;
                    }
                }
            }
        }
        int batch_size = #batch_size;
        int block_size = #block_size;
        build_graph_batch_kernel<<<batch_size, block_size>>>(@ARGS);
    """
    def execute(self, x, batch_x, thre, nsample):
        """
        x:          [N, 3]
        batch:      [N, ]
        thre:       float
        nsample:    int
        """

        def degree(row, num_nodes):
            zero = jt.zeros(num_nodes, row.dtype)
            one = jt.ones((row.size(0)), row.dtype)
            zero.scatter_(0, row, one, reduce="add")
            return zero

        size_x = x.size(0)
        assert size_x == batch_x.size(
            0
        ), f"size of x {size_x} should match size of batch_x {batch_x.size(0)}"
        assert nsample <= size_x
        assert x.dtype == "float32"
        assert batch_x.dtype == "int32"

        batch_size = batch_x[-1].item() + 1
        batch_x = degree(batch_x, batch_size)
        batch_x = jt.cat([jt.zeros(1, batch_x.dtype), batch_x.cumsum(0)], 0)

        block_size = optimal_block(size_x)

        cuda_header = self.cuda_header
        cuda_src = (
            self.cuda_src.replace("#batch_size", str(batch_size))
            .replace("#block_size", str(block_size))
            .replace("#thre", str(thre))
            .replace("#nsample", str(nsample))
        )
        idx = jt.full((size_x, nsample), -1, dtype="int32")
        dist = jt.full((size_x, nsample), -1, dtype="float32")
        edge = jt.zeros((size_x, nsample), -1, dtype="float32")

        jt.code(
            inputs=[x, batch_x],
            outputs=[idx, dist, edge],
            cuda_src=cuda_src,
            cuda_header=cuda_header,
        )
        return idx, dist, edge


build_graph_batch = BuildGraphBatch.apply


def find_connected_component(neighbours, min_cluster_size):
    num_points = neighbours.shape[0]
    visited = jt.zeros((num_points,), dtype=bool)
    clusters = []
    for i in range(num_points):
        if visited[i]:
            continue

        cluster = []
        queue = []
        visited[i] = True
        queue.append(i)
        cluster.append(i)

        while len(queue):
            k = queue.pop()
            k_neighbours = neighbours[k]
            for nei in k_neighbours:
                if nei.item() == -1:
                    break

                if not visited[nei]:
                    visited[nei] = True
                    queue.append(nei.item())
                    cluster.append(nei.item())

        if len(cluster) >= min_cluster_size:
            clusters.append(cluster)

    return clusters


def get_clusters(pos, batch, thre, nsample, min_cluster_size):
    """
    pos:        [N, 3]
    batch:      [N]
    """
    assert pos.shape[0] == batch.shape[0]
    graph = build_graph_batch(pos, batch, thre, nsample)
    neighbours = graph[0].cpu().numpy()
    edges = graph[2]
    clusters = find_connected_component(neighbours, min_cluster_size)

    return clusters, edges


def extract_cluster_centriod(pos, edge, batch, clusters):
    """
    pos:        [N, 3], tensor
    edge:       [N, K], tenor
    batch:      [N], tensor
    clusters:   [M], list
    """
    assert pos.shape[0] == batch.shape[0]
    num_clusters = len(clusters)
    if 0 == num_clusters:
        return jt.int32([0])

    indices = []
    for i, cluster in enumerate(clusters):
        idx = cluster[jt.argmax(edge[cluster].sum(-1), dim=0)[0]]
        indices.append(idx)

    indices = jt.int32(indices)
    return indices
