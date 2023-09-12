import jittor as jt
import jittor.nn as nn

from jittor.loss3d import chamfer_loss


class SymmetryDistanceError(nn.Module):
    def __init__(self):
        super(SymmetryDistanceError, self).__init__()

    def _distance(self, pos, plane):
        """
        pos: [n, 3]
        plane: [4]
        """

        normal = plane[:3]
        d = plane[3]

        ref_pos = pos - 2 * (jt.sum(pos * normal, dim=1) + d).unsqueeze(-1) * normal
        return chamfer_loss(ref_pos.unsqueeze(0), pos.unsqueeze(0), bidirectional=True)

    def execute(self, pos, plane, batch):
        """
        pos:    [b, n, 3]
        plane:  [m, 4]
        batch:  [m]
        """
        b, n, _ = pos.size()
        m = plane.size(0)
        assert batch.size(0) == m

        loss = jt.float32(0.0)
        for i in range(m):
            cur_plane = plane[i]
            cur_batch = batch[i]
            cur_pos = pos[cur_batch.item()]
            loss += self._distance(cur_pos, cur_plane)

        loss /= m
        return loss
