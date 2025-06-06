import torch
from smplpytorch.pytorch import rodrigues_layer


def th_posemap_axisang(pose_vectors):
    '''
    Converts axis-angle to rotmat
    pose_vectors (Tensor (batch_size x num_joints * 3)): pose parameters in axis-angle representation
    '''
    rot_nb = int(pose_vectors.shape[1] / 3)
    rot_mats = []
    for joint_idx in range(rot_nb):
        axis_ang = pose_vectors[:, joint_idx * 3:(joint_idx + 1) * 3]
        rot_mat = rodrigues_layer.batch_rodrigues(axis_ang)
        rot_mats.append(rot_mat)

    rot_mats = torch.cat(rot_mats, 1)
    return rot_mats


def th_with_zeros(tensor):
    batch_size = tensor.shape[0]
    padding = tensor.new_tensor([0.0, 0.0, 0.0, 1.0])
    padding.requires_grad = False

    concat_list = [tensor, padding.view(1, 1, 4).repeat(batch_size, 1, 1)]
    cat_res = torch.cat(concat_list, 1)
    return cat_res


def th_pack(tensor):
    batch_size = tensor.shape[0]
    padding = torch.zeros((batch_size, 4, 3), dtype=tensor.dtype, device=tensor.device)
    padding.requires_grad = False
    pack_list = [padding, tensor]
    pack_res = torch.cat(pack_list, 2)
    return pack_res


def subtract_flat_id(rot_mats):
    # Subtracts identity as a flattened tensor
    # rot_mats is expected to be (batch_size, num_articulated_joints * 9)
    # num_articulated_joints is (total_num_joints - 1)
    
    num_repeats = int(rot_mats.shape[1] / 9) # Calculate repeats dynamically

    id_flat = torch.eye(
        3, dtype=rot_mats.dtype, device=rot_mats.device).view(1, 9).repeat(
            rot_mats.shape[0], num_repeats) # Use calculated num_repeats
    # id_flat.requires_grad = False # Identity matrix usually doesn't require grad
                                  # but subtracting it from a tensor that requires grad
                                  # will result in a tensor that requires grad.
                                  # Setting requires_grad on id_flat might not be necessary
                                  # or could be set based on rot_mats.requires_grad
    results = rot_mats - id_flat
    return results


def make_list(tensor):
    return tensor
