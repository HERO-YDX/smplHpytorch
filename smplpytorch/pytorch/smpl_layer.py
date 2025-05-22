import os
import numpy as np
import torch
from torch.nn import Module
import scipy.sparse

from smplpytorch.native.webuser.serialization import ready_arguments
from smplpytorch.pytorch import rodrigues_layer
from smplpytorch.pytorch.tensutils import (th_posemap_axisang, th_with_zeros, th_pack, make_list, subtract_flat_id)


class SMPL_Layer(Module):
    """
    PyTorch layer for the Skinned Multi-Person Linear (SMPL) model.
    Handles loading SMPL/SMPL-H model parameters and performs the forward pass
    to compute posed and shaped mesh vertices and joint locations.
    """
    __constants__ = ['kintree_parents', 'gender', 'center_idx', 'num_joints', 'model_type']

    def __init__(self,
                 center_idx=None,
                 gender='neutral',
                 model_root='smpl/native/models', # Default root for SMPL models
                 smplh_model_root='smplh/native/models', # Specify a root for SMPL-H models
                 model_type='smpl'):
        """
        Args:
            center_idx (int, optional): Index of the joint to center the model on if no translation is provided.
            gender (str, optional): Gender of the model ('neutral', 'female', 'male'). Defaults to 'neutral'.
            model_root (str, optional): Path to the directory containing SMPL model files (.pkl).
            smplh_model_root (str, optional): Path to the directory containing SMPL-H model files (.pkl).
            model_type (str, optional): Type of model to load ('smpl' or 'smplh'). Defaults to 'smpl'.
        """
        super().__init__()

        self.center_idx = center_idx
        self.gender = gender
        self.model_type = model_type

        if model_type == 'smpl':
            if gender == 'neutral':
                self.model_path = os.path.join(model_root, 'basicModel_neutral_lbs_10_207_0_v1.0.0.pkl')
            elif gender == 'female':
                self.model_path = os.path.join(model_root, 'basicModel_f_lbs_10_207_0_v1.0.0.pkl')
            elif gender == 'male':
                self.model_path = os.path.join(model_root, 'basicModel_m_lbs_10_207_0_v1.0.0.pkl')
            else:
                raise ValueError(f"Unsupported gender: {gender} for model_type: {model_type}")
        elif model_type == 'smplh':
            if gender == 'female':
                self.model_path = os.path.join(smplh_model_root, 'SMPLH_FEMALE.pkl')
            elif gender == 'male':
                self.model_path = os.path.join(smplh_model_root, 'SMPLH_MALE.pkl')
            elif gender == 'neutral': # SMPL-H typically does not have a dedicated neutral model
                print(f"Warning: Neutral gender not typically provided for SMPL-H. Using 'male' as default for SMPL-H neutral.")
                self.model_path = os.path.join(smplh_model_root, 'SMPLH_MALE.pkl')
            else:
                raise ValueError(f"Unsupported gender: {gender} for model_type: {model_type}")
        else:
            raise ValueError(f"Unsupported model_type: {model_type}. Choose 'smpl' or 'smplh'.")

        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found: {self.model_path}")

        # Load SMPL model data using ready_arguments
        smpl_data = ready_arguments(self.model_path)

        self.kintree_table = smpl_data['kintree_table']
        parents_np_data = self.kintree_table[0]
        if not isinstance(parents_np_data, np.ndarray):
             parents_np_data = np.array(parents_np_data)
        self.kintree_parents = list(parents_np_data.astype(np.int32).tolist())
        self.num_joints = len(self.kintree_parents)

        v_template_raw = smpl_data['v_template']
        v_template_np = v_template_raw.r if hasattr(v_template_raw, 'r') else np.array(v_template_raw)
        if v_template_np.ndim != 2 or v_template_np.shape[1] != 3:
            raise ValueError(f"v_template has unexpected shape {v_template_np.shape}. Expected (V, 3).")
        # Register v_template as a buffer: (1, num_vertices, 3)
        self.register_buffer('th_v_template', torch.tensor(v_template_np, dtype=torch.float32).unsqueeze(0))
        num_model_vertices = v_template_np.shape[0]

        # --- Betas and Shapedirs ---
        betas_raw = smpl_data['betas']
        betas_np = betas_raw.r if hasattr(betas_raw, 'r') else np.array(betas_raw)
        if betas_np.ndim != 1:
            raise ValueError(f"Betas have unexpected ndim {betas_np.ndim}. Expected 1.")
        # Register betas as a buffer: (1, num_betas)
        self.register_buffer('th_betas', torch.tensor(betas_np, dtype=torch.float32).unsqueeze(0))
        num_model_betas = betas_np.shape[0]

        shapedirs_raw = smpl_data['shapedirs']
        loaded_shapedirs_np = shapedirs_raw.r if hasattr(shapedirs_raw, 'r') else np.array(shapedirs_raw)
        loaded_shapedirs_tensor = torch.tensor(loaded_shapedirs_np, dtype=torch.float32)
        
        final_shapedirs_for_buffer = None
        # Ensure shapedirs are (num_vertices * 3, num_betas)
        if loaded_shapedirs_tensor.ndim == 3: # (V, 3, num_betas)
            V_file, C_file, B_file = loaded_shapedirs_tensor.shape
            if V_file != num_model_vertices: raise ValueError(f"Shapedirs V dim {V_file} != v_template V dim {num_model_vertices}")
            if C_file != 3: raise ValueError(f"Shapedirs C dim {C_file} != 3")
            if B_file != num_model_betas: raise ValueError(f"Shapedirs Betas dim {B_file} != model betas dim {num_model_betas}")
            final_shapedirs_for_buffer = loaded_shapedirs_tensor.reshape(V_file * C_file, B_file)
            # print(f"INFO: Reshaped shapedirs from {loaded_shapedirs_tensor.shape} to {final_shapedirs_for_buffer.shape}")
        elif loaded_shapedirs_tensor.ndim == 2: # (V*3, num_betas)
            VC_file, B_file = loaded_shapedirs_tensor.shape
            if VC_file != num_model_vertices * 3: raise ValueError(f"Shapedirs VC dim {VC_file} != v_template V*C dim {num_model_vertices*3}")
            if B_file != num_model_betas: raise ValueError(f"Shapedirs Betas dim {B_file} != model betas dim {num_model_betas}")
            final_shapedirs_for_buffer = loaded_shapedirs_tensor
            # print(f"INFO: Using 2D shapedirs as loaded: {final_shapedirs_for_buffer.shape}")
        else:
            raise ValueError(f"Shapedirs has unexpected ndim {loaded_shapedirs_tensor.ndim}. Expected 2 or 3.")
        self.register_buffer('th_shapedirs', final_shapedirs_for_buffer)

        # --- Posedirs ---
        expected_num_pose_basis = (self.num_joints - 1) * 9

        posedirs_raw = smpl_data['posedirs']
        loaded_posedirs_np = posedirs_raw.r if hasattr(posedirs_raw, 'r') else np.array(posedirs_raw)
        loaded_posedirs_tensor = torch.tensor(loaded_posedirs_np, dtype=torch.float32)
        
        final_posedirs_for_buffer = None
        # Ensure posedirs are (num_vertices * 3, (num_joints-1)*9)
        if loaded_posedirs_tensor.ndim == 3: # (V, 3, basis)
            V_file, C_file, B_file = loaded_posedirs_tensor.shape
            if V_file != num_model_vertices: raise ValueError(f"Posedirs V dim {V_file} != v_template V dim {num_model_vertices}")
            if C_file != 3: raise ValueError(f"Posedirs C dim {C_file} != 3")
            if B_file != expected_num_pose_basis: raise ValueError(f"Posedirs Basis dim {B_file} != expected basis {expected_num_pose_basis}")
            final_posedirs_for_buffer = loaded_posedirs_tensor.reshape(V_file * C_file, B_file)
            # print(f"INFO: Reshaped posedirs from {loaded_posedirs_tensor.shape} to {final_posedirs_for_buffer.shape}")
        elif loaded_posedirs_tensor.ndim == 2: # (V*3, basis)
            VC_file, B_file = loaded_posedirs_tensor.shape
            if VC_file != num_model_vertices * 3: raise ValueError(f"Posedirs VC dim {VC_file} != v_template V*C dim {num_model_vertices*3}")
            if B_file != expected_num_pose_basis: raise ValueError(f"Posedirs Basis dim {B_file} != expected basis {expected_num_pose_basis}")
            final_posedirs_for_buffer = loaded_posedirs_tensor
            # print(f"INFO: Using 2D posedirs as loaded: {final_posedirs_for_buffer.shape}")
        else:
            raise ValueError(f"Posedirs has unexpected ndim {loaded_posedirs_tensor.ndim}. Expected 2 or 3.")
        self.register_buffer('th_posedirs', final_posedirs_for_buffer)
        
        # --- J_regressor ---
        # J_regressor can be a sparse matrix
        j_regressor_raw = smpl_data['J_regressor']
        j_regressor_np = None
        data_to_check_sparse = j_regressor_raw.r if hasattr(j_regressor_raw, 'r') else j_regressor_raw
        
        if scipy.sparse.issparse(data_to_check_sparse):
            j_regressor_np = data_to_check_sparse.toarray()
        elif isinstance(data_to_check_sparse, np.ndarray):
            j_regressor_np = data_to_check_sparse
        else:
            j_regressor_np = np.array(data_to_check_sparse)
        
        if j_regressor_np.shape[0] != self.num_joints or j_regressor_np.shape[1] != num_model_vertices:
            raise ValueError(f"J_regressor shape {j_regressor_np.shape} inconsistent with num_joints {self.num_joints} and num_vertices {num_model_vertices}")
        # Register J_regressor as a buffer: (num_joints, num_vertices)
        self.register_buffer('th_J_regressor', torch.tensor(j_regressor_np, dtype=torch.float32))

        # --- Weights (Skinning weights) ---
        weights_raw = smpl_data['weights']
        weights_np = weights_raw.r if hasattr(weights_raw, 'r') else np.array(weights_raw)
        if weights_np.shape[0] != num_model_vertices or weights_np.shape[1] != self.num_joints:
            raise ValueError(f"Weights shape {weights_np.shape} inconsistent with num_vertices {num_model_vertices} and num_joints {self.num_joints}")
        # Register weights as a buffer: (num_vertices, num_joints)
        self.register_buffer('th_weights', torch.tensor(weights_np, dtype=torch.float32))

        # --- Faces ---
        faces_np_data = smpl_data['f']
        if not isinstance(faces_np_data, np.ndarray): faces_np_data = np.array(faces_np_data)
        if faces_np_data.ndim != 2 or faces_np_data.shape[1] != 3:
            raise ValueError(f"Faces shape {faces_np_data.shape} unexpected. Expected (F,3).")
        # Register faces as a buffer: (num_faces, 3)
        self.register_buffer('th_faces', torch.Tensor(faces_np_data.astype(np.int32)).long())


    def forward(self,
                th_pose_axisang, # Tensor (batch_size x (num_joints * 3))
                th_betas=None,    # Tensor (batch_size x num_betas)
                th_trans=None):   # Tensor (batch_size x 3)
        """
        Forward pass of the SMPL model.

        Args:
            th_pose_axisang (torch.Tensor): Pose parameters in axis-angle representation.
                                            Shape: (batch_size, num_joints * 3).
                                            e.g., for SMPL (24 joints): (batch_size, 72).
                                            e.g., for SMPL-H (52 joints): (batch_size, 156).
            th_betas (torch.Tensor, optional): Shape parameters.
                                               Shape: (batch_size, num_betas).
                                               Defaults to the model's registered th_betas.
            th_trans (torch.Tensor, optional): Global translation.
                                               Shape: (batch_size, 3). Defaults to zeros.
        Returns:
            A tuple containing:
            - th_verts (torch.Tensor): Posed and shaped mesh vertices (batch_size, num_vertices, 3).
            - th_jtr (torch.Tensor): Posed joint locations (batch_size, num_joints, 3).
        """
        batch_size = th_pose_axisang.shape[0]

        # Validate input pose dimensions
        expected_pose_dims = self.num_joints * 3
        if th_pose_axisang.shape[1] != expected_pose_dims:
            raise ValueError(
                f"th_pose_axisang has {th_pose_axisang.shape[1]} dimensions ({th_pose_axisang.shape[1]//3} joints), "
                f"but model {self.model_type} with {self.num_joints} joints expects {expected_pose_dims} dimensions."
            )

        # Handle betas
        model_num_betas = self.th_betas.shape[1] 
        if th_betas is None:
            th_betas_used = self.th_betas.expand(batch_size, -1)
        else:
            if th_betas.shape[1] == model_num_betas:
                th_betas_used = th_betas
            elif th_betas.shape[1] == 1 and model_num_betas > 1: # Allow single beta to scale first component
                th_betas_used = torch.zeros(batch_size, model_num_betas,
                                            dtype=th_betas.dtype, device=th_betas.device)
                th_betas_used[:, 0] = th_betas[:, 0] 
            else:
                raise ValueError(
                    f"th_betas has {th_betas.shape[1]} shape parameters, but model expects {model_num_betas}. "
                    f"Please provide either a complete set of {model_num_betas} parameters or a single parameter "
                    f"(which will be applied to the first beta component)."
                )
        
        # Handle translation
        if th_trans is None:
            th_trans_used = torch.zeros(batch_size, 3, dtype=th_pose_axisang.dtype, device=th_pose_axisang.device)
        else:
            th_trans_used = th_trans

        # Convert axis-angle pose to rotation matrices
        th_pose_rotmat_full = th_posemap_axisang(th_pose_axisang) # (batch_size, num_joints * 9)

        root_rot = th_pose_rotmat_full[:, :9].view(batch_size, 3, 3)
        th_pose_rotmat_joints = th_pose_rotmat_full[:, 9:] # (batch_size, (num_joints-1)*9)

        # Create pose map for posedirs (flat (J-1)*9 rotation matrices - identity)
        if self.th_posedirs is not None and th_pose_rotmat_joints.shape[1] != self.th_posedirs.shape[1]:
             raise ValueError(
                 f"Dimension mismatch for posedirs. "
                 f"Pose matrices (rotmats for articulated joints) have {th_pose_rotmat_joints.shape[1]} elements ({th_pose_rotmat_joints.shape[1]//9} joints), "
                 f"posedirs expect {self.th_posedirs.shape[1]} elements ({self.th_posedirs.shape[1]//9} basis)."
            )
        th_pose_map = subtract_flat_id(th_pose_rotmat_joints) # (batch_size, (num_joints-1)*9)
        th_pose_map = torch.tensor(th_pose_map, dtype=torch.float32, device=th_pose_axisang.device) # Ensure type and device consistency

        # --- 1. Apply shape blend shapes ---
        # v_shaped = v_template + shapedirs * betas
        # self.th_shapedirs: (V*3, num_betas)
        # th_betas_used.transpose(1,0): (num_betas, B)
        shape_blend = torch.matmul(self.th_shapedirs, th_betas_used.transpose(1, 0))
        th_v_shaped_delta = shape_blend.transpose(1,0).view(batch_size, -1, 3) # (B, V, 3)
        
        th_v_shaped = self.th_v_template.expand(batch_size, -1, -1) + th_v_shaped_delta

        # Calculate joint locations from shaped mesh
        # self.th_J_regressor: (num_joints, V)
        # th_v_shaped: (B, V, 3)
        # th_j: (B, num_joints, 3)
        th_j = torch.matmul(self.th_J_regressor, th_v_shaped)

        # --- 2. Apply pose blend shapes ---
        # v_posed = v_shaped + posedirs * pose_map
        # self.th_posedirs: (V*3, (num_joints-1)*9)
        # th_pose_map.transpose(0,1): ((num_joints-1)*9, B)
        pose_blend = torch.matmul(self.th_posedirs, th_pose_map.transpose(0, 1))
        th_v_posed_delta = pose_blend.transpose(1,0).view(batch_size, -1, 3) # (B, V, 3)
        th_v_posed = th_v_shaped + th_v_posed_delta
        # th_v_posed is the final T-pose mesh with shape and pose blend shapes applied.

        # --- 3. Kinematic chain and skinning (LBS) ---
        th_results_global_kinematic = [] # Stores 4x4 world transformation for each joint

        root_j = th_j[:, 0, :].contiguous().view(batch_size, 3, 1)
        # Initial root transformation (rotation + T-pose root joint location)
        th_results_global_kinematic.append(th_with_zeros(torch.cat([root_rot, root_j], 2)))

        # Rotate each part relative to its parent
        for i in range(self.num_joints - 1):
            joint_idx_in_pose_rotmats = i 
            i_val = int(i + 1) # Actual joint index (1 to num_joints-1)

            joint_rot = th_pose_rotmat_joints[:, joint_idx_in_pose_rotmats * 9 : (joint_idx_in_pose_rotmats + 1) * 9].contiguous().view(batch_size, 3, 3)
            
            joint_j = th_j[:, i_val, :].contiguous().view(batch_size, 3, 1)
            
            parent_idx = self.kintree_parents[i_val]
            parent_j = th_j[:, parent_idx, :].contiguous().view(batch_size, 3, 1)
            
            # Relative transformation: R | (j - p_j)
            joint_rel_transform = th_with_zeros(
                torch.cat([joint_rot, joint_j - parent_j], 2)) # (B, 4, 4)
            
            # Chain multiplication: T_world_current = T_world_parent * T_relative_current
            th_results_global_kinematic.append(
                torch.matmul(th_results_global_kinematic[parent_idx], joint_rel_transform))
        
        # Stack list of (B,4,4) tensors to (B, num_joints, 4, 4)
        th_stacked_A = torch.stack(th_results_global_kinematic, dim=1)
        th_stacked_A = torch.tensor(th_stacked_A, dtype=torch.float32, device=th_pose_axisang.device) # Ensure type/device

        # Skinning: lbs
        # Calculate T_k = A_k - pack(A_k * j_k)
        # th_j (B, num_joints, 3) -> th_j_homo (B, num_joints, 4, 1)
        th_j_homo = torch.cat([th_j, torch.zeros(batch_size, self.num_joints, 1, dtype=th_j.dtype, device=th_j.device)], dim=2)
        th_j_homo = th_j_homo.unsqueeze(-1)
        th_j_homo = torch.tensor(th_j_homo, dtype=torch.float32, device=th_pose_axisang.device) # Ensure type/device

        # A_k * j_k
        adjusted_j_location = torch.matmul(th_stacked_A, th_j_homo) # (B, num_joints, 4, 1)

        # Create the equivalent of pack(A_k * j_k)
        packed_adjusted_j = torch.zeros_like(th_stacked_A)
        packed_adjusted_j[:, :, :, 3] = adjusted_j_location.squeeze(-1) # (B, J, 4)
        
        th_Gk_inv_part = th_stacked_A - packed_adjusted_j # (B, num_joints, 4, 4)

        # Apply skinning weights
        # self.th_weights: (V, num_joints) -> W: (num_joints, V)
        W = self.th_weights.transpose(0,1)
        # th_Gk_inv_part (B, J, 4, 4) -> (B, J, 16)
        th_Gk_inv_part_reshaped = th_Gk_inv_part.view(batch_size, self.num_joints, 16)
        
        # (B, 16, J) @ (J, V) -> (B, 16, V)
        th_T_Gesamt = torch.matmul(th_Gk_inv_part_reshaped.transpose(1,2), W)
        th_T_Gesamt = th_T_Gesamt.transpose(1,2).contiguous().view(batch_size, -1, 4, 4) # (B, V, 4, 4)

        # Transform T-pose vertices by per-vertex transformation matrices
        # th_v_posed (B, V, 3) -> th_v_posed_homo (B, V, 4, 1)
        th_v_posed_homo = torch.cat([th_v_posed, torch.ones(batch_size, th_v_posed.shape[1], 1, dtype=th_v_posed.dtype, device=th_v_posed.device)], dim=2)
        th_v_posed_homo = th_v_posed_homo.unsqueeze(-1)

        # (B, V, 4, 4) @ (B, V, 4, 1) -> (B, V, 4, 1)
        th_verts_homo = torch.matmul(th_T_Gesamt, th_v_posed_homo)
        th_verts = th_verts_homo[:, :, :3, 0] # (B, V, 3)

        # Final joint locations are the translation part of global kinematic transforms
        th_jtr = th_stacked_A[:, :, :3, 3] # (B, num_joints, 3)

        # Apply global translation and centering
        if bool(torch.norm(th_trans_used) == 0): # If th_trans was not provided
            if self.center_idx is not None and 0 <= self.center_idx < self.num_joints :
                center_joint = th_jtr[:, self.center_idx, :].unsqueeze(1) # (B, 1, 3)
                th_jtr = th_jtr - center_joint
                th_verts = th_verts - center_joint
        else: # If th_trans was provided by the user
            th_jtr = th_jtr + th_trans_used.unsqueeze(1)
            th_verts = th_verts + th_trans_used.unsqueeze(1)

        return th_verts, th_jtr
