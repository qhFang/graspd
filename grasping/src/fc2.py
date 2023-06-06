import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import pyvista as pv
from scipy.spatial.transform import Rotation
from functools import partial

import mdmm
print(mdmm.__file__)
from pyquaternion import Quaternion
from hydra.utils import to_absolute_path

from manotorch.manolayer import ManoLayer

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))))

#from utils import tip_idx as palm_idx 
from utils import palm_idx, tip_idx, finger_idx
import dflex as df
from dflex import sim
from dflex.model import Mesh
from dflex.tests import test_util

from pytorch3d.structures import Meshes
from pxr import Usd, UsdGeom, Gf, Sdf

def _usd_set_xform(xform, transform, scale, time):

    xform_ops = xform.GetOrderedXformOps()

    pos = tuple(transform[0])
    rot = tuple(transform[1])

    xform_ops[0].Set(Gf.Vec3d(pos), time)
    xform_ops[1].Set(Gf.Quatf(rot[3], rot[0], rot[1], rot[2]), time)
    xform_ops[2].Set(Gf.Vec3d(scale), time)




class FC2():
    def __init__(self, fc_config):
        self.config = fc_config
        self.inds = [0,1,2,3,4,5,6]
        self.n_inds = len(self.inds)

        self.device = fc_config.device
        self.sim_dt = 1e-5

        self.obj_name = fc_config.name

        self.n_vels = 3
        #self.task_vels = torch.vstack((torch.eye(3,device=self.device),-torch.eye(3,device=self.device),torch.zeros(1,3,device=self.device)))
        self.task_vels = torch.tensor([[0.0,0.0,0.0],[1.0,1.0,1.0],[-1.0,-1.0,-1.0]],device=self.device)
        #self.task_vels = torch.tensor([[0.0,0.0,0.0]],device=self.device)
        self.task_vels *= 0.01

        self.quat = None
        self.pos = None

        self.pos_ratio=1.0
        self.quat_ratio=1.0
        #self.pos_ratio=1.0
        #self.quat_ratio=1.0

        #self.pos_ratio=1e-2
        #self.quat_ratio=1e-1
        self.ncomps = 44
        #self.mano_layer = ManoLayer(
        #    mano_assets_root=to_absolute_path('/apdcephfs/private_qihangfang/Data/fcdata/assets_mano'),
        #    use_pca=False, ncomps=self.ncomps, flat_hand_mean=True).to(self.device)
        self.mano_layer = ManoLayer(
            mano_assets_root=to_absolute_path('/apdcephfs/private_qihangfang/Data/fcdata/assets_mano'),
            use_pca=True, ncomps=self.ncomps, flat_hand_mean=True).to(self.device)

        self.mano_shape = torch.zeros(1, 10, device=self.device)

        self.backoff = 0.2

        self.coarse_to_fine = fc_config.coarse_to_fine

        # VARIES 1.0 for shapenet, 2.0 for YCB
        if fc_config.dataset == "YCB":
            self.mano_ratio = 2.0
            self.initial_backoff = 0.2
        elif fc_config.dataset == "shapenet":
            self.mano_ratio = 1.0
            self.initial_backoff = 0.2
        elif fc_config.dataset == "IBS":
            self.mano_ratio = 2.0
            self.initial_backoff = 0.2

        self.nums = -1


    def build_model(self, obj_config, with_viewer_mesh=False):
        builder = df.ModelBuilder()

        # test_util.urdf_load(
        #     builder,
        #     #to_absolute_path("dflex/tests/assets/allegro_hand_description/allegro_hand_description_right.urdf"),
        #     to_absolute_path("grasping/data/mano/ManoHand.urdf"),
        #     #"/home/dylanturpin/repos/ros_thermal_grasp/urdf/allegro.urdf",
        #     df.transform((0.0, 0.0, 0.0), df.quat_from_axis_angle((0.0, 0.0, 1.0), math.pi*0.5)),
        #     floating=True,
        #     limit_ke=0.0,#1.e+3,
        #     limit_kd=0.0)#1.e+2)
        # rigid = test_util.build_rigid(builder,rpy=False)
        builder.add_articulation()
        rigid = -1
        rigid = builder.add_link(
            parent=rigid,
            X_pj=df.transform((0.0,0.0,0.0), df.quat_identity()),
            axis=(1.0, 0.0, 0.0),
            type=df.JOINT_FREE)

        batch_size=1
        # Generate random shape parameters
        # Generate random pose parameters, including 3 values for global axis-angle rotation
        random_pose = torch.zeros(batch_size, self.ncomps+3, device=self.device)
        #random_pose = torch.zeros(batch_size, 47, device=self.device)
        mano_output = self.mano_layer(random_pose, self.mano_shape)
        mano_vertices = mano_output.verts
        mano_vertices *= self.mano_ratio
        mano_vertices = mano_vertices[0,:,:].cpu().numpy()
        faces = self.mano_layer.th_faces
        faces = faces.cpu().numpy().flatten()
        #faces = test_util.pyvistaToTrimeshFaces(faces.cpu().numpy()).flatten()

        builder.add_shape_mesh(
            body=rigid,
            pos=(0.0, 0.0, 0.0),
            rot=(0.0, 0.0, 0.0, 1.0),
            mesh=Mesh(mano_vertices,faces),
            scale=(1.0,1.0,1.0),
            ke=100000.0,
            kd=1000.0,
            kf=1000.0,
            mu=0.5
        )

        if obj_config.primitive_type == "sphere":
            obj = test_util.build_rigid(builder)
            builder.add_shape_sphere(
                body=obj,
                pos=(0.0, 0.0, 0.0),
                rot=(0.0, 0.0, 0.0, 1.0),
                radius=obj_config.radius,
                ke=100000.0,
                kd=1000.0,
                kf=1000.0,
                mu=0.5)
        elif obj_config.primitive_type == "box":
            obj = test_util.build_rigid(builder)
            builder.add_shape_box(
                body=obj,
                pos=(0.0, 0.0, 0.0),
                rot=(0.0, 0.0, 0.0, 1.0),
                hx=obj_config.hx,
                hy=obj_config.hy,
                hz=obj_config.hz,
                ke=100000.0,
                kd=1000.0,
                kf=1000.0,
                mu=0.5
            )
        elif obj_config.primitive_type == "mesh_sdf":
            rescale = obj_config.rescale
            mesh_pv = pv.read(to_absolute_path(obj_config.mesh_path))

            faces = test_util.pyvistaToTrimeshFaces(np.array(mesh_pv.faces)).flatten()
            # VARIES 10.0 for YCB, 1.0 for ShapeNet
            if self.config.dataset == "YCB":
                mesh_norm_factor = 10.0
                #mesh_norm_factor = 1.0
            elif self.config.dataset == "shapenet":
                mesh_norm_factor = 1.0
            elif self.config.dataset == "IBS":
                mesh_norm_factor = 1.0
            vertices = mesh_pv.points

            com = mesh_pv.center_of_mass()*rescale*mesh_norm_factor
            self.com = torch.from_numpy(com).float().to(self.device)
            obj = test_util.build_rigid(builder, com=com)
            #obj = test_util.build_rigid(builder)

            sdf_data = np.load(to_absolute_path(obj_config.sdf_path), allow_pickle=True).item()
            sdf = sdf_data["sdf"]
            
            #pos = sdf_data["pos"]
            #scale = sdf_data["scale"]

            # multiply by 10 then pad by 0.1
            # upscale = 10.0
            if self.config.dataset == "YCB":
                pad = 0.1 # already scaled by mesh_norm_factor
                #pad = 0.06 # already scaled by mesh_norm_factor
            elif self.config.dataset == "shapenet":
                pad = 0.06 # already scaled by mesh_norm_factor
            elif self.config.dataset == "IBS":
                pad = 0.1 # already scaled by mesh_norm_factor
            min_bounds = vertices.min(axis=0) * mesh_norm_factor - pad
            max_bounds = vertices.max(axis=0) * mesh_norm_factor + pad
            scale = max_bounds - min_bounds
            pos = 0.5*(min_bounds + max_bounds)
            builder.add_shape_mesh(
                body=obj,
                #pos=(0.0, 0.0, 0.0),
                pos=-com,
                rot=(0.0, 0.0, 0.0, 1.0),
                mesh=Mesh(vertices,faces),
                scale=(mesh_norm_factor*rescale,mesh_norm_factor*rescale,mesh_norm_factor*rescale),
                ke=100000.0,
                kd=1000.0,
                kf=1000.0,
                mu=0.5
            )

            #import matplotlib.pyplot as plt
            #plt.imshow(sdf.reshape((256,256,256),order='F')[:,128,:]*rescale)

            builder.add_shape_sdf(
                body=obj,
                pos=(pos[0]*rescale, pos[1]*rescale, pos[2]*rescale)-com,
                #pos=(pos[0]*rescale, pos[1]*rescale, pos[2]*rescale) - com,
                rot=(0.0, 0.0, 0.0, 1.0),
                sdf=torch.tensor(sdf, dtype=torch.float32, device=self.device)*rescale,
                scale=(scale[0]*rescale, scale[1]*rescale, scale[2]*rescale),
                ke=100000.0,
                kd=1000.0,
                kf=1000.0,
                mu=0.5,
            )


        builder.joint_target = np.copy(builder.joint_q)
        builder.joint_target_ke = 0.0*np.array(builder.joint_target_ke)
        builder.joint_target_kd = 0.0*np.array(builder.joint_target_kd)
        builder.joint_limit_ke = 0.0*np.array(builder.joint_limit_ke)
        builder.joint_limit_kd = 0.0*np.array(builder.joint_limit_ke)

        self.model = builder.finalize(adapter=self.device)
        self.model.ground = False
        self.model.enable_tri_collisions = False
        self.model.gravity = torch.tensor((0.0, 0.0, 0.0), dtype=torch.float32, device=self.device)
        state = self.model.state()
        self.model.collide(state)

        self.integrator = df.sim.SemiImplicitIntegrator()
        self.box_contact_inds = torch.where(self.model.contact_body1 == 6)[0]
        self.self_contact_inds = torch.where(self.model.contact_body1 !=6)[0]
        self.builder = builder

        # for i in range(0,len(self.model.joint_type)):
        #     if (self.model.joint_type[i] == df.JOINT_REVOLUTE):
        #         dof = self.model.joint_q_start[i]
        #         if dof not in self.inds: continue
        #         mid = (self.model.joint_limit_lower[dof] + self.model.joint_limit_upper[dof])*0.5
        #         fully_open = self.model.joint_limit_lower[dof]
        #         fully_closed = self.model.joint_limit_upper[dof]

        #         # default joints to fully_open
        #         #self.model.joint_q[dof] = fully_open
        #         self.model.joint_q[dof] = fully_open

        #         # set rotational joints of fingers and thumb to mid
        #         if dof in [7,11,15, 19]:
        #             self.model.joint_q[dof] = mid

                # start thumb opposing fingers
                #if dof in [19]:
                    #self.model.joint_q[dof] = fully_closed

        if obj_config.primitive_type == "mesh_sdf":
            # keep restarting until we find a good point
            df.config.no_grad = True
            n_attempts = 1000
            for i in range(n_attempts):
                print(f"attempt {i} to find initialize")
                # pick a random point, get the normal, save the starting pos for the hand
                ind = np.random.choice(np.arange(0,mesh_pv.point_normals.shape[0]))
                pos = vertices[ind,:] * mesh_norm_factor * rescale
                n = mesh_pv.point_normals[ind,:]
                R = test_util.rotation_matrix_from_vectors(np.array([0.0,1.0,0.0]), n)
                if np.isnan(R).any():
                    continue
                R = Rotation.from_matrix(R)

                rand_roll = Quaternion._from_axis_angle(np.array([0.0,1.0,0.0]), np.random.rand()*2*np.pi)
                rand_roll = rand_roll.elements[[1,2,3,0]]
                rand_roll = Rotation.from_quat(rand_roll)
                self.quat = (R*rand_roll).as_quat()
                if np.linalg.norm(n) == 0.0:
                    continue
                n /= np.linalg.norm(n)
                self.pos = pos + self.initial_backoff * n

                #mano_input = torch.zeros(self.ncomps+3, device=self.device)
                mano_input = torch.zeros(3+self.ncomps, device=self.device)
                mano_output = self.mano_layer(mano_input[None,:], self.mano_shape)
                mano_vertices = mano_output.verts
                mano_vertices *= self.mano_ratio
                self.model.contact_point0[:778:] = mano_vertices[0,:,:].detach()

                state = self.model.state()
                state.joint_q[0:3] = torch.tensor(self.pos,dtype=torch.float,device=self.device)
                state.joint_q[3:7] = torch.tensor(self.quat,dtype=torch.float,device=self.device)

                m = 1
                for k in range(m):
                    state = self.integrator.forward(
                        self.model, state, self.sim_dt,
                        update_mass_matrix=True)

                dist = state.contact_world_dist[self.box_contact_inds].min()
                # No wrench
                #if True:
                if state.contact_f_s[self.box_contact_inds,:].abs().sum() == 0.0:# and dist <= 0.27 and dist >= 0.25:
                    self.backoff = dist.detach() - 0.01
                    break
            df.config.no_grad = False
            
        

    def sample_initial_guess(self):
        state = self.model.state()
        if self.pos is None:
            quat = Quaternion.random()
            scale = self.model.shape_geo_scale[-1][0]
            scale = torch.tensor([2*scale,2*scale,2*scale])
            obj_pos = self.model.shape_transform[-1,:3]
            #backoff = (scale/2).abs().max()
            #backoff = 0.4
            pos = -0.2 * torch.tensor(quat.rotate(torch.tensor([1.0,0.0,0.0])), device=self.device) + obj_pos
            state.joint_q[:3] = torch.tensor(pos,device=self.device,dtype=torch.float)
            state.joint_q[3:7] = torch.tensor(quat.elements[[1,2,3,0]],device=self.device,dtype=torch.float)

            m = 1
            for k in range(m):
                state = self.integrator.forward(
                    self.model, state, self.sim_dt,
                    update_mass_matrix=True)

            dist = state.contact_world_dist[self.box_contact_inds].min()
            self.backoff = dist.detach() - 0.01            
            state = self.model.state()
            state.joint_q[:3] = torch.tensor(pos,device=self.device,dtype=torch.float)
            state.joint_q[3:7] = torch.tensor(quat.elements[[1,2,3,0]],device=self.device,dtype=torch.float)
        else:
            pos = self.pos
            quat = self.quat
            state.joint_q[:3] = torch.tensor(pos,device=self.device,dtype=torch.float)
            state.joint_q[3:7] = torch.tensor(quat,device=self.device,dtype=torch.float)
        #quat = self.builder.joint_q[3:7]
        #pos = -0.2*quat.rotate(torch.tensor([1.0,0.0,0.0]))
        #pos = torch.tensor([0.0, -0.07, 0.0])
        #pos = self.builder.joint_q[0:3]
        #state.joint_q[:3] = torch.tensor(pos,device=self.device,dtype=torch.float)
        #state.joint_q[3:7] = torch.tensor(quat.elements[[1,2,3,0]],device=self.device,dtype=torch.float)
        #state.joint_q[3:7] = torch.tensor(quat,device=self.device,dtype=torch.float)


        hand_q = torch.zeros(7, device=self.device)
        #mano_q = torch.zeros((self.ncomps+3), device=self.device)
        mano_q = torch.zeros((self.ncomps+3), device=self.device)
        #hand_q = torch.randn(7+self.ncomps+3, device=self.device)
        hand_q[:7] = state.joint_q[:7]
        hand_q[0:3] /= self.pos_ratio
        hand_q[3:7] /= self.quat_ratio
        #hand_q[7:] = 0.1*torch.ones_like(hand_q[7:])
        hand_q = torch.tensor([0.1201, -0.2255,  0.0062, -0.9454, -0.2133, -0.2378,  0.0644]).float().to(self.device)
        print(hand_q)

        contact_forces = torch.zeros(len(palm_idx), 6, device=self.device)

        return (hand_q, mano_q, contact_forces)

    def run(self, initial_guess):
        self.nums += 1
        
        hand_q, mano_q, contact_forces = initial_guess

        initial_hand_q = torch.clone(hand_q.detach())

        render = True

        if render:
            from pxr import Usd
            from dflex.render import UsdRenderer
            stage_name = f"outputs/current.usd"
            stage = Usd.Stage.CreateNew(stage_name)
            renderer = UsdRenderer(self.model, stage)
            renderer.draw_points = False
            renderer.draw_springs = False
            renderer.draw_shapes = True

        ###### OPTIMIZATION
        #n = 50000
        n = 5000
        n_pose_only=0
        level_sets = torch.zeros(n, device=self.device)
        #if self.coarse_to_fine:
        if True:
            level_sets[n_pose_only:n-0] = torch.linspace(self.backoff.item(), 0.0, n-0-n_pose_only)
            level_sets[:n_pose_only] = level_sets[n_pose_only]
        alpha_schedule = torch.linspace(0.0001, 0.0, n)
        record_every = 20

        #inds_by_depth = [0,1,2, 3,4,5,6] + list(reversed([7,11,15,19, 8,12,16,20, 9,13,17,21, 10,14,18,22]))
        #inds_by_depth = [0,1,2, 6,5,4,3, 7,11,15,19, 8,12,16,20, 9,13,17,21, 10,14,18,22]
        #inds_by_depth = [0,1,2, 3,4,5,6, ]
        #inds_by_depth = [list(range(7+self.ncomps+3))]
        #inds_by_depth = [[0,1,2], [0,1,2], list(range(self.ncomps+3))[3:]]
        inds_by_depth = [[0,1,2], list(range(self.ncomps+3))]
        #inds_by_depth = [[0,1,2],[7,16,25,34,43],[8,17,26,35,44],[9,18,27,36,45],[10,19,28,37,46],[11,20,29,38,47],[12,21,30,39,48],[13,22,31,40,49],[14,23,32,41,50],[15,24,33,42,51],[52,53,54]]
        #inds_by_depth = [[0,1,2], [3,4,5,6,], [7,11,15,19], [8,12,16,20], [9,13,17,21], [10,14,18,22]]

        n_inner_contact_forces = 1
        n_inner_hand_q = len(inds_by_depth)
        n_inner_distance = 1
        n_inner_net_wrench = 5
        n_inner_pene = 1
        n_warmup = 200

        # stats for contact_forces update loop
        show_plots = False
        stats_l_physics_ = torch.zeros((n,n_inner_contact_forces, self.n_vels), device=self.device)
        stats_c_task = torch.zeros((n,n_inner_contact_forces), device=self.device)
        stats_contact_forces_grad_norm = torch.zeros((n,n_inner_contact_forces), device=self.device)
        stats_contact_forces_delta = torch.zeros((n,n_inner_contact_forces), device=self.device)
        stats_contact_forces = torch.zeros((n,n_inner_contact_forces,self.n_vels,6), device=self.device)
        stats_contact_forces_l1 = torch.zeros((n,n_inner_contact_forces,self.n_vels,6), device=self.device)
        # stats for hand_q update loop
        stats_l_physics = torch.zeros((n,n_inner_hand_q, self.n_vels), device=self.device)
        stats_l_self_collision = torch.zeros((n,n_inner_hand_q), device=self.device)
        stats_l_joint = torch.zeros((n,n_inner_hand_q), device=self.device)
        stats_c_joint = torch.zeros((n,n_inner_hand_q), device=self.device)
        stats_hand_q_grad_norm = torch.zeros((n,n_inner_hand_q), device=self.device)
        stats_hand_q_delta = torch.zeros((n,n_inner_hand_q), device=self.device)
        stats_hand_q = torch.zeros((n,n_inner_hand_q,hand_q.shape[0]), device=self.device)
        stats_hand_q_exp_avg = torch.zeros((n,n_inner_hand_q,hand_q.shape[0]), device=self.device)
        #stats_hand_q_exp_inf = torch.zeros((n,n_inner_hand_q,hand_q.shape[0]), device=self.device)
        stats_actual_forces = torch.zeros((n,n_inner_hand_q,self.n_vels,6), device=self.device)
        stats_actual_forces_l1 = torch.zeros((n,n_inner_hand_q,self.n_vels,6), device=self.device)

        mano_q.requires_grad_()
        hand_q.requires_grad_()
        contact_forces.requires_grad_()

        force_achieves_task_constraint = mdmm.MaxConstraintHard(partial(self.force_achieves_task,contact_forces=contact_forces), 0.00001, damping=1.0) # was 50.0, 10.0
        #joint_limits_constraint = mdmm.MaxConstraintHard(partial(self.joint_limits,hand_q=hand_q), 1e3, damping=1.0)
        mdmm_module = mdmm.MDMM([force_achieves_task_constraint])
        #mdmm_module_hand_q = mdmm.MDMM([joint_limits_constraint])
        #lr_contact_forces = 1e-3 # gave reasonable result
        lr_contact_forces = 1e-2
        contact_forces_optimizer = mdmm_module.make_optimizer([contact_forces],optimizer=torch.optim.Adamax,lr=lr_contact_forces)



        #lr_contact_forces = 1e-6
        #contact_forces_optimizer = torch.optim.Adam([{"params": [contact_forces], "lr": lr_contact_forces}])

        lr_hand_q = 1e-4 # 2e-4 worked well
        lr_mano_q = 3e-4 # 3e-3 worked well
        hand_q_optimizer = torch.optim.Adam([
            {"params": [hand_q], "lr": lr_hand_q},
            {"params": [mano_q], "lr": lr_mano_q}])

        load_opt_state = False
        if load_opt_state:
            checkpoint = torch.load("/home/dylanturpin/repos/dflex_clean_explore/grasping/data/checkpoint.pth")
            for param_group_state in checkpoint["hand_q_optimizer_state_dict"]["state"].values():
                param_group_state["exp_avg"].zero_()
            for param_group_state in checkpoint["contact_forces_optimizer_state_dict"]["state"].values():
                param_group_state["exp_avg"].zero_()
            hand_q_optimizer.load_state_dict(checkpoint["hand_q_optimizer_state_dict"])
            contact_forces_optimizer.load_state_dict(checkpoint["contact_forces_optimizer_state_dict"])
            mdmm_module.load_state_dict(checkpoint["mdmm_module_state_dict"])
            mdmm_module_hand_q.load_state_dict(checkpoint["mdmm_module_hand_q_state_dict"])

        history = {
            'joint_q': torch.zeros((n//record_every+1, self.model.joint_q.shape[0])),
            'mano_q': torch.zeros((n//record_every+1, mano_q.shape[0])),
            'joint_angles': torch.zeros((n//record_every+1, 21, 3)),
            'c_task': torch.zeros((n//record_every+1)),
            'l_rank': torch.zeros((n//record_every+1)),
            'l_distance': torch.zeros((n//record_every+1)),
            'l_netwrench': torch.zeros((n//record_every+1)),
            'l_self_collision': torch.zeros((n//record_every+1)),
            'l_joint': torch.zeros((n//record_every+1)),
            'contact_forces': torch.zeros(n//record_every+1, *contact_forces.shape),
            'actual_contact_forces': torch.zeros(n//record_every+1, *contact_forces.shape)
        }

        box_contact_forces = torch.zeros_like(contact_forces)
        for i in range(n):
            self.model.contact_point0 = self.model.contact_point0.detach()
            self.model.contact_kd = level_sets[i].item()
            self.model.contact_alpha = alpha_schedule[i].item()
            if i == 0 and render:
                contact_point0_before = torch.clone(self.model.contact_point0)
                mano_output = self.mano_layer(mano_q[None,:], self.mano_shape)
                vertices = mano_output.verts
                vertices *= self.mano_ratio
                
                self.model.contact_point0[:778,:]= vertices[0,:,:]
                state = self.model.state()
                state.joint_q[self.inds] = hand_q[:self.n_inds].detach()
                state.joint_q[self.inds[0:3]] = self.pos_ratio*hand_q[0:3]
                state.joint_q[self.inds[3:7]] = self.quat_ratio * hand_q[3:7]

                m = 1
                for k in range(m):
                    state = self.integrator.forward(
                        self.model, state, self.sim_dt,
                        update_mass_matrix=True)
                self.model.contact_point0 = contact_point0_before

                m=stage.GetObjectAtPath("/root/body_0/mesh_0")
                m.GetAttribute("points").Set(vertices[0,:,:].detach().cpu().numpy(),i)
                renderer.update(state, i)
                stage.Save()
                state = self.model.state()
                state.joint_q[self.inds] = hand_q[:7]
                state.joint_q[self.inds[0:3]] = self.pos_ratio*hand_q[0:3]
                state.joint_q[self.inds[3:7]] = self.quat_ratio * hand_q[3:7]
                history['joint_q'][0,:] = state.joint_q.detach().cpu()
                history['joint_angles'][0,:,:] = mano_output.joints[0,:,:].detach().cpu()
                history['mano_q'][0,:] = mano_q.detach().cpu()
                history['c_task'][0] = 0.0
                history['l_distance'][0] = 0.0
                history['l_netwrench'][0] = 0.0
                history['l_rank'][0] = 0.0
                history['contact_forces'][0,:,:] = contact_forces.detach().cpu()

            #self.model.contact_kd = 0.00001
            #self.model.contact_kd = level_sets[i]

            for j in range(1):
                contact_point0_before = torch.clone(self.model.contact_point0)
                mano_output = self.mano_layer(mano_q[None,:], self.mano_shape)
                vertices = mano_output.verts            
                vertices *= self.mano_ratio
                self.model.contact_point0[:778,:] = vertices[0,:,:]

                state = self.model.state()
                state.joint_q[self.inds] = hand_q[:self.n_inds]
                state.joint_q[self.inds[0:3]] = self.pos_ratio*hand_q[0:3]
                state.joint_q[self.inds[3:7]] = self.quat_ratio * hand_q[3:7]
                m = 1

                for k in range(m):
                    state = self.integrator.forward(
                        self.model, state, self.sim_dt,
                        update_mass_matrix=True)
            
                l_distance = torch.mean(state.contact_world_dist[self.box_contact_inds][palm_idx].abs())
                l_distance_tip = torch.mean(state.contact_world_dist[self.box_contact_inds][tip_idx].abs())
                l_distance = l_distance + l_distance_tip

                hand_mesh = Meshes(state.contact_world_pos[self.box_contact_inds].unsqueeze(0), self.mano_layer.th_faces.unsqueeze(0))
                hand_normals = hand_mesh.verts_normals_packed()
                l_normals = 1 + torch.sum(hand_normals[palm_idx] * state.contact_world_n[self.box_contact_inds][palm_idx], dim=-1)
                l_normals = torch.mean(l_normals * l_normals)
                
                hand_com = state.contact_world_pos[self.box_contact_inds].unsqueeze(0).mean(dim=0)
                l_com = (hand_com - self.com).norm()
                l_com = 0
                
                
                grasp_matrix = state.contact_matrix[self.box_contact_inds][palm_idx].permute(1, 2, 0).reshape(6, -1)
                temp = torch.tensor(0.001).float().to(self.device) * torch.tensor(np.eye(6)).float().to(self.device)
                temp = torch.matmul(grasp_matrix, grasp_matrix.transpose(0, 1)) - temp
                eigval = torch.linalg.eigh(temp.cpu())[0].to(self.device)
                rnev = F.relu(-eigval)
                l_rank = torch.sum(rnev * rnev)

                l_penetration = torch.sum(F.relu(-state.contact_world_dist[self.box_contact_inds]))
                
                l_all = 1000 * l_distance + l_normals + 1000 * l_rank + 0.1 * l_com + 1000 * l_penetration
                
                hand_q_old = hand_q.clone()
                mano_q_old = mano_q.clone()

                hand_q_optimizer.zero_grad()
                l_all.backward()
                hand_q_optimizer.step()

                self.model.contact_point0 = contact_point0_before
                
                
            box_contact_forces = state.contact_leaky_f_s[self.box_contact_inds][palm_idx].detach()


            l_contact_wrench = 0
            for j in range(0):
                #print(contact_forces)
                #contact_wrench = (F.sigmoid(contact_forces) + 1e-5) / (F.sigmoid(contact_forces) + 1e-5).norm()
                #contact_wrench = contact_forces * state.contact_f_s_normal[self.box_contact_inds][palm_idx].detach()

                if i<n_warmup:
                    #for param_group in contact_forces_optimizer.param_groups:
                    param_group = contact_forces_optimizer.param_groups[0]
                    param_group["lr"] = i * (lr_contact_forces / n_warmup)
                else:
                    #for param_group in contact_forces_optimizer.param_groups:
                    param_group = contact_forces_optimizer.param_groups[0]
                    param_group["lr"] = lr_contact_forces


                l_physics_ = 0.0
                l_ = ((contact_forces - box_contact_forces.detach())**2).sum()
                l_physics_ = 1e-2 * l_
                    
                mdmm_return_ = mdmm_module(l_physics_)
                contact_forces_optimizer.zero_grad()
                mdmm_return_.value.backward()
                contact_forces_optimizer.step()

                l_contact_wrench = mdmm_return_.fn_values[0]
                

                #contact_forces_optimizer.zero_grad()
                #l_contact_wrench.backward()
                #
                
            #if i == 2:
            #    exit()
            #print(l_physics_.item(), l_contact_wrench.item(), mdmm_return_.value.item())

            l_pene_regu = 0
            for j in range(0):
                contact_point0_before = torch.clone(self.model.contact_point0)
                mano_output = self.mano_layer(mano_q[None,:], self.mano_shape)
                vertices = mano_output.verts            
                vertices *= self.mano_ratio
                self.model.contact_point0[:778,:] = vertices[0,:,:]

                state = self.model.state()
                state.joint_q[self.inds] = hand_q[:self.n_inds]
                state.joint_q[self.inds[0:3]] = self.pos_ratio*hand_q[0:3]
                state.joint_q[self.inds[3:7]] = self.quat_ratio * hand_q[3:7]
                m = 1

                for k in range(m):
                    state = self.integrator.forward(
                        self.model, state, self.sim_dt,
                        update_mass_matrix=True)

                #l_pene_regu = ((state.contact_world_dist[self.box_contact_inds][palm_idx] + ((F.sigmoid(contact_forces) + 1e-5) / (F.sigmoid(contact_forces) + 1e-5).norm() / 1000).squeeze(1).detach()) ** 2).sum()
                l_pene_regu = ((state.contact_leaky_f_s[self.box_contact_inds][palm_idx] - contact_forces.detach()) ** 2).sum()

                hand_q_optimizer.zero_grad()
                l_pene_regu.backward()
                hand_q_optimizer.step()
            
         
            members = [attr for attr in dir(state) if not callable(getattr(state, attr)) and not attr.startswith("__") and type(getattr(state, attr)) == torch.Tensor]
            if (torch.isnan(hand_q).any()):
                print(state.body_v_s)
                print(state.joint_q)
                for attr in members:
                    if torch.isnan(getattr(state, attr)).any():
                        print(attr)
                exit()

            # plot!
            if i == 99000:
                print("break here")
            if show_plots:
                fig, axes = plt.subplots(10)
                axes[0].plot(np.arange((i+1)*n_inner_contact_forces), stats_l_physics_[:(i+1),:,0].flatten().cpu(), label="physics0", alpha=0.7)
                axes[0].plot(np.arange((i+1)*n_inner_contact_forces), stats_l_physics_[:(i+1),:,1].flatten().cpu(), label="physics1", alpha=0.7)
                #axes[0].plot(np.arange((i+1)*n_inner_contact_forces), stats_l_physics_[:(i+1),:,2].flatten().cpu(), label="physics2", alpha=0.7)
                axes[0].legend()
                axes[1].plot(np.arange((i+1)*n_inner_contact_forces), stats_c_task[:(i+1),:].flatten().cpu(), label="task")
                axes[1].legend()
                axes[2].plot(np.arange((i+1)*n_inner_hand_q), stats_l_physics[:(i+1),:,0].flatten().cpu(), label="physics0", alpha=0.7)
                axes[2].plot(np.arange((i+1)*n_inner_hand_q), stats_l_physics[:(i+1),:,1].flatten().cpu(), label="physics1", alpha=0.7)
                #axes[2].plot(np.arange((i+1)*n_inner_hand_q), stats_l_physics[:(i+1),:,2].flatten().cpu(), label="physics2", alpha=0.7)
                axes[2].legend()
                axes[3].plot(np.arange((i+1)*n_inner_hand_q), stats_l_self_collision[:(i+1),:].flatten().cpu(), label="l_self_collision")
                axes[3].legend()
                axes[4].plot(np.arange((i+1)*n_inner_hand_q), stats_l_joint[:(i+1),:].flatten().cpu(), label="l_joint")
                axes[4].legend()
                axes[5].plot(np.arange((i+1)*n_inner_hand_q), stats_c_joint[:(i+1),:].flatten().cpu(), label="c_joint")
                axes[5].legend()
                axes[6].plot(np.arange((i+1)*n_inner_contact_forces), stats_contact_forces_grad_norm[:(i+1),:].flatten().cpu(), label="contact_forces_grad_norm")
                axes[6].legend()
                axes[7].plot(np.arange((i+1)*n_inner_contact_forces), stats_contact_forces_delta[:(i+1),:].flatten().cpu(), label="norm_contact_forces_delta")
                axes[7].legend()
                axes[8].plot(np.arange((i+1)*n_inner_hand_q), stats_hand_q_grad_norm[:(i+1),:].flatten().cpu(), label="hand_q_grad_norm")
                axes[8].legend()
                axes[9].plot(np.arange((i+1)*n_inner_hand_q), stats_hand_q_delta[:(i+1),:].flatten().cpu(), label="norm_hand_q_delta")
                axes[9].legend()

                fig2, axes2 = plt.subplots(12,2)
                for l in range(hand_q.shape[0]):
                    axes2[l % 12, l // 12].plot(np.arange((i+1)*n_inner_hand_q), stats_hand_q[:(i+1),:,l].flatten().cpu(), label=str(l))
                    if l>6:
                        axes2[l % 12, l // 12].axhline(y=self.model.joint_limit_lower[l], color='r', linestyle='--', alpha=0.1)
                        axes2[l % 12, l // 12].axhline(y=self.model.joint_limit_upper[l], color='r', linestyle='--', alpha=0.1)
                    axes2[l % 12, l // 12].legend()

                fig3, axes3 = plt.subplots(12,2)
                for l in range(hand_q.shape[0]):
                    axes3[l % 12, l // 12].plot(np.arange((i+1)*n_inner_hand_q), stats_hand_q_exp_avg[:(i+1),:,l].flatten().cpu(), label=f"hand_q_exp_avg{l}")
                    axes3[l % 12, l // 12].legend()

                #fig4, axes4 = plt.subplots(12,2)
                #for l in range(hand_q.shape[0]):
                    #axes4[l % 12, l // 12].plot(np.arange((i+1)*n_inner_hand_q), stats_hand_q_exp_inf[:(i+1),:,l].flatten().cpu(), label=f"hand_q_exp_inf{l}")
                    #axes4[l % 12, l // 12].legend()

                fig5, axes5 = plt.subplots(4,6)
                for l in range(6):
                    for m in range(self.n_vels):
                        axes5[0, l].plot(np.arange((i+1)*n_inner_contact_forces), stats_contact_forces[:(i+1),:,m,l].flatten().cpu(), label=f"contact_forces sim{m} dim{l}", alpha=0.7)
                    axes5[0,l].legend()

                    for m in range(self.n_vels):
                        axes5[1, l].plot(np.arange((i+1)*n_inner_contact_forces), stats_contact_forces_l1[:(i+1),:,m,l].flatten().cpu(), label=f"contact_forces_l1 sim{m} dim{l}", alpha=0.7)
                    axes5[1,l].legend()

                    for m in range(self.n_vels):
                        #alpha= 1.0 if m == 0 else 0.1
                        alpha=0.7
                        axes5[2,l].plot(np.arange((i+1)*n_inner_hand_q), stats_actual_forces[:(i+1),:,m,l].flatten().cpu(), label=f"actual_forces sim{m} dim{l}", alpha=alpha)
                    axes5[2,l].legend()

                    for m in range(self.n_vels):
                        #alpha= 1.0 if m == 0 else 0.01
                        alpha=0.7
                        axes5[3,l].plot(np.arange((i+1)*n_inner_hand_q), stats_actual_forces_l1[:(i+1),:,m,l].flatten().cpu(), label=f"actual_forces_l1 sim{m} dim{l}", alpha=alpha)
                    axes5[3,l].legend()
                fig.tight_layout()
                fig2.tight_layout()
                fig3.tight_layout()
                fig4.tight_layout()
                fig5.tight_layout()
                fig.show()
                fig2.show()
                fig3.show()
                fig4.show()
                fig5.show()


            if i % record_every == 0:
                mano_output = self.mano_layer(mano_q[None,:], self.mano_shape)
                if render:
                    vertices = mano_output.verts
                    vertices *= self.mano_ratio

                    m=stage.GetObjectAtPath("/root/body_0/mesh_0")
                    #vertices = np.array(m.GetAttribute("points").Get(i))*0.9
                    #m.GetAttribute("points").Set(vertices,i+1.0)
                    m.GetAttribute("points").Set(vertices[0,:,:].detach().cpu().numpy(),i+1)
                    renderer.update(state, i+1)
                    stage.Save()

                print(f"{i}/{n} task:{l_all} distance:{l_distance} normal: {l_normals}  rank: {l_rank} {torch.linalg.matrix_rank(grasp_matrix)} l_com: {l_com}  penetration: {l_penetration} l_contact_wrench:{l_contact_wrench} l_pene_regu: {l_pene_regu}")
                finger_dis = [state.contact_world_dist[self.box_contact_inds][finger_idx[i]].abs().min() for i in range(len(finger_idx))]
                finger_dis = torch.tensor(finger_dis)
                print(finger_dis)
                state = self.model.state()
                state.joint_q[self.inds] = hand_q[:7]
                state.joint_q[self.inds[0:3]] = self.pos_ratio*hand_q[0:3]
                state.joint_q[self.inds[3:7]] = self.quat_ratio * hand_q[3:7]
                history['joint_q'][i // record_every+1,:] = state.joint_q.detach().cpu()
                history['mano_q'][i // record_every+1,:] = mano_q.detach().cpu()
                history['joint_angles'][i // record_every+1,:,:] = mano_output.joints[0,:,:].detach().cpu()
                #history['c_task'][i // record_every+1] = mdmm_return_.fn_values[0].detach().cpu()
                history['l_distance'][i // record_every+1] = l_distance.detach().cpu()
                #history['l_netwrench'][i // record_every+1] = l_netwrench.detach().cpu()
                history['l_rank'][i // record_every+1] = l_rank.detach().cpu()
                #history['l_self_collision'][i // record_every+1] = l_self_collision.detach().cpu()
                #history['l_joint'][i // record_every + 1] = l_joint.detach().cpu()
                history['contact_forces'][i // record_every + 1,:,:] = contact_forces.detach().cpu()
                #history['actual_contact_forces'][i // record_every +1,:,:,:] = box_contact_forces.detach().cpu()
                #history['l_joint'][i // record_every] = mdmm_return.fn_values[1].detach().cpu()

        record_opt_state = False
        if record_opt_state:
            print("recording")
            path="/home/dylanturpin/repos/dflex_clean_explore/grasping/data/checkpoint.pth"
            torch.save({
                "hand_q_optimizer_state_dict": hand_q_optimizer.state_dict(),
                "contact_forces_optimizer_state_dict": contact_forces_optimizer.state_dict(),
                "mdmm_module_state_dict": mdmm_module.state_dict(),
                "mdmm_module_hand_q_state_dict": mdmm_module_hand_q.state_dict()
            },
            "path.pth")




        # record final results
        #print(f"{mdmm_return.fn_values[0]} {l_self_collision} {l_joint} {l_physics}")
        #print(f"{mdmm_return.fn_values[0]} {l_self_collision} {mdmm_return.fn_values[1]} {l_physics}")
        mano_output = self.mano_layer(mano_q[None,:], self.mano_shape)
        contact_num = (state.contact_world_dist[self.box_contact_inds][palm_idx] < 0).sum()
        max_penetration = state.contact_world_dist[self.box_contact_inds][palm_idx].min()
        tip_dis =  state.contact_world_dist[self.box_contact_inds][tip_idx].clone()
        state = self.model.state()
        state.joint_q[self.inds] = hand_q
        state.joint_q[self.inds[0:3]] = self.pos_ratio*hand_q[0:3]
        state.joint_q[self.inds[3:7]] = self.quat_ratio * hand_q[3:7]
        results = {
            'final_joint_q': state.joint_q.detach().cpu(),
            'final_mano_q': mano_q.detach().cpu(),
            'final_joint_angles': mano_output.joints[0,:,:].detach().cpu(),
            'final_l_rank': l_rank.detach().cpu(),
            'final_l_distance': l_distance.detach().cpu(),
            #'final_l_netwrench': l_netwrench.detach().cpu(),
            #'final_l_self_collision': l_self_collision.detach().cpu(),
            #'final_l_joint': l_joint.detach().cpu(),
            'final_contact_forces': contact_forces.detach().cpu(),
            'actual_contact_forces': box_contact_forces.detach().cpu(),
            #'final_l_joint': mdmm_return.fn_values[1].detach().cpu(),
            'history': history
        }

        final_joint_q = hand_q.detach().cpu().numpy()
        final_mano_q = mano_q.detach().cpu().numpy()
        succ_dir = self.geometry_normal_test_num(hand_q, mano_q)
        '''
        f = open(f'/apdcephfs/private_qihangfang/{self.obj_name}', 'a')
        f.write(f'{self.nums} ')
        for i in range(7):
            f.write(f'{final_joint_q[i]} ')
        for i in range(final_mano_q.shape[0]):
            f.write(f'{final_mano_q[i]} ')

        for i in range(len(succ_dir)):
            f.write(f'|| {succ_dir[i]} ')

        for i in range(5):
            f.write(f'a {tip_dis[i]}')

        f.write(f'{l_netwrench.item()} {l_distance.item()} {torch.linalg.matrix_rank(grasp_matrix).item()} {contact_num} {max_penetration} \n')
        f.close()
        '''
        return results

    @torch.no_grad()
    def geometry_test(self, model_state, hand_q, mano_q, translations=None):
        return self.geometry_normal_test(hand_q, mano_q, translations)
        if translations == None:
            translations = [[0.001,0,0], [-0.001,0,0], [0,0.001,0], [0,-0.001,0], [0,0,0.001], [0,0,-0.001]]
            succ_dir = [False for i in range(len(translations))]
        dist_old = model_state.contact_world_dist[self.box_contact_inds][palm_idx].clone()
        for i in range(len(translations)):
            contact_point0_before = torch.clone(self.model.contact_point0)
            mano_output = self.mano_layer(mano_q.reshape(-1)[None,:], self.mano_shape)
            vertices = mano_output.verts.clone()
            vertices *= self.mano_ratio
            vertices[0, :, 0] = vertices[0, :, 0] + translations[i][0]
            vertices[0, :, 1] = vertices[0, :, 1] + translations[i][1]
            vertices[0, :, 2] = vertices[0, :, 2] + translations[i][2]
            
            self.model.contact_point0[self.box_contact_inds,:] = vertices[0,:,:]
            state = self.model.state()
            state.joint_q[self.inds] = hand_q[:self.n_inds]
            state.joint_q[self.inds[0:3]] = self.pos_ratio*hand_q[0:3]
            state.joint_q[self.inds[3:7]] = self.quat_ratio * hand_q[3:7]

            m = 1
            for k in range(m):
                model_state = self.integrator.forward(
                    self.model, state, self.sim_dt,
                    update_mass_matrix=True)
                
            self.model.contact_point0 = contact_point0_before
            dist_new = model_state.contact_world_dist[self.box_contact_inds][palm_idx].clone()
            
            # more far
            if (dist_old <= dist_new).all():
                succ_dir[i] = False
            # no penetration
            elif (dist_new[dist_old > dist_new] > 0).all():
                succ_dir[i] = False
            else:
                succ_dir[i] = True
        
        return succ_dir

    @torch.no_grad()
    def geometry_normal_test(self, hand_q, mano_q, translations=None, sdf_check=True):
        if translations == None:
            translations = torch.tensor([[[1,0,0]], [[-1,0,0]], [[0,1,0]], [[0,-1,0]], [[0,0,1]], [[0,0,-1]]]).to(hand_q.device)
            succ_dir = [False for i in range(len(translations))]
            
        contact_point0_before = torch.clone(self.model.contact_point0)
        mano_output = self.mano_layer(mano_q[None,:], self.mano_shape)
        vertices = mano_output.verts            
        vertices *= self.mano_ratio
        self.model.contact_point0[self.box_contact_inds,:] = vertices[0,:,:]
        state = self.model.state()
        state.joint_q[self.inds] = hand_q[:self.n_inds]
        state.joint_q[self.inds[0:3]] = self.pos_ratio*hand_q[0:3]
        state.joint_q[self.inds[3:7]] = self.quat_ratio * hand_q[3:7].detach()
        m = 1

        for k in range(m):
            state = self.integrator.forward(
                self.model, state, self.sim_dt,
                update_mass_matrix=True)

        hand_mesh = Meshes(state.contact_world_pos[self.box_contact_inds].unsqueeze(0), self.mano_layer.th_faces.unsqueeze(0))
        hand_normals = hand_mesh.verts_normals_packed()[palm_idx]
        sdf_normals = state.contact_world_n[self.box_contact_inds][palm_idx]
        
        if (state.contact_world_dist[self.box_contact_inds][palm_idx] < 0.001).sum() < 3:
            return [False for i in range(len(translations))]
        hand_normals = hand_normals[state.contact_world_dist[self.box_contact_inds][palm_idx] < 0.001]
        sdf_normals = sdf_normals[state.contact_world_dist[self.box_contact_inds][palm_idx] < 0.001]
        
        if sdf_check:
            cos_normals = (hand_normals * sdf_normals).sum(-1)
            angle_normals = np.pi - torch.arccos(cos_normals)
            if (angle_normals <= (np.pi / 4)).sum() < 3:
                return [False for i in range(len(translations))]
            sdf_normals = sdf_normals[angle_normals <= (np.pi / 4)]
        
        for i in range(len(translations)):
            cos_normals = (hand_normals * translations[i]).sum(-1)
            angle_normals = np.pi - torch.arccos(cos_normals)
            if (angle_normals <= (np.pi / 2)).any():
                succ_dir[i] = True
            
        self.model.contact_point0 = contact_point0_before 
                
        return succ_dir
    

    @torch.no_grad()
    def geometry_normal_test_num(self, hand_q, mano_q, translations=None, sdf_check=True):
        if translations == None:
            translations = torch.tensor([[[1,0,0]], [[-1,0,0]], [[0,1,0]], [[0,-1,0]], [[0,0,1]], [[0,0,-1]]]).to(hand_q.device)
            succ_dir = [False for i in range(len(translations))]
            
        contact_point0_before = torch.clone(self.model.contact_point0)
        mano_output = self.mano_layer(mano_q[None,:], self.mano_shape)
        vertices = mano_output.verts            
        vertices *= self.mano_ratio
        self.model.contact_point0[self.box_contact_inds,:] = vertices[0,:,:]
        state = self.model.state()
        state.joint_q[self.inds] = hand_q[:self.n_inds]
        state.joint_q[self.inds[0:3]] = self.pos_ratio*hand_q[0:3]
        state.joint_q[self.inds[3:7]] = self.quat_ratio * hand_q[3:7].detach()
        m = 1

        for k in range(m):
            state = self.integrator.forward(
                self.model, state, self.sim_dt,
                update_mass_matrix=True)

        hand_mesh = Meshes(state.contact_world_pos[self.box_contact_inds].unsqueeze(0), self.mano_layer.th_faces.unsqueeze(0))
        hand_normals = hand_mesh.verts_normals_packed()[palm_idx]
        sdf_normals = state.contact_world_n[self.box_contact_inds][palm_idx]
        
        if (state.contact_world_dist[self.box_contact_inds][palm_idx] < 0.001).sum() < 3:
            return [False for i in range(len(translations))]
        hand_normals = hand_normals[state.contact_world_dist[self.box_contact_inds][palm_idx] < 0.001]
        sdf_normals = sdf_normals[state.contact_world_dist[self.box_contact_inds][palm_idx] < 0.001]
        
        if sdf_check:
            cos_normals = (hand_normals * sdf_normals).sum(-1)
            angle_normals = np.pi - torch.arccos(cos_normals)
            if (angle_normals <= (np.pi / 4)).sum() < 3:
                return [False for i in range(len(translations))]
            sdf_normals = sdf_normals[angle_normals <= (np.pi / 4)]
        
        for i in range(len(translations)):
            cos_normals = (sdf_normals * translations[i]).sum(-1)
            angle_normals = np.pi - torch.arccos(cos_normals)
            angle_normals = angle_normals / np.pi * 180
            #if (angle_normals <= (np.pi / 2)).all():
            succ_dir[i] = angle_normals.min().item()
            
        self.model.contact_point0 = contact_point0_before 

        #hand_vertices = state.contact_world_pos[self.box_contact_inds].cpu().numpy()
        #np.save(f'/apdcephfs/private_qihangfang/{self.nums}_vertices.npy', hand_vertices)
        #sdf_normals = state.contact_world_n[self.box_contact_inds].cpu().numpy()
        #np.save(f'/apdcephfs/private_qihangfang/{self.nums}_sdf_normals.npy', sdf_normals)
        #distance = state.contact_world_dist[self.box_contact_inds].cpu().numpy()
        #np.save(f'/apdcephfs/private_qihangfang/{self.nums}_distance.npy', distance)
        #hand_normals = hand_mesh.verts_normals_packed().cpu().numpy()
        #np.save(f'/apdcephfs/private_qihangfang/{self.nums}_hand_normals.npy', hand_normals)



        return succ_dir
    
    

    def force_achieves_task(self, contact_forces):

        contact_wrench = contact_forces.mean(dim=0)
        contact_wrench = contact_wrench.norm()
        l = contact_wrench


        return l

    def joint_limits(self, hand_q):
        l_joint = -torch.minimum(hand_q[7:] - self.model.joint_limit_lower[self.inds][7:], torch.zeros_like(hand_q[7:])).sum() - torch.minimum(self.model.joint_limit_upper[self.inds][7:] - hand_q[7:], torch.zeros_like(hand_q[7:])).sum()
        #l_joint += (1.0 - torch.norm(hand_q[3:7])).abs()
        return 1e7*l_joint
