import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import pyvista as pv
from scipy.spatial.transform import Rotation
from functools import partial

import mdmm
from pyquaternion import Quaternion
from hydra.utils import to_absolute_path

#from handtorch.handlayer import handLayer

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))))
sys.path.append((os.path.dirname(os.path.realpath(__file__))))
sys.path.append('/apdcephfs/private_qihangfang/Codes/Form-closure/env')
from hand import Hand
#from utils import tip_idx as palm_idx 

import dflex as df
from dflex import sim
from dflex.model import Mesh
from dflex.tests import test_util

from pytorch3d.structures import Meshes

palm_idx = np.array(range(45))

class Shadow():
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
        self.action_space = 24
        self.hand = Hand("/apdcephfs/private_qihangfang/Codes/IBS-Grasping/src/ibs_env/scripts/hand/ShadowHand/", 0.001, use_joint_limit=False, use_quat=True, use_eigen=False).to(self.device)
        self.hand.sample(13000, re_sample=True, fully_sample=True)
        #self.backoff = 0.2

        self.coarse_to_fine = fc_config.coarse_to_fine

        # VARIES 1.0 for shapenet, 2.0 for YCB
        if fc_config.dataset == "YCB":
            self.mesh_norm_factor = 10.0
            self.hand_ratio = 2.0
            self.initial_backoff = 0.3
        elif fc_config.dataset == "shapenet":
            self.mesh_norm_factor = 1.0
            self.hand_ratio = 1.0
            self.initial_backoff = 0.2
        elif fc_config.dataset == "IBS":
            self.mesh_norm_factor = 1.0 
            self.hand_ratio = 2.0
            self.initial_backoff = 0.3

        self.nums = -1

        torch.manual_seed(3047)
        torch.cuda.manual_seed(3047)
        np.random.seed(3047)

        self.model = None

    def build_model(self, obj_config, with_viewer_mesh=False):
        builder = df.ModelBuilder()
        if self.model != None:
            # keep restarting until we find a good point
            df.config.no_grad = True
            n_attempts = 1000
            for i in range(n_attempts):
                print(f"attempt {i} to find initialize")
                # pick a random point, get the normal, save the starting pos for the hand
                ind = np.random.choice(np.arange(0,self.mesh_pv.point_normals.shape[0]))
                vertices = self.mesh_pv.points
                pos = vertices[ind,:] * self.mesh_norm_factor * self.rescale
                n = self.mesh_pv.point_normals[ind,:]
                R = test_util.rotation_matrix_from_vectors(np.array([1.0,0.0,0.0]), n)
                if np.isnan(R).any():
                    continue
                R = Rotation.from_matrix(R)

                rand_roll = Quaternion._from_axis_angle(np.array([1.0,0.0,0.0]), np.random.rand()*2*np.pi)
                rand_roll = rand_roll.elements[[1,2,3,0]]
                rand_roll = Rotation.from_quat(rand_roll)
                self.quat = (R*rand_roll).as_quat()
                if np.linalg.norm(n) == 0.0:
                    continue
                n /= np.linalg.norm(n)
                self.pos = pos + self.initial_backoff * n

                local_q = torch.zeros(25, device=self.device)
                local_q[:3] = torch.tensor(self.pos,device=self.device,dtype=torch.float)
                local_q[3:7] = torch.tensor(self.quat,device=self.device,dtype=torch.float)[[3,0,1,2]]
                hand_vertices = self.hand.forward_sampled(local_q[None,:])
                rep, ren, _ = self.hand(local_q[None,:])
                hand_vertices = torch.cat([rep, hand_vertices], dim=-1).transpose(1, 2)
                hand_vertices *= self.hand_ratio
                #print(hand_vertices.isnan())
                self.model.contact_point0[self.box_contact_inds,:] = hand_vertices[0,:,:].detach()

                state = self.model.state()
                #state.joint_q[0:3] = torch.tensor(self.pos,dtype=torch.float,device=self.device)
                #state.joint_q[3:7] = torch.tensor(self.quat,dtype=torch.float,device=self.device)

                m = 1
                for k in range(m):
                    state = self.integrator.forward(
                        self.model, state, self.sim_dt,
                        update_mass_matrix=True)

                dist = state.contact_world_dist[self.box_contact_inds].min()
                # No wrench
                #if True:
                if state.contact_f_s[self.box_contact_inds,:].abs().sum() == 0.0:# and dist <= 0.27 and dist >= 0.25:
                    #self.backoff = dist.detach() - 0.01
                    break
            df.config.no_grad = False
            return

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
        #action = torch.zeros(batch_size, 25).float().cuda()
        #retp,retn,_ = self.hand(action)
        #retp = retp.transpose(1, 2)
        #retn = retn.transpose(1, 2)
        dofs=np.zeros(self.hand.nr_dof())
        edofs = np.zeros(self.hand.extrinsic_size)
        edofs[3] = 1.0
        edofs = torch.from_numpy(edofs).float().cuda()
        dofs = torch.from_numpy(dofs).float().cuda()
        self.hand.forward_kinematics(edofs,dofs)
        pose = torch.zeros((1, 25), device=self.device).float()
        pose[0, 3] = 1.0
        rep, ren, _ = self.hand(pose)
        rep = rep.transpose(1, 2)[0] * self.hand_ratio
        mesh = self.hand.draw()
        hand_vertices = mesh.vertices
        hand_vertices *= self.hand_ratio
        hand_vertices = np.concatenate([rep.cpu().numpy(), hand_vertices], axis=0)
        #retp = retp[0,:,:]
        faces = mesh.faces
        faces = faces.copy().flatten()
        #faces = test_util.pyvistaToTrimeshFaces(faces.cpu().numpy()).flatten()

        builder.add_shape_mesh(
            body=rigid,
            pos=(0.0, 0.0, 0.0),
            rot=(0.0, 0.0, 0.0, 1.0),
            mesh=Mesh(hand_vertices,faces),
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
            self.rescale = rescale
            mesh_pv = pv.read(to_absolute_path(obj_config.mesh_path))
            self.mesh_pv = mesh_pv
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
            print(com)
            #obj = test_util.build_rigid(builder, com=mesh_pv.center_of_mass()*rescale*mesh_norm_factor)
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
            if self.config.dataset == "IBS":
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
                R = test_util.rotation_matrix_from_vectors(np.array([1.0,0,0.0]), n)
                if np.isnan(R).any():
                    continue
                R = Rotation.from_matrix(R)

                rand_roll = Quaternion._from_axis_angle(np.array([1.0,0.0,0.0]), np.random.rand()*2*np.pi)
                rand_roll = rand_roll.elements[[1,2,3,0]]
                rand_roll = Rotation.from_quat(rand_roll)                
                self.quat = (R*rand_roll).as_quat()
                if np.linalg.norm(n) == 0.0:
                    continue
                n /= np.linalg.norm(n)
                self.pos = pos + self.initial_backoff * n
                local_q = torch.zeros(25, device=self.device)
                local_q[:3] = torch.tensor(self.pos,device=self.device,dtype=torch.float)
                local_q[3:7] = torch.tensor(self.quat,device=self.device,dtype=torch.float)[[3,0,1,2]]
                hand_vertices = self.hand.forward_sampled(local_q[None,:])
                rep, ren, _ = self.hand(local_q[None,:])
                hand_vertices = torch.cat([rep, hand_vertices], dim=-1).transpose(1, 2)
                hand_vertices *= self.hand_ratio
                #print(hand_vertices.isnan())
                self.model.contact_point0[self.box_contact_inds,:] = hand_vertices[0,:,:].detach()

                state = self.model.state()
                #state.joint_q[0:3] = torch.tensor(self.pos,dtype=torch.float,device=self.device)
                #state.joint_q[3:7] = torch.tensor(self.quat,dtype=torch.float,device=self.device)

                m = 1
                for k in range(m):
                    state = self.integrator.forward(
                        self.model, state, self.sim_dt,
                        update_mass_matrix=True)

                dist = state.contact_world_dist[self.box_contact_inds].min()
                #print(state.contact_f_s[self.box_contact_inds,:].abs().sum())
                # No wrench
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
            #state.joint_q[:3] = torch.tensor(pos,device=self.device,dtype=torch.float)
            #state.joint_q[3:7] = torch.tensor(quat,device=self.device,dtype=torch.float)

        global_q = torch.zeros(7, device=self.device)
        local_q = torch.zeros(25, device=self.device)
        local_q[:7] = state.joint_q[:7]
        local_q[0:3] = torch.tensor(pos,device=self.device,dtype=torch.float) / self.pos_ratio
        local_q[3:7] = torch.tensor(quat,device=self.device,dtype=torch.float)[[3,0,1,2]] / self.quat_ratio


        return (global_q, local_q)

    def run(self, initial_guess):
        self.nums += 1
        
        global_q, local_q = initial_guess


        render = False

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
        n = 10000
        n_pose_only=0
        level_sets = torch.zeros(n, device=self.device)
        #if self.coarse_to_fine:
        #    level_sets[n_pose_only:n-0] = torch.linspace(self.backoff.item(), 0.0, n-0-n_pose_only)
        #    level_sets[:n_pose_only] = level_sets[n_pose_only]
        alpha_schedule = torch.linspace(0.0001, 0.0, n)
        record_every = 20

        n_inner_contact_forces = 1
        n_warmup = 200

        #global_q.requires_grad_()
        local_q.requires_grad_()

        lr_global_q = 1e-3 # 3e-3 worked well
        lr_local_q = 3e-4 # 3e-3 worked well
        global_q_optimizer = torch.optim.Adam([
            {"params": [local_q], "lr": lr_local_q}])

        history = {
            'joint_q': torch.zeros((n//record_every+1, self.model.joint_q.shape[0])),
            'local_q': torch.zeros((n//record_every+1, 25)),
            'l_rank': torch.zeros((n//record_every+1)),
            'l_distance': torch.zeros((n//record_every+1)),
            'l_netwrench': torch.zeros((n//record_every+1)),
            'l_self_collision': torch.zeros((n//record_every+1)),
            'l_joint': torch.zeros((n//record_every+1)),
        }




        for i in range(n):
            self.model.contact_point0 = self.model.contact_point0.detach()
            self.model.contact_kd = level_sets[i].item()
            self.model.contact_alpha = alpha_schedule[i].item()
            if i == 0 and render:
                contact_point0_before = torch.clone(self.model.contact_point0)
                hand_vertices = self.hand.forward_sampled(local_q[None,:])
                rep, ren, _ = self.hand(local_q[None,:])
                vertices = torch.cat([rep, hand_vertices], dim=-1).transpose(1, 2)
                vertices *= self.hand_ratio                
                self.model.contact_point0[self.box_contact_inds,:] = vertices[0,:,:]
                state = self.model.state()
                #state.joint_q[self.inds] = global_q[:self.n_inds]
                #state.joint_q[self.inds[0:3]] = self.pos_ratio*global_q[0:3]
                #state.joint_q[self.inds[3:7]] = self.quat_ratio * global_q[3:7]
    
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
                #state.joint_q[self.inds] = global_q[:7]
                #state.joint_q[self.inds[0:3]] = self.pos_ratio*global_q[0:3]
                #state.joint_q[self.inds[3:7]] = self.quat_ratio * global_q[3:7]
                history['joint_q'][0,:] = state.joint_q.detach().cpu()
                history['local_q'][0,:] = local_q.detach().cpu()
                history['l_distance'][0] = 0.0
                history['l_netwrench'][0] = 0.0
                history['l_rank'][0] = 0.0


            contact_point0_before = torch.clone(self.model.contact_point0)
            #import time
            #start = time.time()

            hand_vertices = self.hand.forward_sampled(local_q[None,:])
            rep, ren, _ = self.hand(local_q[None,:])
            vertices = torch.cat([rep, hand_vertices], dim=-1).transpose(1, 2)
            vertices *= self.hand_ratio
            #print(vertices)
            self.model.contact_point0[self.box_contact_inds,:] = vertices[0,:,:]

            #print(time.time() - start)

            state = self.model.state()
            #state.joint_q[self.inds] = global_q[:self.n_inds]
            #state.joint_q[self.inds[0:3]] = self.pos_ratio * global_q[0:3]
            #state.joint_q[self.inds[3:7]] = self.quat_ratio * global_q[3:7]
            m = 1


            
            for k in range(m):
                state = self.integrator.forward(
                    self.model, state, self.sim_dt,
                    update_mass_matrix=True)

            
            l_distance = torch.mean(state.contact_world_dist[self.box_contact_inds][palm_idx].abs())
            
            hand_normals = rep.transpose(1, 2)[0]
            l_normals = 1 + torch.sum(hand_normals * state.contact_world_n[self.box_contact_inds][palm_idx], dim=-1)
            l_normals = torch.mean(l_normals * l_normals)

            grasp_matrix = state.contact_matrix[self.box_contact_inds][palm_idx].permute(1, 2, 0).reshape(6, -1)
            temp = torch.tensor(0.001).float().to(self.device) * torch.tensor(np.eye(6)).float().to(self.device)
            temp = torch.matmul(grasp_matrix, grasp_matrix.transpose(0, 1)) - temp
            eigval = torch.linalg.eigh(temp.cpu())[0].to(self.device)
            rnev = F.relu(-eigval)
            l_rank = torch.sum(rnev * rnev)

            net_wrench = torch.matmul(state.contact_matrix[self.box_contact_inds][palm_idx], state.contact_world_n[self.box_contact_inds][palm_idx].unsqueeze(-1))            # x, 6, 3
            net_wrench[:, 3:] = net_wrench[:, 3:] * 0.001
            net_wrench = net_wrench.squeeze(-1) * state.contact_world_dist[self.box_contact_inds][palm_idx].unsqueeze(dim=1)
            net_wrench = net_wrench.mean(dim = 0)
            l_netwrench = (net_wrench * net_wrench).sum()
            
            l_self_collision = (state.contact_f_s[self.self_contact_inds,:]**2).sum()
            middle_inds = self.inds[7:]
            l_joint = torch.norm(local_q[middle_inds],dim=0)
            
            l_penetration = torch.sum(F.relu(-state.contact_world_dist[self.box_contact_inds]))
            
            
            
            #l_all = 100 * l_distance + l_normals + 100 * l_penetration + 1000 * l_rank + 5 * l_netwrench + 0.01 * global_q.norm()
            l_all = 100 * l_distance + l_normals + 100 * l_penetration + 1000 * l_rank + 100 * l_netwrench
            #print(l_distance)
            #grad = torch.autograd.grad(l_all, global_q, torch.ones_like(l_all), retain_graph=True)[0]
            #print(grad)
            #l_all = 100 * l_distance + l_normals + 100 * l_penetration + 1000 * l_rank + 0.01 * global_q.norm()

            #l_all = l_netwrench
            #l_all = 100 * l_distance + l_normals + 100 * l_penetration
            #l_all = 100 * l_distance + 100 * l_normals + l_penetration + 1000 * l_rank + l_netwrench + 0.01 * global_q.norm()
            #l_all = 100 * l_distance + 100 * l_penetration + 1000 * l_rank + l_netwrench + 0.01 * global_q.norm()
            
            #l_all = l_distance + l_normals + 0.0000005 * l_netwrench
            #l_all = 0.1 * l_normals + l_distance
            #l_all = l_distance + l_normals + l_penetration
            #l_all = l_distance
            #l_all = l_distance + l_penetration + l_netwrench
            
            global_q_old = global_q.clone()
            local_q_old = local_q.clone()

            global_q_optimizer.zero_grad()
            l_all.backward()
            #print(local_q)
            global_q_optimizer.step()
                
            
            
            self.model.contact_point0 = contact_point0_before

            with torch.no_grad():
                if i % record_every == 0:
                    hand_vertices = self.hand.forward_sampled(local_q[None,:])
                    rep, ren, _ = self.hand(local_q[None,:])
                    vertices = torch.cat([rep, hand_vertices], dim=-1).transpose(1, 2)
                    if render:
                        vertices *= self.hand_ratio

                        m=stage.GetObjectAtPath("/root/body_0/mesh_0")
                        #vertices = np.array(m.GetAttribute("points").Get(i))*0.9
                        #m.GetAttribute("points").Set(vertices,i+1.0)
                        m.GetAttribute("points").Set(vertices[0,:,:].detach().cpu().numpy(),i+1)
                        renderer.update(state, i+1)
                        stage.Save()

                    try:
                        rank=torch.linalg.matrix_rank(grasp_matrix).item()
                    except:
                        rank=0


                    print(f"{i}/{n} task:{l_all} distance:{l_distance} normal: {l_normals} netwrench:{l_netwrench} rank: {l_rank} {rank} penetration: {l_penetration}")
                    #print(global_q)
                    print(local_q)


                    state = self.model.state()
                    #state.joint_q[self.inds] = global_q[:7]
                    #state.joint_q[self.inds[0:3]] = self.pos_ratio*global_q[0:3]
                    #state.joint_q[self.inds[3:7]] = self.quat_ratio * global_q[3:7]
                    
                    history['joint_q'][i // record_every+1,:] = state.joint_q.detach().cpu()
                    history['local_q'][i // record_every+1,:] = local_q.detach().cpu()
                    history['l_distance'][i // record_every+1] = l_distance.detach().cpu()
                    history['l_netwrench'][i // record_every+1] = l_netwrench.detach().cpu()
                    history['l_rank'][i // record_every+1] = l_rank.detach().cpu()
                    history['l_self_collision'][i // record_every+1] = l_self_collision.detach().cpu()
                    history['l_joint'][i // record_every + 1] = l_joint.detach().cpu()


        with torch.no_grad():
            # record final results
            contact_num = (state.contact_world_dist[self.box_contact_inds][palm_idx] < 0).sum()
            max_penetration = state.contact_world_dist[self.box_contact_inds][palm_idx].min()
            state = self.model.state()
            #state.joint_q[self.inds] = global_q
            #state.joint_q[self.inds[0:3]] = self.pos_ratio*global_q[0:3]
            #state.joint_q[self.inds[3:7]] = self.quat_ratio * global_q[3:7]
            results = {
                'final_joint_q': state.joint_q.detach().cpu(),
                'final_local_q': local_q.detach().cpu(),
                'final_l_rank': l_rank.detach().cpu(),
                'final_l_distance': l_distance.detach().cpu(),
                'final_l_netwrench': l_netwrench.detach().cpu(),
                'final_l_self_collision': l_self_collision.detach().cpu(),
                'final_l_joint': l_joint.detach().cpu(),
                'history': history,
                #'final_l_joint': mdmm_return.fn_values[1].detach().cpu(),
            }
            
            final_joint_q = global_q.detach().cpu().numpy()
            final_local_q = local_q.detach().cpu().numpy()
            #succ_dir = self.geometry_normal_test_num(global_q, global_q)

            f = open(f'/apdcephfs/private_qihangfang/ibsshadow/{self.obj_name}', 'a')
            f.write(f'{self.nums} ')
            for i in range(7):
                f.write(f'{final_joint_q[i]} ')
            for i in range(final_local_q.shape[0]):
                f.write(f'{final_local_q[i]} ')

            try:
                rank=torch.linalg.matrix_rank(grasp_matrix).item()
            except:
                rank=0



            f.write(f'{l_netwrench.item()} {l_distance.item()} {l_penetration.item()} {rank} \n')
            f.close()
            
            return results

    def geometry_test(self, model_state, global_q, mano_q, translations=None):
        return self.geometry_normal_test(global_q, mano_q, translations)
        if translations == None:
            translations = [[0.001,0,0], [-0.001,0,0], [0,0.001,0], [0,-0.001,0], [0,0,0.001], [0,0,-0.001]]
            succ_dir = [False for i in range(len(translations))]
        dist_old = model_state.contact_world_dist[self.box_contact_inds][palm_idx].clone()
        for i in range(len(translations)):
            contact_point0_before = torch.clone(self.model.contact_point0)
            vertices = self.hand.forward_sampled(global_q.reshape(-1)[None,:], self.hand_shape)
            vertices *= self.hand_ratio
            vertices[0, :, 0] = vertices[0, :, 0] + translations[i][0]
            vertices[0, :, 1] = vertices[0, :, 1] + translations[i][1]
            vertices[0, :, 2] = vertices[0, :, 2] + translations[i][2]
            
            self.model.contact_point0[self.box_contact_inds,:] = vertices[0,:,:]
            state = self.model.state()
            #state.joint_q[self.inds] = global_q[:self.n_inds]
            #state.joint_q[self.inds[0:3]] = self.pos_ratio*global_q[0:3]
            #state.joint_q[self.inds[3:7]] = self.quat_ratio * global_q[3:7]

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
    def geometry_normal_test(self, global_q, mano_q, translations=None, sdf_check=True):
        if translations == None:
            translations = torch.tensor([[[1,0,0]], [[-1,0,0]], [[0,1,0]], [[0,-1,0]], [[0,0,1]], [[0,0,-1]]]).to(global_q.device)
            succ_dir = [False for i in range(len(translations))]
            
        contact_point0_before = torch.clone(self.model.contact_point0)
        vertices = self.hand.forward_sampled(global_q[None,:])
        vertices *= self.hand_ratio
        self.model.contact_point0[self.box_contact_inds,:] = vertices[0,:,:]
        state = self.model.state()
        #state.joint_q[self.inds] = global_q[:self.n_inds]
        #state.joint_q[self.inds[0:3]] = self.pos_ratio*global_q[0:3]
        #state.joint_q[self.inds[3:7]] = self.quat_ratio * global_q[3:7].detach()
        m = 1

        for k in range(m):
            state = self.integrator.forward(
                self.model, state, self.sim_dt,
                update_mass_matrix=True)

        hand_mesh = Meshes(state.contact_world_pos[self.box_contact_inds].unsqueeze(0), self.hand_layer.th_faces.unsqueeze(0))
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
    def geometry_normal_test_num(self, global_q, mano_q, translations=None, sdf_check=True):
        if translations == None:
            translations = torch.tensor([[[1,0,0]], [[-1,0,0]], [[0,1,0]], [[0,-1,0]], [[0,0,1]], [[0,0,-1]]]).to(global_q.device)
            succ_dir = [False for i in range(len(translations))]
            
        contact_point0_before = torch.clone(self.model.contact_point0)
        vertices = self.hand.forward_sampled(global_q[None,:])
        vertices *= self.hand_ratio
        self.model.contact_point0[self.box_contact_inds,:] = vertices[0,:,:]
        state = self.model.state()
        #state.joint_q[self.inds] = global_q[:self.n_inds]
        #state.joint_q[self.inds[0:3]] = self.pos_ratio*global_q[0:3]
        #state.joint_q[self.inds[3:7]] = self.quat_ratio * global_q[3:7].detach()
        m = 1

        for k in range(m):
            state = self.integrator.forward(
                self.model, state, self.sim_dt,
                update_mass_matrix=True)

        hand_mesh = Meshes(state.contact_world_pos[self.box_contact_inds].unsqueeze(0), self.hand_layer.th_faces.unsqueeze(0))
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

        hand_vertices = state.contact_world_pos[self.box_contact_inds].cpu().numpy()
        np.save(f'/apdcephfs/private_qihangfang/{self.nums}_vertices.npy', hand_vertices)
        sdf_normals = state.contact_world_n[self.box_contact_inds].cpu().numpy()
        np.save(f'/apdcephfs/private_qihangfang/{self.nums}_sdf_normals.npy', sdf_normals)
        distance = state.contact_world_dist[self.box_contact_inds].cpu().numpy()
        np.save(f'/apdcephfs/private_qihangfang/{self.nums}_distance.npy', distance)
        hand_normals = hand_mesh.verts_normals_packed().cpu().numpy()
        np.save(f'/apdcephfs/private_qihangfang/{self.nums}_hand_normals.npy', hand_normals)



        return succ_dir
    
    

    def force_achieves_task(self, contact_forces):
        old_contact_ke = self.model.contact_ke
        old_contact_kf = self.model.contact_kf
        self.model.contact_ke = 0.0
        self.model.contact_kf = 0.0
        #self.model.contact_kd = 0.00001
        #self.model.contact_kd = level_set_i
        l = 0.0
        for i in range(self.n_vels):
            state = self.model.state()
            state.joint_act[-6:] = -contact_forces[i,:,:].sum(dim=0)
            state.joint_qd[-3:] = self.task_vels[i,:]
            state.joint_q[-3:] += self.task_vels[i,:]*self.sim_dt

            m = 1
            for k in range(m):
                state = self.integrator.forward(
                    self.model, state, self.sim_dt,
                    update_mass_matrix=True)
            #l += 1e5*((state.joint_qd[-6:] - torch.zeros_like(state.joint_qd[-6:]))**2).sum()
            l += 1e5*((state.joint_qd[-6:])**2).sum()
        #self.model.contact_kd = 0.0000
        self.model.contact_ke = old_contact_ke
        self.model.contact_kf = old_contact_kf

        #joint_tau = -contact_forces.sum(dim=1)
        #m = 9.0578
        #joint_qdd = joint_tau/m
        #task_vel = torch.zeros(self.n_vels,6,device=self.device)
        ##task_vel[:,-3:] = self.task_vels
        #joint_qd = task_vel + self.sim_dt * joint_qdd
        #l = 1e5 * (joint_qd**2).sum()
        return l

    def joint_limits(self, global_q):
        l_joint = -torch.minimum(global_q[7:] - self.model.joint_limit_lower[self.inds][7:], torch.zeros_like(global_q[7:])).sum() - torch.minimum(self.model.joint_limit_upper[self.inds][7:] - global_q[7:], torch.zeros_like(global_q[7:])).sum()
        #l_joint += (1.0 - torch.norm(global_q[3:7])).abs()
        return 1e7*l_joint
