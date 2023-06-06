import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import pyvista as pv
from scipy.spatial.transform import Rotation
from functools import partial
import pytorch3d.transforms.rotation_conversions as tf

import mdmm
from pyquaternion import Quaternion
from hydra.utils import to_absolute_path
import trimesh
from manotorch.manolayer import ManoLayer
#from manopth.manolayer import ManoLayer as manopth

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

new_faces = np.array([[92, 234, 239],[92, 239, 279],[92, 279, 215],[92, 215, 214],[92, 214, 121],[92, 121, 78],[92, 78, 79],[92, 79, 108],[92, 108, 120],[92, 120, 119],[92, 119, 117],[92, 117, 118],[92, 118, 122],[92, 122, 38]])

def get_sample_intersect_volume(hand_mesh, obj_mesh):
    obj_vox = obj_mesh.voxelized(pitch=0.01)
    obj_points = obj_vox.points
    inside = hand_mesh.contains(obj_points)
    volume = inside.sum() * np.power(0.5, 3)

    return volume

def cmpToKey(mycmp):
    'Convert a cmp= function into a key= function'
    class K(object):
        def __init__(self,obj,*args):
            self.obj=obj
        def __lt__(self,other):
            return mycmp(self.obj,other.obj)<0
        def __gt__(self,other):
            return mycmp(self.obj,other.obj)>0
        def __eq__(self,other):
            return mycmp(self.obj,other.obj)==0
        def __le__(self,other):
            return mycmp(self.obj,other.obj)<=0  
        def __ge__(self,other):
            return mycmp(self.obj,other.obj)>=0
        def __ne__(self,other):
            return mycmp(self.obj,other.obj)!=0
    return K


def _usd_set_xform(xform, transform, scale, time):

    xform_ops = xform.GetOrderedXformOps()

    pos = tuple(transform[0])
    rot = tuple(transform[1])

    xform_ops[0].Set(Gf.Vec3d(pos), time)
    xform_ops[1].Set(Gf.Quatf(rot[3], rot[0], rot[1], rot[2]), time)
    xform_ops[2].Set(Gf.Vec3d(scale), time)


def get_directions(res=4, dim=6):
    dirs=[]
    res=res
    dim=dim
    
    def addDirs(d0,d,n):
        if d0==dim:
            dirs.append(n.astype(np.float32))
        elif d0==d:
            addDirs(d0+1,d,n)
        else:
            for i in range(res):
                n[d0]=-1+2*i/float(res-1)
                addDirs(d0+1,d,n)
    
    n=np.array([0.0]*dim,dtype=np.float32)
    for d in range(dim):
        n[d]= 1
        addDirs(0,d,n)
        n[d]=-1
        addDirs(0,d,n)
    #sort dir
    def cmp(A,B):
        for d in range(dim):
            if A[d]<B[d]:
                return -1
            elif A[d]>B[d]:
                return 1
        return 0
    dirs=sorted(dirs,key=cmpToKey(cmp))
    #make compact
    j=0
    for i in range(len(dirs)):
        if i>0 and (dirs[i]==dirs[i-1]).all():
            continue
        else: 
            dirs[j]=dirs[i]
            j+=1
    dirs=dirs[0:j]
    #normalize
    for i in range(len(dirs)):
        dirs[i]/=np.linalg.norm(dirs[i])
        
    dirs = torch.from_numpy(np.array(dirs))
    return dirs
    
    
    
    



def compute_Q1(grasp_matrix, normal, distance, directions, mu=0):
    grasp_matrix = grasp_matrix.view([1, -1, 6, 3])
    normal = normal.view([1, -1, 3, 1])
    mask = distance < 0.01
    if mask.sum() == 0:
        return 0
    else:
        grasp_matrix = grasp_matrix[:,mask,:,:]
        normal = normal[:,mask,:,:]

    np = mask.sum()
    #print(np)
    nd = directions.shape[0]
    
    
    
    #w_perp
    grasp_matrix_n=torch.matmul(grasp_matrix,normal).view([1,np,1,6,1])
    w_perp=torch.matmul(directions.view([nd,1,6]).type(grasp_matrix_n.type()),grasp_matrix_n).view([-1,np,nd])
    #w_para
    grasp_matrix_Innt=grasp_matrix.view([-1,np,1,6,3])-torch.matmul(grasp_matrix_n,normal.view([-1,np,1,1,3]))
    w_para=torch.matmul(directions.view([nd,1,6]).type(grasp_matrix_Innt.type()),grasp_matrix_Innt).view([-1,np,nd,3])
    w_para=torch.norm(w_para,p=None,dim=3)
    #support analytic
    cond=w_perp*mu>w_para
    #print(cond.sum(), w_para.min())
    in_cone=w_perp+w_para**2/w_perp
    not_in_cone=torch.clamp(w_perp+mu*w_para,min=0)
    support=torch.where(cond,in_cone,not_in_cone)
    #Q1
    Q1,max_index=torch.max(support,dim=1)
    #print(Q1)
    Q1,min_index=torch.min(Q1,dim=1)
    #print(Q1.item(), min_index)
    return Q1.item()

class REF():
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
        
        #self.manopth = manopth(ncomps=self.ncomps, root_rot_mode='axisang', robust_rot=False, mano_root=to_absolute_path('/apdcephfs/private_qihangfang/Data/fcdata/assets_mano/models'), flat_hand_mean=True).to(self.device)

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
        self.obj_config = obj_config

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
            self.obj_mesh = trimesh.load(to_absolute_path(obj_config.mesh_path))
            self.obj_mesh.vertices = self.obj_mesh.vertices * 0.2

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
            '''  
            path = os.path.join(f'/apdcephfs/share_1330077/qihangfang/test/quat.obj')                
            handmesh = trimesh.Trimesh(state.contact_world_pos.detach().cpu().numpy(), self.mano_layer.th_faces.detach().cpu().numpy())
            handmesh.export(path)
            
            mano_input = torch.zeros(3+self.ncomps, device=self.device)
            mano_output = self.mano_layer(mano_input[None,:], self.mano_shape)
            mano_vertices = mano_output.verts
            mano_vertices *= self.mano_ratio
            path = os.path.join(f'/apdcephfs/share_1330077/qihangfang/test/tpose.obj')                
            handmesh = trimesh.Trimesh(mano_vertices[0].detach().cpu().numpy(), self.mano_layer.th_faces.detach().cpu().numpy())
            handmesh.export(path)


            mano_input = torch.zeros(3+self.ncomps, device=self.device)
            mano_output = self.mano_layer(mano_input[None,:], self.mano_shape)
            mano_vertices = mano_output.verts
            mano_vertices *= self.mano_ratio
            
            path = os.path.join(f'/apdcephfs/share_1330077/qihangfang/test/tpose.obj')                
            handmesh = trimesh.Trimesh(mano_vertices[0].detach().cpu().numpy(), self.mano_layer.th_faces.detach().cpu().numpy())
            handmesh.export(path)
            '''

            
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
        #hand_q = torch.tensor([0.1201, -0.2255,  0.0062, -0.9454, -0.2133, -0.2378,  0.0644]).float().to(self.device)

        contact_forces = torch.zeros(self.n_vels, self.box_contact_inds.numel(), 6, device=self.device)
        #for i in range(self.n_vels):
            #contact_forces[i,:,-3:] = 1e6 * self.task_vels[i,:] / self.box_contact_inds.numel()

        return (hand_q, mano_q, contact_forces)

    @torch.no_grad()
    def physical_test(self, hand_q, mano_q, mu=0, contact_kf=1e8):
        gravity_old = self.model.gravity
        rand_x = torch.sin(torch.rand(5, 1, device=self.device) * 2 * np.pi)
        rand_y = torch.sin(torch.rand(5, 1, device=self.device) * 2 * np.pi)
        rand_z = torch.sin(torch.rand(5, 1, device=self.device) * 2 * np.pi)
        test_gravities = torch.cat([rand_x, rand_y, rand_z], dim=1)
        test_gravities = test_gravities / test_gravities.norm(dim=1).unsqueeze(-1) * 0.02
        #test_gravities = torch.tensor([[0.,0.,0.], [1.,0.,0.], [-1.,0.,0.], [0.,1.,0.], [0.,-1.,0.], [0.,0.,1.], [0.,0.,-1.]], device=self.device, dtype=torch.float32)*0.02
        mu_old = self.model.contact_mu
        contact_kf_old = self.model.contact_kf
        self.model.contact_mu = mu
        self.model.contact_kf = contact_kf
        
        contact_point0_before = torch.clone(self.model.contact_point0)
        mano_output = self.mano_layer(mano_q[None,:], self.mano_shape)
        vertices = mano_output.verts
        vertices *= self.mano_ratio
        
        velocity = 0
        for i in range(len(test_gravities)):
            test_gravity = test_gravities[i]
            self.model.gravity = test_gravities[0]
            
            self.model.contact_point0[:778,:]= vertices[0,:,:]
            state = self.model.state()
            state.joint_q[self.inds] = hand_q[:self.n_inds].detach()
            state.joint_q[self.inds[0:3]] = self.pos_ratio*hand_q[0:3]
            state.joint_q[self.inds[3:7]] = self.quat_ratio * hand_q[3:7]
            state.joint_qd[-3:] = test_gravity[:]
            state.joint_q[-3:] += test_gravity[:]*0.1
            m = 1
            for k in range(m):
                state = self.integrator.forward(
                    self.model, state, self.sim_dt,
                    update_mass_matrix=True)
            self.model.contact_point0 = contact_point0_before
            self.model.gravity = gravity_old
            #print(state.joint_qd[-3:], test_gravity[:])
            #print(state.body_v_s)


            velocity += ((state.joint_qd[-3:])**2).sum().sqrt()
        velocity = velocity / test_gravities.shape[0]
        self.model.contact_mu = mu_old
        self.model.contact_kf = contact_kf_old
        return velocity.item()
        

    def run(self, initial_guess):
        q1 = torch.zeros(11, device=self.device)
        simution = torch.zeros(11, device=self.device)


        directions = get_directions()
        #deny_lis= [3, 4, 5, 9, 12, 14, 17, 19, 28, 29, 30, 31, 32, 34, 35, 36, 37, 38, 41, 42, 43, 44]
        #refs = np.loadtxt(f'/apdcephfs/private_qihangfang/pure_{self.obj_name}')[:, 1:-2]
        #refs = np.loadtxt(f'/apdcephfs/private_qihangfang/pure_{self.obj_name}')[:100, 1:-2]
        #refs = trimesh.load(f'/apdcephfs/share_1330077/qihangfang/baselines/das/002_master_chef_can/0.obj')
        #os.makedirs(f'/apdcephfs/private_qihangfang/rl/{self.obj_name}', exist_ok=True)
        refs = np.loadtxt(f'/apdcephfs/private_qihangfang/ibsdataset/{self.obj_name}')[:100]
        #refs = np.loadtxt(f'/apdcephfs/private_qihangfang/ibsdataset/{self.obj_name}')[:100, 1:-8]
        count = 0
        select_list = []
        contact_points = []
        penetration_points = []
                
        for ii in range(100):
            #if ii in deny_lis:
            #    continue
            ref = refs[ii]
            hand_q, mano_q = torch.from_numpy(ref[1:8]).float().cuda(), torch.from_numpy(ref[8:-8]).float().cuda()
            #mesh = trimesh.load(f'/apdcephfs/share_1330077/qihangfang/baselines/das/{self.obj_name}/{ii}.obj')
            #vertices = torch.from_numpy(mesh.vertices).float().to(self.device).unsqueeze(0) * 2.0
            contact_point0_before = torch.clone(self.model.contact_point0)
            mano_output = self.mano_layer(mano_q[None,:], self.mano_shape)
            vertices = mano_output.verts
            vertices *= self.mano_ratio
                
                
            self.model.contact_point0[:778,:]= vertices[0,:,:]
            state = self.model.state()
            state.joint_q[self.inds] = hand_q[:self.n_inds].detach()
            state.joint_q[self.inds[0:3]] = self.pos_ratio*hand_q[0:3]
            #state.joint_q[self.inds[3:7]] = self.quat_ratio * hand_q[3:7]


            m = 1
            for k in range(m):
                state = self.integrator.forward(
                    self.model, state, self.sim_dt,
                    update_mass_matrix=True)
            self.model.contact_point0 = contact_point0_before
            
            contact_world_pos = state.contact_world_pos

            l_distance = torch.mean(state.contact_world_dist[self.box_contact_inds][palm_idx].abs())
            #l_distance_tip = torch.mean(state.contact_world_dist[self.box_contact_inds][tip_idx].abs())
            l_distance_finger = 0
            for i in range(len(finger_idx)):
                l_distance_finger += state.contact_world_dist[self.box_contact_inds][finger_idx[i]].abs().min()
            
            l_distance = l_distance + l_distance_finger / 5
            
            hand_mesh = Meshes(state.contact_world_pos[self.box_contact_inds].unsqueeze(0), self.mano_layer.th_faces.unsqueeze(0))
            hand_normals = hand_mesh.verts_normals_packed()

            l_normals = 1 + torch.sum(hand_normals[palm_idx] * state.contact_world_n[self.box_contact_inds][palm_idx], dim=-1)
            l_normals = torch.mean(l_normals * l_normals)

            net_wrench = torch.matmul(state.contact_matrix[self.box_contact_inds][palm_idx], state.contact_world_n[self.box_contact_inds][palm_idx].unsqueeze(-1))            # x, 6, 3
            net_wrench[:, 3:] = net_wrench[:, 3:] * 0.001
            net_wrench = net_wrench.squeeze(-1) * state.contact_world_dist[self.box_contact_inds][palm_idx].unsqueeze(dim=1)
            net_wrench = net_wrench.mean(dim = 0)
            l_netwrench = (net_wrench * net_wrench).sum()

            grasp_matrix = state.contact_matrix[self.box_contact_inds][palm_idx].permute(1, 2, 0).reshape(6, -1)
            temp = torch.tensor(0.001).float().to(self.device) * torch.tensor(np.eye(6)).float().to(self.device)
            temp = torch.matmul(grasp_matrix, grasp_matrix.transpose(0, 1)) - temp
            eigval = torch.linalg.eigh(temp.cpu())[0].to(self.device)
            rnev = F.relu(-eigval)
            l_rank = torch.sum(rnev * rnev)

            l_penetration = torch.sum(F.relu(-state.contact_world_dist[self.box_contact_inds]))

            #path = os.path.join(f'/apdcephfs/share_1330077/qihangfang/test/{self.obj_name}/after_{ii}.obj')                
            #handmesh = trimesh.Trimesh(contact_world_pos.detach().cpu().numpy(), self.mano_layer.th_faces.detach().cpu().numpy())
            #handmesh.export(path)

            
            #print([state.contact_world_dist[self.box_contact_inds][finger_idx[i]].abs().min().item() for i in range(5)])
            #print(l_distance, l_penetration, l_netwrench)

            try:
                os.makedirs(f'/apdcephfs/share_1330077/qihangfang/baseline/fc/{self.obj_name}')
            except OSError:
                pass


            path = os.path.join(f'/apdcephfs/share_1330077/qihangfang/baseline/fc/{self.obj_name}/post_{ii}.obj')                
            handmesh = trimesh.Trimesh(contact_world_pos.detach().cpu().numpy(), np.concatenate([self.mano_layer.th_faces.detach().cpu().numpy(), new_faces],axis=0))
            handmesh.export(path)
            #exit()


            try:
                rank = torch.linalg.matrix_rank(grasp_matrix)
            except:
                rank = 0
            #if (l_netwrench < 0.5) and (rank == 6):
            if True:
                for j in range(11):
                    mu = 0.1 * j
                    contact_kf = 1e7 * j
                    q1[j] += compute_Q1(state.contact_matrix[self.box_contact_inds][palm_idx], state.contact_world_n[self.box_contact_inds][palm_idx], state.contact_world_dist[self.box_contact_inds][palm_idx], directions, mu=mu)
                    simution[j] += self.physical_test(hand_q, mano_q, mu=mu, contact_kf=contact_kf)
                #print(simution[0])
            #for i in range(5):
            #    print(state.contact_world_dist[self.box_contact_inds][finger_idx[i]].abs().min())
            #
            #print(q1)
            #exit()
            #print(l_penetration)
            #if state.contact_world_dist[self.box_contact_inds][tip_idx][0] < 0.25:
            #    if (state.contact_world_dist[self.box_contact_inds][tip_idx] < 0.25).sum() >= 2:
            if (l_netwrench < 1e-3) and (rank == 6):
                select_list.append(ii)
                #print(((state.contact_world_dist[self.box_contact_inds][palm_idx].abs() < 0.01).sum()).item(), ((state.contact_world_dist[self.box_contact_inds] < -(0.01)).sum()).item())
            contact_points.append(((state.contact_world_dist[self.box_contact_inds][palm_idx].abs() < 0.01).sum()).item())
            penetration_points.append(get_sample_intersect_volume(handmesh, self.obj_mesh))


            #l_phy = 0
            #for i in range(self.n_vels):
            #    contact_point0_before = torch.clone(self.model.contact_point0)
            #    self.model.contact_point0[:778,:]= vertices[0,:,:]
            #    state = self.model.state()
            #    state.joint_qd[-3:] = self.task_vels[i,:]
            #    state.joint_q[-3:] += self.task_vels[i,:]*self.sim_dt
                
            #    #print(state.joint_qd)
            #    m = 1
            #    for k in range(m):
            #        state = self.integrator.forward(
            #            self.model, state, self.sim_dt,
            #            update_mass_matrix=True)
            #    self.model.contact_point0 = contact_point0_before
            #    l_phy += ((state.joint_qd[-6:])**2).sum()
            #    #print(state.joint_qd)

            #path = os.path.join(os.path.dirname(self.obj_config.sdf_path), f'fc_{ii}.obj')                
            #handmesh = trimesh.Trimesh(contact_world_pos.detach().cpu().numpy(), self.mano_layer.th_faces.detach().cpu().numpy())
            #handmesh.export(path)

                

            #print(ii, 'distance:', l_distance.item(), 'normal', l_normals.item(), 'wernch', l_netwrench.item(), 'rank', l_rank.item(), 'phy', l_phy.item(), 'l_penetration', l_penetration.item())
            #print(state.contact_world_dist[self.box_contact_inds][tip_idx])
            #print('------------')
        f = open('/apdcephfs/share_1330077/qihangfang/baselines/results_fc', 'a')
        out_str = f'{self.obj_name},{len(select_list)},{np.mean(contact_points)},{np.mean(penetration_points),}'
        q1_auc = q1.mean()
        simution_auc = simution.mean()
        for i in range(11):
            out_str += (f'{q1[i].item()},')
        for i in range(11):
            out_str += (f'{simution[i].item()},')
        out_str += f'{q1_auc},'
        out_str += f'{simution_auc}\n'
        f.write(out_str)
        f.close()
        exit()
        #print(len(select_list), len(refs), np.mean(contact_points), np.mean(penetration_points))
        #np.savetxt(os.path.join(os.path.dirname(self.obj_config.sdf_path), 'target_pose.txt'), refs[select_list])
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

        return l

    def joint_limits(self, hand_q):
        l_joint = -torch.minimum(hand_q[7:] - self.model.joint_limit_lower[self.inds][7:], torch.zeros_like(hand_q[7:])).sum() - torch.minimum(self.model.joint_limit_upper[self.inds][7:] - hand_q[7:], torch.zeros_like(hand_q[7:])).sum()
        #l_joint += (1.0 - torch.norm(hand_q[3:7])).abs()
        return 1e7*l_joint
