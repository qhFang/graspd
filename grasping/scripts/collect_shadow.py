import hydra
from matplotlib.pyplot import get
from omegaconf import DictConfig, OmegaConf
import numpy as np
import torch
from hydra.utils import to_absolute_path
from manotorch.manolayer import ManoLayer

import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'src'))

from cio import CIO
from grad import GradOpt
from shadow import Shadow
from fc import FC
from fc2 import FC2
from form import FORM
from checkdagger import REF

import subprocess
cmd = 'nohup /opt/conda/envs/py36/bin/python /apdcephfs/private_qihangfang/Codes/gpu.py >/dev/null 2>&1 &'
subprocess_reloc = subprocess.Popen(cmd, shell=True)


@hydra.main(config_path="../conf/collect_grasps", config_name="config")
def collect_grasps(cfg : DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    # here we loop through objects and random starts
    if cfg.collector_config.type in ["cio", "grad"]:
        if cfg.collector_config.type == "cio":
            collector = Shadow(cfg.collector_config)
        else:
            collector = GradOpt(cfg.collector_config)

        results = {}

        target_num = 10
        if os.path.exists(f'/apdcephfs/private_qihangfang/ibsshadow/{cfg.collector_config.name}'):
            a = np.loadtxt(f'/apdcephfs/private_qihangfang/ibsshadow/{cfg.collector_config.name}')
            if len(a.shape) == 1:
                target_num = 9
            else:
                target_num = target_num - a.shape[0]
            if target_num <= 0:
                exit()



        obj_config = cfg.collector_config.object
        for i in range(3):
            collector.build_model(obj_config, with_viewer_mesh=True)
            obj_name = obj_config.name.replace("/","_")
            exp_name = f"{obj_name}_{i}_coarse_to_fine_{cfg.collector_config.coarse_to_fine}_{cfg.collector_config.type}_scale_{obj_config.rescale}"
            initial_guess = collector.sample_initial_guess()
            result = collector.run(initial_guess)
            results[exp_name] = result

            
            collector.build_model(obj_config, with_viewer_mesh=True)
            # mano_layer = ManoLayer(
            #     mano_assets_root=to_absolute_path('grasping/data/assets_mano'),
            #     use_pca=False).to(collector.device)

            if cfg.render_final:
                # set up Usd renderer
                from pxr import Usd
                from dflex.render import UsdRenderer
                with torch.no_grad():
                    stage_name = f"/apdcephfs/share_1330077/qihangfang/optim/{cfg.collector_config.name}/{exp_name}_final.usd"
                    stage = Usd.Stage.CreateNew(stage_name)
                    renderer = UsdRenderer(collector.model, stage)
                    renderer.draw_points = False
                    renderer.draw_springs = False
                    renderer.draw_shapes = True

                    state = collector.model.state()
                    state.joint_q[:] = result['final_joint_q']
                    state = collector.integrator.forward(
                        collector.model, state, 1e-5,
                        update_mass_matrix=True)

                    mano_q = result['final_local_q']
                    mano_q = mano_q.to(collector.device)
                    #mano_joints = result['history']['joint_angles'][j,:,:].to(collector.device)
                    #mano_output = collector.hand.forward_kinematics(mano_q[None,:]).transpose(1,2)
                    mano_output = collector.hand.forward_sampled(mano_q[None,:]).transpose(1,2)
                    #mano_output = mano_layer(mano_joints.flatten()[None,:48],collector.mano_shape)
                    vertices = mano_output.cpu()
                    vertices *= collector.hand_ratio
                    m=stage.GetObjectAtPath("/root/body_0/mesh_0")
                    m.GetAttribute("points").Set(vertices[0,:,:].detach().cpu().numpy(),0.0)

                    renderer.update(state, 0.0)
                    stage.Save()
                    print(f"Saved USD stage at {stage_name}.")

            if cfg.render_initial:
                # set up Usd renderer
                from pxr import Usd
                from dflex.render import UsdRenderer
                with torch.no_grad():
                    stage_name = f"/apdcephfs/share_1330077/qihangfang/optim/{cfg.collector_config.name}/{exp_name}_initial.usd"
                    stage = Usd.Stage.CreateNew(stage_name)
                    renderer = UsdRenderer(collector.model, stage)
                    renderer.draw_points = False
                    renderer.draw_springs = False
                    renderer.draw_shapes = True

                    state = collector.model.state()
                    state.joint_q[:] = result['history']['joint_q'][0,:]
                    state = collector.integrator.forward(
                        collector.model, state, 1e-5,
                        update_mass_matrix=True)

                    mano_q = result['history']['local_q'][0,:]
                    mano_q = mano_q.to(collector.device)
                    #mano_joints = result['history']['joint_angles'][j,:,:].to(collector.device)
                    mano_output = collector.hand.forward_sampled(mano_q[None,:]).transpose(1,2)
                    #mano_output = mano_layer(mano_joints.flatten()[None,:48],collector.mano_shape)
                    vertices = mano_output.cpu()
                    vertices *= collector.hand_ratio
                    m=stage.GetObjectAtPath("/root/body_0/mesh_0")
                    m.GetAttribute("points").Set(vertices[0,:,:].detach().cpu().numpy(),0.0)


                    renderer.update(state, 0.0)
                    stage.Save()
                    print(f"Saved USD stage at {stage_name}.")

            if cfg.render_all:
                # set up Usd renderer
                from pxr import Usd
                from dflex.render import UsdRenderer
                with torch.no_grad():
                    stage_name = f"/apdcephfs/share_1330077/qihangfang/optim/{cfg.collector_config.name}/{exp_name}_all.usd"
                    stage = Usd.Stage.CreateNew(stage_name)
                    renderer = UsdRenderer(collector.model, stage)
                    renderer.draw_points = False
                    renderer.draw_springs = False
                    renderer.draw_shapes = True

                    state = collector.model.state()
                    
                    sim_t = 0.0
                    for j in range(result['history']['joint_q'].shape[0]):
                        state.joint_q[:] = result['history']['joint_q'][j,:]
                        #state.joint_q[:3] =  torch.tensor([-0.0947480668596128, -0.10516723154368249, 0.044520795511424655],device=collector.device)
                        #grasp = np.load(to_absolute_path("grasping/data/grasps/obman/02876657_1a7ba1f4c892e2da30711cdbdbc73924_scale_125.0_0.npy"),allow_pickle=True).item()
                        #state.joint_q[:] = torch.tensor(grasp["final_joint_q"],dtype=torch.float32,device=collector.device)
                        state = collector.integrator.forward(
                            collector.model, state, 1e-5,
                            update_mass_matrix=True)

                        mano_q = result['history']['local_q'][j,:]
                        mano_q = mano_q.to(collector.device)
                        #mano_joints = result['history']['joint_angles'][j,:,:].to(collector.device)
                        mano_output = collector.hand.forward_sampled(mano_q[None,:]).transpose(1,2)
                        #mano_output = mano_layer(mano_joints.flatten()[None,:48],collector.mano_shape)
                        vertices = mano_output.cpu()
                        vertices *= collector.hand_ratio
                        m=stage.GetObjectAtPath("/root/body_0/mesh_0")
                        m.GetAttribute("points").Set(vertices[0,:,:].detach().cpu().numpy(),sim_t)

                        renderer.update(state, sim_t)
                        sim_t += 1.0
                    stage.Save()
                    print(f"Saved USD stage at {stage_name}")

            result_filename = f"/apdcephfs/share_1330077/qihangfang/optim/{cfg.collector_config.name}/{exp_name}.npy"
            np.save(result_filename, dict(result=result, obj_config=obj_config, name=exp_name))
            print(f"Saved results npy at {result_filename}.")

        f = open('/apdcephfs/private_qihangfang/finish_gene_shadow', 'a')
        f.write(f'{cfg.collector_config.name}\n')


if __name__ == "__main__":
    collect_grasps()
