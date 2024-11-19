from ip.utils.common_utils import pose_to_transform, downsample_pcd, transform_pcd
import numpy as np
from rlbench.backend.spawn_boundary import BoundingBox
from tqdm import tqdm
from rlbench.action_modes.action_mode import MoveArmThenGripper
from rlbench.action_modes.arm_action_modes import JointVelocity, EndEffectorPoseViaIK
from rlbench.action_modes.gripper_action_modes import Discrete
from rlbench.environment import Environment
from rlbench.observation_config import CameraConfig, ObservationConfig
from tqdm import trange
from ip.utils.common_utils import *
from ip.utils.data_proc import *
from ip.utils.rl_bench_tasks import TASK_NAMES


def rollout_model(model, num_demos, task_name='phone_on_base', max_execution_steps=30,
                  execution_horizon=8, num_rollouts=2, headless=False, num_traj_wp=10, restrict_rot=True):
    ####################################################################################################################
    obs_config = ObservationConfig()
    obs_config.set_all(True)
    action_mode = MoveArmThenGripper(
        arm_action_mode=EndEffectorPoseViaIK(),
        gripper_action_mode=Discrete()
    )
    env = Environment(action_mode,
                      './',
                      obs_config=obs_config,
                      headless=headless)
    env.launch()
    task = env.get_task(TASK_NAMES[task_name])

    def temp(position, euler=None, quaternion=None, ignore_collisions=False, trials=300, max_configs=1,
             distance_threshold=0.65, max_time_ms=10, trials_per_goal=1, algorithm=None, relative_to=None):
        return env._robot.arm.get_linear_path(position, euler, quaternion, ignore_collisions=ignore_collisions,
                                              relative_to=relative_to)

    env._robot.arm.get_path = temp
    env._scene._start_arm_joint_pos = np.array([6.74760377e-05, -1.91104114e-02, -3.62065766e-05, -1.64271665e+00,
                                                -1.14094291e-07, 1.55336857e+00, 7.85427451e-01])

    rot_bounds = env._scene.task.base_rotation_bounds()
    mean_rot = (rot_bounds[0][2] + rot_bounds[1][2]) / 2
    if restrict_rot:
        env._scene.task.base_rotation_bounds = lambda: ((0.0, 0.0, max(rot_bounds[0][2], mean_rot - np.pi / 3)),
                                                        (0.0, 0.0, min(rot_bounds[1][2], mean_rot + np.pi / 3)))

    ####################################################################################################################
    full_sample = {
        'demos': [dict()] * num_demos,
        'live': dict(),
    }

    for i in tqdm(range(num_demos), desc=f'Collecting demos', total=num_demos, leave=False):
        done = False
        while not done:
            try:
                # task.set_variation(i % 2 * 2)
                demos = task.get_demos(1, live_demos=True, max_attempts=1000)  # -> List[List[Observation]]
                sample = rl_bench_demo_to_sample(demos[0])
                full_sample['demos'][i] = sample_to_cond_demo(sample, num_traj_wp)
                assert len(full_sample['demos'][i]['obs']) == num_traj_wp
                done = True
            except:
                continue

    ####################################################################################################################
    successes = []
    pbar = trange(num_rollouts, desc=f'Evaluating model, SR: 0/{num_rollouts}', leave=False)
    for i in pbar:
        done = False
        while not done:
            try:
                task.reset()
                done = True
            except:
                continue

        env_action = np.zeros(8)
        # number of steps in rollouts.
        success = 0
        for k in range(max_execution_steps):
            curr_obs = task.get_observation()
            T_w_e = pose_to_transform(curr_obs.gripper_pose)
            full_sample['live']['obs'] = [transform_pcd(subsample_pcd(get_point_cloud(curr_obs)),
                                                        np.linalg.inv(T_w_e))]
            full_sample['live']['grips'] = [curr_obs.gripper_open]
            full_sample['live']['actions_grip'] = [np.zeros(8)]
            full_sample['live']['T_w_es'] = [T_w_e]
            full_sample['live']['actions'] = [full_sample['live']['T_w_es'][0].reshape(1, 4, 4).repeat(8, axis=0)]
            data = save_sample(full_sample, None)

            if k == 0:
                demo_scene_node_embds, demo_scene_node_pos = model.model.get_demo_scene_emb(
                    data.to(model.config['device']))
            live_scene_node_embds, live_scene_node_pos = model.model.get_live_scene_emb(data.to(model.config['device']))
            data.live_scene_node_embds = live_scene_node_embds.clone()
            data.live_scene_node_pos = live_scene_node_pos.clone()
            data.demo_scene_node_embds = demo_scene_node_embds.clone()
            data.demo_scene_node_pos = demo_scene_node_pos.clone()

            with torch.no_grad():
                with torch.autocast(dtype=torch.float32, device_type=model.config['device']):
                    actions, grips = model.test_step(data.to(model.config['device']), 0)
                actions = actions.squeeze().cpu().numpy()
                grips = grips.squeeze().cpu().numpy()

            for j in range(execution_horizon):
                env_action[:7] = transform_to_pose(T_w_e @ actions[j])
                env_action[7] = int((grips[j] + 1) / 2 > 0.5)
                try:
                    curr_obs, reward, terminate = task.step(env_action)
                    success = int(terminate and reward > 0.)
                except Exception as e:
                    terminate = True
                if terminate:
                    break

            else:
                continue
            break
        successes.append(success)
        pbar.set_description(f'Evaluating model, SR: {sum(successes)}/{len(successes)}')
        pbar.refresh()
    pbar.close()
    env.shutdown()
    return sum(successes) / len(successes)
    ####################################################################################################################


def override_bounds(pos, rot, env):
    if pos is not None:
        BoundingBox.within_boundary = lambda x, y, z: True  # Where we are going, we don't need boundaries
        env._scene._workspace_boundary._boundaries[0]._get_position_within_boundary = lambda x, y: pos
    env._scene.task.base_rotation_bounds = lambda: ((0.0, 0.0, rot - 0.0001), (0.0, 0.0, rot + 0.0001))


def rl_bench_demo_to_sample(demo):
    sample = {'pcds': [], 'T_w_es': [], 'grips': []}

    for k, obs in enumerate(demo):
        pcd = get_point_cloud(obs)
        sample['pcds'].append(pcd)
        sample['T_w_es'].append(pose_to_transform(obs.gripper_pose))
        sample['grips'].append(obs.gripper_open)

    return sample


def get_point_cloud(obs, camera_names=('front', 'left_shoulder', 'right_shoulder')):
    pcds = []
    for camera_name in camera_names:
        ordered_pcd = getattr(obs, f'{camera_name}_point_cloud')
        mask = getattr(obs, f'{camera_name}_mask')
        masked_pcd = ordered_pcd[mask > 60]  # Hack to get segmentations easily.
        pcds.append(masked_pcd)

    return downsample_pcd(np.concatenate(pcds, axis=0))
