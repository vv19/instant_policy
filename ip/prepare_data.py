'''
This script shows how to convert demonstrations performing the same task (or a pseudo-task) to a format suitable to
train our In-Context Imitation Learning model.
'''
import torch
from ip.models.scene_encoder import SceneEncoder
from ip.utils.data_proc import *


if __name__ == '__main__':
    ####################################################################################################################
    num_demos = 2  # Demos in the context.
    num_waypoints_demo = 10  # Number of waypoints in one demo.
    pred_horizon = 8  # How many future actions we are predicting.
    # Maximum spacing between subsequent actions (1cm, 3degrees).
    live_spacing_trans = 0.01
    live_spacing_rot = 3
    # We can pre-compute geometry embeddings while processing data, reducing computational requirements during training.
    compute_embeddings = True
    save_dir = './data/train'  # Path to where the data should be saved.
    ####################################################################################################################
    # Load scene encoder if we are pre-computing geometry embeddings.
    if compute_embeddings:
        scene_encoder = SceneEncoder(num_freqs=10, embd_dim=512)
        scene_encoder.load_state_dict(
            torch.load(f'./checkpoints/scene_encoder.pt'))
        scene_encoder = scene_encoder.to('cuda')
        scene_encoder.eval()
    else:
        scene_encoder = None
    ####################################################################################################################
    # Get everything into the right format.
    full_sample = {
        'demos': [dict()] * num_demos,
        'live': dict(),
    }

    # TODO: Collect or load demonstrations in a form of {'pcds': [], 'T_w_es': [], 'grips': []}
    # TODO: You can also shuffle them here or create permutations to create more data samples.
    demos = []
    for i, sample in enumerate(demos):
        if i < num_demos:
            full_sample['demos'][i] = sample_to_cond_demo(sample, num_waypoints_demo)
        else:
            full_sample['live'] = sample_to_live(sample, pred_horizon, 2048,
                                                 live_spacing_trans, live_spacing_rot, subsample=False)
            break
    ####################################################################################################################
    # Save the data. Data will be saved as data_{i + offset}.pt in save_dir for i [0, len(full_sample['live'])].
    # In our experiments, we continuously generate data and save it by adjusting the offset, overwriting the old samples.
    offset = 0
    save_sample(full_sample, save_dir=save_dir, offset=offset, scene_encoder=scene_encoder)
