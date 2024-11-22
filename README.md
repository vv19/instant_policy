# Instant Policy

Code for the paper: "Instant Policy: In-Context Imitation Learning via Graph Diffusion", 
[Project Webpage](https://www.robot-learning.uk/instant-policy)

<p align="center">
<img src="./media/rollout_roll.gif" alt="drawing" width="700"/>
</p>

## Setup

**Clone this repo**

```
git clone https://github.com/vv19/instant_policy.git
cd instant_policy
```

**Create conda environment**

```
conda env create -f environment.yml
conda activate ip_env
pip install pyg-lib -f https://data.pyg.org/whl/torch-2.2.0+cu118.html
pip install -e .
```

Install RLbench by following the instructions in the https://github.com/stepjam/RLBench.

## Quick Start

### Try our pre-trained model for RLBench tasks.

Download pre-trained weights.

```
cd ip
./scripts/download_weights.sh
```

Run inference.

```
python eval.py \
 --task_name='plate_out' \
 --num_demos=2 \
 --num_rollouts=10
```

Try it out with different tasks, e.g. `open_box` or `toilet_seat_down`! More in `utils/rl_bench_tasks.py`.

## Deploy on Your Robot

Every robot (and its user) uses different controllers and gets observations in different ways. 
In `deployment.py`, we provide examples of how to use Instant Policy for deployment on any robotic manipulator using parallel-jaw gripper. 
Plug in your controller, get observations in a form of segmented point clouds, end-effector poses and gripper states, and you are all set! 

## Training and Fine-tuning

To train the graph diffusion model from scratch or fine-tune it using your own data, use `train.py`.
First, you'll have to convert your data into appropriate format. Example of how to do it can be found in `prepare_data.py`. 

Then to fine-tune your model, run: 
```
python train.py \
 --run_name='fine-tunning_ip' \
 --record=1 \
 --use_wandb=1 \
 --fine_tune=1 \
 --data_path_train='PATH/TO/TRAIN/DATA' \
 --data_path_val='PATH/TO/VAL/DATA' \
```

For more argument options, use `python train.py --help` and see parameters defined in `configs/base_config.py`. 

## Notes on Observed Performance

To reach the best performance when deploying the current implementation of Instant Policy, there is a number of things to consider:

- Objects of interest should be well segmented.
- Tasks should follow Markovian assumption (there is no history of observations).
- Demonstrations should be short and consistent, without a lot of task irrelevant motions.
- Inference parameters (e.g. number of demonstrations and number of diffusion timesteps) can greatly influence the performance.
- Model uses segmented point clouds expressed in the end-effector frame -- when an object is grasped, there needs to be at least one more object in the observation to ground the motion.
- Compiling the model and using fewer diffusion steps will result in significantly faster inference times.

If the deployed policy doesn't perform well, please feel free to contact me, I'll be happy!

# Citing

If you find our paper interesting or this code useful in your work, please cite our paper:

```
@article{vosylius2024instant,
  title={Instant Policy: In-Context Imitation Learning via Graph Diffusion},
  author={Vosylius, Vitalis and Johns, Edward},
  journal={arXiv preprint arXiv:2411.12633},
  year={2024}
}
```

