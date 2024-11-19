from typing import Union, Sequence, Any

import torch
from lightning import Callback
from lightning.pytorch.utilities.types import STEP_OUTPUT
from torch import nn
from ip.utils.common_utils import printarr, PositionalEncoder, SinusoidalPosEmb
from ip.models.scene_encoder import SceneEncoder
import torch.nn.functional as F
import lightning as L
from torch_geometric.nn import fps, nearest
from ip.utils.running_dataset import RunningDataset
from torch_geometric.data import DataLoader
import warnings
import os
from lightning.pytorch.loggers import WandbLogger
from ip.utils.visualiser import *

warnings.filterwarnings("ignore")


class AutoEncoder(L.LightningModule):
    def __init__(self, local_nn_dims, local_num_freq=10, save_dir=None, save_every=1000,
                 embd_dim=512, log_every_n_steps=500):
        super().__init__()

        self.save_dir = save_dir
        self.save_every = save_every
        self.running_loss = [0] * log_every_n_steps

        self.local_position_encoder = PositionalEncoder(3, local_num_freq, log_space=False)

        local_decoder_dims = local_nn_dims
        local_decoder_dims[-1] = 1
        local_decoder_dims[0] = embd_dim + self.local_position_encoder.d_output
        self.local_decoder = Decoder(local_decoder_dims)

        self.scene_encoder = SceneEncoder(num_freqs=local_num_freq, embd_dim=embd_dim, num_layers=2)
        self.loss_fn = nn.BCEWithLogitsLoss()

        self.plotter = pv.Plotter()
        self.plotter.background_color = 'white'

    def training_step(self, data, batch_idx):
        node_embds, node_pos, node_batch = self.scene_encoder(None, data.pos, data.batch_pos)
        row = nearest(data.queries, node_pos, data.batch_queries, node_batch)

        # add_pcd_to_plotter(self.plotter, node_pos.cpu().numpy(), color='blue', name='scene_c', radius=0.008 * self.scale)
        # add_pcd_to_plotter(self.plotter, data.pos.cpu().numpy(), color='blue', name='scene', radius=0.005 * self.scale)
        # add_parts_to_potter(self.plotter, data.queries.cpu().numpy(), row.cpu().numpy(), cmap='Paired', opacity=.3,
        #                     scale=self.scale, name='a')
        # self.plotter.show(auto_close=False)

        local_queries = self.local_position_encoder(node_pos[row] - data.queries)
        query_x = torch.cat([node_embds[row], local_queries], dim=1)
        occupancy = self.local_decoder(query_x).squeeze()
        target_occupancy = data.occupancy.squeeze()
        loss = self.loss_fn(occupancy, target_occupancy)

        self.running_loss[self.global_step % len(self.running_loss)] = loss
        self.log('train_loss', sum(self.running_loss) / len(self.running_loss),
                 prog_bar=True, on_step=True, on_epoch=False)

        return loss

    def validation_step(self, data, batch_idx):
        node_embds, node_pos, node_batch = self.scene_encoder(None, data.pos, data.batch_pos)
        row = nearest(data.queries, node_pos, data.batch_queries, node_batch)

        local_queries = self.local_position_encoder(node_pos[row] - data.queries)
        query_x = torch.cat([node_embds[row], local_queries], dim=1)
        occupancy = self.local_decoder(query_x).squeeze()
        target_occupancy = data.occupancy.squeeze()
        loss = self.loss_fn(occupancy, target_occupancy)
        acc = ((occupancy > 0).float() == target_occupancy).float().mean()
        self.log('val_loss', loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log('val_acc', acc, prog_bar=True, on_step=False, on_epoch=True)

        ## Visualisation
        add_pcd_to_plotter(self.plotter, data.pos.cpu().numpy(),
                           color='blue', name='scene_c', radius=0.008)
        self.plotter.show(auto_close=False)

        add_pcd_to_plotter(self.plotter, data.pos[row].cpu().numpy(),
                           color='blue', name='scene_c', radius=0.03)
        self.plotter.show(auto_close=False)

        add_pcd_to_plotter(self.plotter, data.queries[occupancy > 0].cpu().numpy(),
                           color='red', name='scene_c', radius=0.008, opacity=1)
        add_pcd_to_plotter(self.plotter, data.queries[occupancy < 0].cpu().numpy(),
                           color='green', name='scene_ca', radius=0.008, opacity=0.03)
        self.plotter.show(auto_close=False)

        return loss.item(), acc.item()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-4, weight_decay=1e-4)
        # lr_scheduler = get_scheduler(
        #     name='cosine',
        #     optimizer=optimizer,
        #     num_warmup_steps=self.config['num_warmup_steps'],
        #     num_training_steps=self.config['num_iters'],
        # )
        return optimizer

    def save_encoder(self, path):
        torch.save(self.scene_encoder.state_dict(), path)

    def save_chpt(self, path):
        self.trainer.save_checkpoint(path)

    def on_train_batch_end(self, *args, **kwargs):
        if self.global_step % self.save_every == 0 and record:
            self.save_encoder(f'{self.save_dir}/scene_encoder.pt')
            self.save_chpt(f'{self.save_dir}/model.pt')


class Decoder(nn.Module):
    def __init__(self, nn_dims):
        super().__init__()

        self.linear_layers = nn.ModuleList([nn.Linear(nn_dims[i], nn_dims[i + 1]) for i in range(len(nn_dims) - 1)])
        self.act = nn.GELU(approximate='tanh')

    def forward(self, x):
        for i, layer in enumerate(self.linear_layers):
            if i == 0 or i == len(self.linear_layers) - 1:
                x = layer(x)
            else:
                x = x + layer(x)
            if i != len(self.linear_layers) - 1:
                x = self.act(x)
        return x


if __name__ == '__main__':
    run_name = 'REC_768_NO_LN'
    record = False
    save_dir = f'./runs/{run_name}/'
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)

    model = AutoEncoder(local_nn_dims=[1, 512, 512, 512],
                        save_dir=save_dir,
                        save_every=5000,
                        local_num_freq=10,
                        embd_dim=768,
                        log_every_n_steps=500)

    dset = RunningDataset(data_path='/home/vv19/reconstruction_data', num_samples=5000, rec=True,
                          random_rotation=False)

    val_dset = [dset[i] for i in range(10)]
    dataloader = DataLoader(dset, batch_size=10, shuffle=True, num_workers=4)

    dataloader_val = DataLoader(val_dset, batch_size=10, shuffle=False, num_workers=4)
    if record:
        logger = WandbLogger(project='Imperio',
                             name=run_name,
                             save_dir=save_dir,
                             log_model=False)
    else:
        logger = None

    trainer = L.Trainer(
        enable_checkpointing=False,  # We save the models manually.
        accelerator='gpu',
        devices=1,
        max_steps=10000000000000000000,
        enable_progress_bar=True,
        precision='16-mixed',
        val_check_interval=5000,
        num_sanity_val_steps=0,
        check_val_every_n_epoch=None,
        logger=logger,
        log_every_n_steps=500,
    )

    trainer.fit(
        model=model,
        train_dataloaders=dataloader,
        val_dataloaders=dataloader_val,
    )
