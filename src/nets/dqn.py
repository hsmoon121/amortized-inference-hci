import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import LinearBlock


class DQNFrame(nn.Module):
    def __init__(self, config, name=None, task_name="sample", model_path=None):
        """
        config: dict, contains network configuration parameters
        name: str, name of the network (optional)
        task_name: str, name of the task (default: "sample")
        model_path: str, path to save or load model checkpoints (optional)
        """
        super().__init__()
        self.name = name
        self.task_name = task_name
        if model_path is None:
            self.model_path = f"src/simulators/{task_name}/models"
        else:
            self.model_path = model_path

        self.in_size = config["obs_size"]
        self.out_size = config["act_size"]
        self.z_size = config["z_size"]
        self.hidden_size = config["hidden_size"]
        self.hidden_depth = config["hidden_depth"]
        
        if "device" in config:
            self.device = torch.device(config["device"])
        else:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    def forward(self, state):
        pass

    def compute_total_params(self):
        """
        Compute the total number of trainable parameters in the network.
        """
        params = 0
        for p in list(self.parameters()):
            params += np.prod(list(p.size()))
        self.total_params = params
        print(f"[ simulator ] total trainable parameters : {self.total_params}")

    def soft_update_params(self, source, tau=0.0001):
        """
        Soft update the model's parameters using a source model.

        source: nn.Module, source model to update from
        tau: float, interpolation factor (default: 0.0001)
        """
        for target_param, source_param in zip(self.parameters(), source.parameters()):
            target_param.data.copy_(tau * source_param.data + (1.0 - tau) * target_param.data)
            
    def hard_update_params(self, source):
        """
        Hard update the model's parameters using a source model.

        source: nn.Module, source model to update from
        """
        self.load_state_dict(source.state_dict())

    def save(self, ep, path=None):
        """
        Save a model checkpoint.

        ep: int, current episode number
        path: str, custom path to save checkpoint (optional)
        """
        if path is None:
            os.makedirs(f"{self.model_path}/{self.name}", exist_ok=True)
            ckpt_path = f"{self.model_path}/{self.name}/{ep//1000:04d}K.pt"
        else:
            os.makedirs(path, exist_ok=True)
            ckpt_path = path + f"{ep//1000:04d}K.pt"
        torch.save({
            "episode": ep,
            "model_state_dict": self.state_dict()
        }, ckpt_path)
        print(f"[ simulator - saved checkpoint - ep: {ep} ]")

    def load(self, return_ep=False):
        """
        Load the latest model checkpoint.

        return_ep: bool, whether to return the loaded episode number (default: False)
        ---
        output: int, loaded episode number if return_ep=True, else None
        """
        ckpt_path = self.find_last_ckpt(f"{self.model_path}/{self.name}")
        if ckpt_path is not None:
            ckpt = torch.load(ckpt_path, map_location=self.device)
            if "model_state_dict" in ckpt.keys():
                self.load_state_dict(ckpt["model_state_dict"])
            else:
                self.load_state_dict(ckpt["model"])
            print(f"[ simulator - loaded checkpoint ]\n\t{ckpt_path}")

            if return_ep:
                ckpt_name = os.path.basename(ckpt_path)
                if ckpt_name.endswith("K.pt"):
                    print(f" - loaded epsiode: {int(ckpt_name[-8:-4])}K")
                    return int(ckpt_name[-8:-4]) * 1000
                else:
                    print(f" - loaded epsiode: 0")
                    return 0
        else:
            print(f"[ simulator - no checkpoint ] start training from scratch.")

    def find_last_ckpt(self, save_path):
        ckpts = os.listdir(save_path)
        ckpts_w_step = [f for f in ckpts if f.endswith("K.pt")]
        if ckpts_w_step:
            return os.path.join(save_path, max(ckpts_w_step))
        elif ckpts:
            return os.path.join(save_path, max(ckpts))
        else:
            return None


class BaseDQN(DQNFrame):
    """
    BaseDQN implements a simple feedforward neural network for the DQN.
    It can either include or exclude the z_size from the input.
    """
    def __init__(self, config, without_z=True, name=None, task_name="sample", model_path=None):
        super().__init__(config, name=name, task_name=task_name, model_path=model_path)
        if without_z:
            self.z_size = 0

        batch_norm = False if "batch_norm" not in config else config["batch_norm"]
        activation = "relu" if "activation" not in config else config["activation"]

        self.primary_net = LinearBlock(
            self.in_size + self.z_size,
            self.out_size,
            hidden_sz=self.hidden_size,
            hidden_depth=self.hidden_depth,
            batch_norm=batch_norm,
            activation=activation,
        )
        self.compute_total_params()
        self.to(self.device)   
    
    def forward(self, state, z=None):
        """
        Forward pass through the BaseDQN network.

        state: A tensor or numpy array representing the input state of shape (batch_size, in_size), 
          where batch_size is the number of input samples and in_size is the dimension of the input state.
        z (optional): If provided, z values will be concatenated with the state before
          being passed to the primary_net.
        ---
        outputs: A tensor of shape (batch_size, out_size), representing the network's output.
        """
        state = torch.Tensor(state).to(self.device)
        if z is not None:
            z = torch.Tensor(z).to(self.device)
            z_batch = z.expand(state.shape[0], -1) if len(state.shape) > 1 else z
            state = torch.cat([state, z_batch], dim=-1)
            
        if state.shape[-1] > self.in_size + self.z_size:
            state = state[..., :self.in_size + self.z_size]
            
        x = state.unsqueeze(0) if len(state.shape) == 1 else state
        x = self.primary_net(x)
        return x


class EmbeddingDQN(DQNFrame):
    """
    EmbeddingDQN implements a more complex feedforward neural network for the DQN.
    This network incorporates embedding layers and can concatenate these embeddings at specified positions
    in the primary network. The primary_layers and embed_nets are lists of layers for the primary network
    and embedding networks, respectively.
    """
    def __init__(self, config, name=None, task_name="sample", model_path=None):
        super().__init__(config, name=name, task_name=task_name, model_path=model_path)

        self.mid_sz = config["mid_sz"]
        self.mid_d = config["mid_d"]
        self.embed_size = config["embed_size"] if not config["no_embed"] else self.z_size
        self.concat_pos = config["concat_pos"]
        self.embed_act = config["embed_act"]

        self.primary_layers = nn.ModuleList([])
        for i in range(self.hidden_depth + 1):
            N_in = self.in_size if i == 0 else self.hidden_size
            N_in += self.embed_size if i in self.concat_pos else 0
            N_out = self.out_size if i == self.hidden_depth else self.hidden_size
            self.primary_layers.append(nn.Linear(N_in, N_out))

        self.embed_nets = nn.ModuleList([])
        for i in range(len(self.concat_pos)):
            if config["no_embed"]:
                self.embed_nets.append(nn.Sequential())
            else:
                self.embed_nets.append(LinearBlock(
                    self.z_size,
                    self.embed_size,
                    self.mid_sz,
                    self.mid_d,
                    activation=self.embed_act
                ))

        self.compute_total_params()
        self.to(self.device)
        
    def forward(self, state, z=None):
        """
        Forward pass through the EmbeddingDQN network.

        state: A tensor or numpy array representing the input state of shape (batch_size, in_size), 
            where batch_size is the number of input samples and in_size is the dimension of the input state.
            If z is not provided, state must also include the z values, resulting in a shape of 
            (batch_size, in_size + z_size).
        z (optional): If not provided, z values are expected to be included in the state.
        ---
        outputs: A tensor of shape (batch_size, out_size), representing the network's output.
        """
        if z is None:
            state, z = torch.Tensor(state).to(self.device).split([self.in_size, self.z_size], -1)
        else:
            state = torch.Tensor(state).to(self.device)
            z = torch.Tensor(z).to(self.device)
            z = z.expand(state.shape[0], -1) if len(state.shape) > 1 else z
            
        if len(state.shape) == 1:
            x = state.unsqueeze(0)
            z = z.unsqueeze(0)
        else:
            x = state
            
        for i in range(self.hidden_depth + 1):
            if i in self.concat_pos:
                embed = self.embed_nets[self.concat_pos.index(i)](z)
                x = torch.cat([x, embed], dim=-1)

            x = self.primary_layers[i](x)
            if i < self.hidden_depth:
                x = F.relu(x)
        return x


class FilmDQN(DQNFrame):
    """
    FilmDQN is a DQN model variant that utilizes Feature-wise Linear Modulation (FiLM) layers.
    """
    def __init__(self, config, name=None, task_name="sample", model_path=None):
        super().__init__(config, name=name, task_name=task_name, model_path=model_path)

        self.mid_sz = config["mid_sz"]
        self.mid_d = config["mid_d"]
        self.no_bias = config["no_bias"]
        self.batch_norm = config["batch_norm"]
        self.film_pos = config["film_pos"]

        self.primary_layers = nn.ModuleList([])
        for i in range(self.hidden_depth + 1):
            N_in = self.in_size if i == 0 else self.hidden_size
            N_out = self.out_size if i == self.hidden_depth else self.hidden_size

            if self.batch_norm and i < self.hidden_depth and (i + 1) in self.film_pos:
                self.primary_layers.append(
                    nn.Sequential(nn.Linear(N_in, N_out), nn.BatchNorm1d(N_out, affine=False))
                )
            else:
                self.primary_layers.append(nn.Linear(N_in, N_out))

        self.film_gen_nets = nn.ModuleList([])
        for i in range(len(self.film_pos)):
            N_out = self.out_size if i == self.hidden_depth else self.hidden_size
            if self.no_bias:
                self.film_gen_nets.append(LinearBlock(
                    self.z_size,
                    N_out,
                    self.mid_sz,
                    self.mid_d
                ))
            else:
                self.film_gen_nets.append(LinearBlock(
                    self.z_size,
                    2 * N_out,
                    self.mid_sz,
                    self.mid_d
                ))

        self.compute_total_params()
        self.to(self.device)
        
    def soft_update_params(self, source, tau=0.0001):
        """
        Softly updates the model parameters from a source model.

        source: nn.Module, source model to update from
        tau: float, interpolation factor (default: 0.0001)
        """
        for target_param, source_param in zip(self.parameters(), source.parameters()):
            target_param.data.copy_(tau * source_param.data + (1.0 - tau) * target_param.data)
            
        if self.batch_norm:
            for i in range(self.hidden_depth):
                if (i + 1) in self.film_pos:
                    self.primary_layers[i][-1].running_mean = source.primary_layers[i][-1].running_mean
                    self.primary_layers[i][-1].running_var = source.primary_layers[i][-1].running_var
    
    def forward(self, state, z=None):
        """
        Forward pass of the FiLM-DQN model.

        state: The input state tensor (or numpy array) with shape (batch_size, in_size).
            If z is not provided, state must also include the z values, resulting in a shape of 
            (batch_size, in_size + z_size).
        z (optional): The input condition tensor (or numpy array) with shape (batch_size, z_size)
            If not provided, it will be split from the input state tensor.
        ---
        outputs: A tensor of shape (batch_size, out_size), representing the network's output.
        """
        if z is None:
            state, z = torch.Tensor(state).to(self.device).split([self.in_size, self.z_size], -1)
        else:
            state = torch.Tensor(state).to(self.device)
            z = torch.Tensor(z).to(self.device)
            z = z.expand(state.shape[0], -1) if len(state.shape) > 1 else z
        
        if len(state.shape) == 1:
            x = state.unsqueeze(0)
            z = z.unsqueeze(0)
        else:
            x = state

        for i in range(self.hidden_depth + 1):
            if i in self.film_pos:
                film_output = self.film_gen_nets[self.film_pos.index(i)](z)
                if self.no_bias:
                    scale = film_output
                    x = scale * x
                else:
                    scale, shift = film_output.split([self.hidden_size, self.hidden_size], -1)
                    x = scale * x + shift

            x = self.primary_layers[i](x)
            if i < self.hidden_depth:
                x = F.relu(x)
        return x