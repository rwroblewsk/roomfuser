# Copyright 2020 LMNT, Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import matplotlib.pyplot as plt
import os
#from roomfuser.dataset.roomfuser_dataset import RirDataset
import torch
import numpy as np
import torch.nn as nn
import auraloss

from torch.nn.parallel import DistributedDataParallel
from torch.utils.tensorboard import SummaryWriter
from torchinfo import summary
from tqdm import tqdm

from roomfuser.dataset import from_path, RandomSinusoidDataset, FastRirDataset, FastRirDeEnvDataset, RirDataset
from roomfuser.models import load_model

from roomfuser.inference import predict_batch
from torch.utils.data import DataLoader

class Trainer:
    def __init__(self, model_dir, model, dataset, optimizer, params, *args, **kwargs):
        os.makedirs(model_dir, exist_ok=True)
        self.model_dir = model_dir
        self.model = model
        self.dataset = dataset
        self.optimizer = optimizer
        self.params = params
        self.autocast = torch.amp.autocast('cuda',enabled=kwargs.get("fp16", False))
        self.scaler = torch.amp.GradScaler(enabled=kwargs.get("fp16", False))
        self.is_master = True
        self.ema = None  # Add EMA support

        self.noise_scheduler = model.noise_scheduler
        self.loss_fn = Loss(params.rir_len, model.device, params.loss_weight, params.loss)
        self.summary_writer = None

    def state_dict(self):
        if hasattr(self.model, "module") and isinstance(self.model.module, nn.Module):
            model_state = self.model.module.state_dict()
        else:
            model_state = self.model.state_dict()
        return {
            "model": {
                k: v.cpu() if isinstance(v, torch.Tensor) else v
                for k, v in model_state.items()
            },
            "optimizer": {
                k: v.cpu() if isinstance(v, torch.Tensor) else v
                for k, v in self.optimizer.state_dict().items()
            },
            "params": dict(self.params),
            "scaler": self.scaler.state_dict(),
            "ema": self.ema.state_dict() if self.ema else None,
        }

    def load_state_dict(self, state_dict):
        if hasattr(self.model, "module") and isinstance(self.model.module, nn.Module):
            self.model.module.load_state_dict(state_dict["model"])#, strict=False)
        else:
            self.model.load_state_dict(state_dict["model"])

        # Restore EMA state if available
        if "ema" in state_dict and state_dict["ema"] is not None and self.ema is not None:
            self.ema.load_state_dict(state_dict["ema"])
            # Move EMA weights to the same device as the model
            device = next(self.model.parameters()).device
            self.ema.to_device(device)

    def save_to_checkpoint(self, n_epoch, filename="weights"):
        save_basename = f"{filename}-{n_epoch}.pt"
        save_name = f"{self.model_dir}/{save_basename}"
        torch.save(self.state_dict(), save_name)
        print(f"Saved best model at {save_name}")

    def restore_from_checkpoint(self):
        model_path = self.params.model_path
        try:
            checkpoint = torch.load(f"{model_path}", map_location="cpu")
            self.load_state_dict(checkpoint)
            print(f"Restored checkpoint from {model_path}")
            return True
        except FileNotFoundError:
            return False

    def train(self):
        device = next(self.model.parameters()).device
        best_loss = float("inf")
        for n_epoch in range(self.params.n_epochs):
            progress_bar = tqdm(total=len(self.dataset))
            progress_bar.set_description(f"Epoch: {n_epoch}")
            epoch_loss = 0.0 # Moving average
            for n_batch, batch in enumerate(self.dataset):
                batch = _nested_map(
                    batch,
                    lambda x: x.to(torch.float32).to(device) if isinstance(x, torch.Tensor) else x,
                )
                loss = self.train_step(batch)
                if torch.isnan(loss).any():
                    raise RuntimeError(f"Detected NaN loss at epoch {n_epoch}.")
                progress_bar.update(1)
                epoch_loss = (epoch_loss*n_batch + loss.item())/(n_batch + 1)
                progress_bar.set_postfix(loss=epoch_loss)#**(1/2))
            
                # Update EMA if available
                if self.ema is not None:
                    self.ema.update()

            progress_bar.close()

            if self.is_master:
                if n_epoch % self.params.n_log_epochs == 0:
                    
                    # Use EMA model for visualization if available
                    if self.ema is not None:
                        self.ema.apply_shadow()
                    
                    # self._write_summary(n_epoch, batch, loss)
                    self._log_output_viz(
                        self.model,
                        self.params.n_viz_samples,
                        self.params.rir_len,
                        n_epoch,
                        self.model_dir,
                    )

                    # Restore model weights after visualization
                    if self.ema is not None:
                        self.ema.restore()
            # Save the model if it's the best one so far
            if self.is_master and epoch_loss < best_loss:
                best_loss = epoch_loss
                self.save_to_checkpoint(n_epoch)
            
    def train_step(self, batch):
        for param in self.model.parameters():
            param.grad = None

        audio = batch["rir"]
        conditioner = batch["conditioner"]
        labels = batch["labels"]

        audio_shape = audio.shape
        if len(audio_shape) == 2:
            batch_size, T = audio.shape
        else:
            # Frequency response
            batch_size, _, T = audio.shape

        device = audio.device
        self.noise_scheduler.noise_level = self.noise_scheduler.noise_level.to(device)

        with self.autocast:
            
            # 1. Assign a timestep to each sample in the batch.
            t = torch.randint(
                0, len(self.noise_scheduler.beta), [batch_size], device=audio.device
            )
            
            # # 2. Get the corresponding noise for each sample, and add it to the audio.
            noisy_audio, noise = self.noise_scheduler.add_noise_to_audio(
                audio, labels, t,
            )

            # 3. Compute the score (gradient of the likelihood) for the noisy audio.
            predicted = self.model(noisy_audio, t, conditioner)

            # 4. Compute the loss.
            loss = self.loss_fn(noise, predicted)

        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.optimizer)
        self.grad_norm = nn.utils.clip_grad_norm_(
            self.model.parameters(), self.params.max_grad_norm or 1e9
        )
        self.scaler.step(self.optimizer)
        self.scaler.update()
        return loss

    def _write_summary(self, step, batch, loss):
        writer = self.summary_writer or SummaryWriter(self.model_dir, purge_step=step)
        writer.add_audio(
            "feature/audio",
            batch["rir"][0],
            step,
            sample_rate=self.params.sample_rate,
        )
        writer.add_scalar("train/loss", loss, step)
        writer.add_scalar("train/grad_norm", self.grad_norm, step)
        writer.flush()
        self.summary_writer = writer

    def _log_output_viz(self, model, n_viz_samples, n_sample, epoch, outputs_dir):
        fig, axs = plt.subplots(nrows=n_viz_samples, ncols=1, figsize=(5, 15))

        scaler = None
        if self.params.dataset_name == "sinusoid": # Debug task
            dataset = RandomSinusoidDataset(n_sample, n_viz_samples)
            target_samples = [
                dataset[i] for i in range(n_viz_samples)
            ]
            conditioner = torch.stack(
                [target_sample["conditioner"] for target_sample in target_samples]
            ).to(model.device)
            audio = torch.stack(
                [target_sample["audio"] for target_sample in target_samples]
            ).to(model.device)
            labels = [target_sample["labels"] for target_sample in target_samples]
        elif self.params.dataset_name == "fast_rir_de_env":
            dataset = FastRirDeEnvDataset(self.params.fast_rir_dataset_path,
            n_rir=self.params.rir_len,
            trim_direct_path=self.params.trim_direct_path,
            scaler_path=self.params.fast_rir_scaler_path,
            frequency_response=self.params.frequency_response)

            target_samples = [
                dataset[i] for i in range(n_viz_samples)
            ]
            conditioner = torch.stack(
                [target_sample["conditioner"] for target_sample in target_samples]
            ).to(model.device)
            audio = torch.stack(
                [target_sample["rir"] for target_sample in target_samples]
            ).to(model.device)
            labels = [target_sample["labels"] for target_sample in target_samples]
        elif self.params.dataset_name in ["roomfuser", "fast_rir"]:
            if self.params.dataset_name == "roomfuser":
                dataset = RirDataset(self.params.roomfuser_dataset_path, n_sample,
                                     trim_direct_path=self.params.trim_direct_path,
                                     scaler_path=self.params.roomfuser_scaler_path,
                                     frequency_response=self.params.frequency_response)
                scaler = dataset.scaler
            elif self.params.dataset_name == "fast_rir":
                dataset = FastRirDataset(self.params.fast_rir_dataset_path, n_sample,
                                         trim_direct_path=self.params.trim_direct_path,
                                         scaler_path=self.params.fast_rir_scaler_path,
                                         frequency_response=self.params.frequency_response)
                scaler = dataset.scaler

            target_samples = [
                dataset[i] for i in range(n_viz_samples)
            ]
            conditioner = torch.stack(
                [target_sample["conditioner"] for target_sample in target_samples]
            ).to(model.device)
            print("Conditioner shape: ", conditioner.shape)
            audio = torch.stack(
                [target_sample["rir"] for target_sample in target_samples]
            ).to(model.device)
            labels = [target_sample["labels"] for target_sample in target_samples]
        
        outputs = predict_batch(
            model, conditioner, n_viz_samples,
            labels=labels, frequency_response=self.params.frequency_response,
            scaler=dataset.scaler
        )[0]
        
        for i in range(n_viz_samples):
            if scaler is not None:
                audio[i] = scaler.descale(audio[i])

            if self.params.frequency_response:
                # In this case, this model is using the frequency response: apply the IDFT to get the audio back
                target = torch.complex(audio[i, 0], audio[i, 1])
                target = torch.fft.irfft(target)
            else:
                target = audio[i]

            target = target / torch.max(torch.abs(target))

            axs[i].plot(target.cpu().detach().numpy(), label="Target", alpha=0.5)
            axs[i].plot(outputs[i].cpu().detach().numpy(), label="Predicted", alpha=0.5)
            axs[i].legend()

        # Save the images
        os.makedirs(outputs_dir, exist_ok=True)
        plt.tight_layout()
        plt.savefig(f"{outputs_dir}/{epoch:04d}.png")
        plt.close()


def _train_impl(replica_id, model, dataset, args, params):
    torch.backends.cudnn.benchmark = True
    if replica_id == 0:  # Only print from the main process
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        # print(f"\n=== MODEL PARAMETERS ===")
        # print(f"Trainable parameters: {trainable_params:,}")
        # print(f"Total parameters: {total_params:,}")
        # print("======================\n")
    
    opt = torch.optim.Adam(model.parameters(), lr=params.learning_rate)

    #opt = torch.optim.Adam(
    #model.parameters(),
    #lr=params.learning_rate,
    #betas=(0.9, 0.999),
    #eps=1e-8,
    #weight_decay=0)
    
    # Add EMA for better inference quality
    ema = EMA(model, decay=0.9999)

    learner = Trainer(
        args.model_dir, model, dataset, opt, params, fp16=args.fp16
    )
    learner.ema = ema
    learner.is_master = replica_id == 0
    learner.restore_from_checkpoint()
    learner.train()


def _nested_map(struct, map_fn):
    if isinstance(struct, tuple):
        return tuple(_nested_map(x, map_fn) for x in struct)
    if isinstance(struct, list):
        return [_nested_map(x, map_fn) for x in struct]
    if isinstance(struct, dict):
        return {k: _nested_map(v, map_fn) for k, v in struct.items()}
    return map_fn(struct)


def train(args, params):
    dataset = from_path(params)

    model = load_model(params)

    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    model = model.to(device)
    model.device = device
    print("DEVICE: ", device)

    _train_impl(0, model, dataset, args, params)


def train_distributed(replica_id, replica_count, port, args, params):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(port)
    torch.distributed.init_process_group(
        "nccl", rank=replica_id, world_size=replica_count
    )

    dataset = from_path(params, is_distributed=False)
    device = torch.device("cuda", replica_id)
    torch.cuda.set_device(device)
    model = load_model(params).to(device)
    noise_scheduler = model.noise_scheduler
    model = DistributedDataParallel(model, device_ids=[replica_id], find_unused_parameters=False)
    model.params = params
    model.noise_scheduler = noise_scheduler
    _train_impl(replica_id, model, dataset, args, params)


class Loss(nn.Module):
    def __init__(self, rir_len, device, loss_weight=None, loss=None):
        super().__init__()
        self.device = device
    
        if loss_weight == "linear":
            loss_weight = torch.linspace(1, 0, rir_len, device=device)
        elif loss_weight == "log":
            loss_weight = torch.linspace(rir_len, 1, rir_len, device=device)
            loss_weight = torch.log(loss_weight)
        elif loss_weight == "lin_reverse":
            loss_weight = torch.linspace(0, 1, rir_len-800, device=device)
        
        # Normalize the loss weight
        if loss_weight is not None:
            loss_weight = loss_weight/loss_weight.sum()

        self.loss_weight = loss_weight

        if loss == "l1":
            self.loss_fn = nn.L1Loss(reduction="none")
        elif loss == "huber":
            self.loss_fn = nn.SmoothL1Loss(reduction="none")
        elif loss == "multi_spect":
            self.loss_fn = auraloss.freq.MultiResolutionSTFTLoss()
        else:
            self.loss_fn = nn.MSELoss(reduction="none")

    def forward(self, noise, predicted):
        batch_size = noise.shape[0]

        loss = self.loss_fn(noise, predicted.squeeze(1))

        if self.loss_weight is not None:
            loss = (loss*self.loss_weight).sum()
            loss = loss / batch_size
        else:
            loss = loss.mean()

        return loss

class EMA:
    def __init__(self, model, decay=0.9999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        
        # Register model parameters
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
    
    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = self.decay * self.shadow[name] + (1.0 - self.decay) * param.data
                self.shadow[name] = new_average.clone()
    
    def apply_shadow(self):
        """Use the EMA parameters for inference"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.backup[name] = param.data
                param.data = self.shadow[name]
    
    def restore(self):
        """Restore the original parameters"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}
    
    def state_dict(self):
        return {
            "decay": self.decay,
            "shadow": self.shadow,
            "backup": self.backup
        }
        
    def load_state_dict(self, state_dict):
        self.decay = state_dict["decay"]
        self.shadow = state_dict["shadow"]
        self.backup = state_dict["backup"]

    def to_device(self, device):
        """Move EMA shadow weights to the specified device"""
        for name in self.shadow:
            self.shadow[name] = self.shadow[name].to(device)
        for name in self.backup:
            self.backup[name] = self.backup[name].to(device)
