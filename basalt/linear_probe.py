from basalt.config import DefaultDataConfig
import os
import torch
import numpy as np
from dataclasses import dataclass
import tyro
from basalt.common import load_model_parameters
from basalt.vpt_lib.agent import MineRLAgent
from torch.utils.data import DataLoader
import time
import torch_xla.core.xla_model as xm
from basalt.vpt_lib.agent import resize_image, AGENT_RESOLUTION

device = xm.xla_device()
import os
import cv2
import torch
import numpy as np
from torch.utils.data import DataLoader
from concurrent.futures import ProcessPoolExecutor, as_completed
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
from sklearn.model_selection import train_test_split
import torch_xla


class LinearProbe(nn.Module):
    def __init__(self, input_size):
        super(LinearProbe, self).__init__()
        self.linear = nn.Linear(input_size, 1)

    def forward(self, x):
        return self.linear(x)

    def load_parameters(self, weight_path):
        self.load_state_dict(torch.load(weight_path))

    def save_parameters(self, weight_path):
        torch.save(self.state_dict(), weight_path)


def calculate_l2_norm_penalty(model):
    l2_norm = 0.0

    for param in model.parameters():
        l2_norm += torch.sum(param**2)

    return l2_norm


def extract_frames(video_path):
    """
    Extract frames at intervals from a video.
    """
    video = cv2.VideoCapture(video_path)
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    labels = []

    timesteps = set(range(0, total_frames, 50))
    timesteps.update([total_frames - 1, total_frames - 10, total_frames - 20])

    for timestep in timesteps:
        video.set(cv2.CAP_PROP_POS_FRAMES, timestep)
        success, frame = video.read()

        if success:
            cv2.cvtColor(frame, code=cv2.COLOR_BGR2RGB, dst=frame)
            frame = np.asarray(np.clip(frame, 0, 255), dtype=np.uint8)
            frame = resize_image(frame, AGENT_RESOLUTION)
            frames.append(frame)
            labels.append(timestep / total_frames)
        else:
            print(timestep, video_path)
    video.release()
    return frames, labels


def process_videos(video_dir):
    """
    Process all videos in the directory using multiprocessing.
    """
    video_files = [f for f in os.listdir(video_dir) if f.endswith(".mp4")]
    all_frames = []
    all_labels = []

    with ProcessPoolExecutor() as executor:
        futures = [
            executor.submit(extract_frames, os.path.join(video_dir, video_file))
            for video_file in video_files
        ]
        for future in as_completed(futures):
            frames, labels = future.result()
            all_frames.extend(frames)
            all_labels.extend(labels)

    return all_frames, all_labels


@dataclass
class Args:
    data: DefaultDataConfig
    task_name: str = "MineRLBasaltFindCave-v0/"


def train(args: Args):
    agent_policy_kwargs, agent_pi_head_kwargs = load_model_parameters(
        args.data.model_path
    )

    agent = MineRLAgent(
        device=device,
        policy_kwargs=agent_policy_kwargs,
        pi_head_kwargs=agent_pi_head_kwargs,
    )
    agent.load_weights(args.data.weights_path)
    policy = agent.policy
    policy.eval()

    net = policy.net
    video_dir = f"{args.data.task_data_prefix}/{args.task_name}"

    frames, labels = process_videos(video_dir)
    print(len(frames))
    batch_size = 128

    with torch.no_grad():
        outputs = []
        for batch in DataLoader(frames, batch_size=batch_size):
            with torch_xla.step():
                output = net.img_preprocess(batch[:, None].to(device))
                output = net.img_process(output).squeeze()
            outputs.append(output)
        outputs = torch.cat(outputs, dim=0)

    labels = (torch.tensor(labels) > 0.9).float()

    X_train, X_val, y_train, y_val = train_test_split(
        outputs, labels, test_size=0.2, random_state=42
    )

    train_dataset = TensorDataset(X_train.to(device), y_train.to(device))
    val_dataset = TensorDataset(X_val.to(device), y_val.to(device))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    linear_probe = LinearProbe(net.hidsize).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(linear_probe.parameters(), lr=0.001)
    num_epochs = 20

    def train_step_fn(linear_probe, batch_features, batch_labels, optimizer):
        optimizer.zero_grad()
        outputs = linear_probe(batch_features.to(device))
        loss = (
            criterion(outputs, batch_labels[:, None])
            + calculate_l2_norm_penalty(linear_probe) * 0.01
        )
        loss.backward()
        optimizer.step()
        return loss

    def val_step_fn(linear_probe, batch_features, batch_labels):
        outputs = linear_probe(batch_features.to(device))
        val_loss = criterion(outputs, batch_labels[:, None]).item()
        return outputs, val_loss

    for epoch in range(num_epochs):
        linear_probe.train()
        start_time = time.time()
        for batch_features, batch_labels in train_loader:
            with torch_xla.step():
                loss = train_step_fn(
                    linear_probe, batch_features, batch_labels, optimizer
                )
        linear_probe.eval()
        val_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_features, batch_labels in val_loader:
                with torch_xla.step():
                    outputs, _val_loss = val_step_fn(
                        linear_probe, batch_features, batch_labels
                    )
                val_loss += _val_loss
                predicted = outputs > 0.0
                total += batch_labels.size(0)
                correct += (predicted.squeeze() == batch_labels.squeeze()).sum()

        if (epoch + 1) % 10 == 0:
            save_path = (
                f"checkpoints/linear_probe/{args.task_name}/epoch_{epoch + 1}.pt"
            )
            directory = os.path.dirname(save_path)
            if not os.path.exists(directory):
                os.makedirs(directory)
            linear_probe.save_parameters(save_path)
            print(f"Saved chekcpoints at {save_path}")

        print(
            f"Epoch {epoch+1}/{num_epochs}",
            f"Train Loss: {loss:.4f}",
            f"Val Loss: {val_loss/len(val_loader):.4f}",
            f"Val Accuracy: {100.*correct/total:.2f}%",
        )

        end_time = time.time()
        elapsed_time = end_time - start_time
        print("FINISH, elapsed_time:", elapsed_time)


if __name__ == "__main__":
    args = tyro.cli(Args)
    train(args)
