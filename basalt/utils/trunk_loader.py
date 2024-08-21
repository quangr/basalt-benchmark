import os
import glob
import json
import cv2
import numpy as np
import multiprocessing as mp
from queue import Empty
from basalt.vpt_lib.agent import resize_image, AGENT_RESOLUTION
from torch.utils.data import Dataset,DataLoader
import torch
QUEUE_TIMEOUT = 10


MINEREC_ORIGINAL_HEIGHT_PX = 720

# If GUI is open, mouse dx/dy need also be adjusted with these scalers.
# If data version is not present, assume it is 1.
MINEREC_VERSION_SPECIFIC_SCALERS = {
    "5.7": 0.5,
    "5.8": 0.5,
    "6.7": 2.0,
    "6.8": 2.0,
    "6.9": 2.0,
}


def resize_image(img, size):
    return cv2.resize(img, size, interpolation=cv2.INTER_LINEAR)


def is_json_action_null(action):
    # Implement your logic to determine if an action is null
    pass


SEQUENCE_LENGTH = 64


def data_loader_worker(tasks_queue, output_queue, quit_workers_event):
    while True:
        try:
            task = tasks_queue.get(timeout=QUEUE_TIMEOUT)
        except Empty:
            if quit_workers_event.is_set():
                break
            continue

        if task is None:
            break

        trajectory_id, video_path, json_path = task
        video = cv2.VideoCapture(video_path)

        if not os.path.isfile(json_path):
            print("Json file not found, skipping this sample", json_path)
            continue

        try:
            with open(json_path) as json_file:
                json_lines = json_file.readlines()
                json_data = "[" + ",".join(json_lines) + "]"
                json_data = json.loads(json_data)
        except Exception as e:
            print("Error loading json file", json_path, e)
            continue

        sequence = []
        mask = []  # Add a mask to track valid frames
        for i in range(len(json_data)):
            if quit_workers_event.is_set():
                break

            action = json_data[i]
            is_null_action = is_json_action_null(action)
            action["is_null_action"] = is_null_action
            action["camera"] = np.array(action["camera"])

            ret, frame = video.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = np.clip(frame, 0, 255).astype(np.uint8)
                frame = resize_image(frame, AGENT_RESOLUTION)
                sequence.append((frame, action))
                mask.append(True)  # Mark the frame as valid

                if len(sequence) == SEQUENCE_LENGTH:
                    output_queue.put(
                        (trajectory_id, sequence, mask), timeout=QUEUE_TIMEOUT
                    )
                    sequence = []
                    mask = []
            else:
                print(f"Could not read frame from video {video_path}")
                break

        # Output any remaining frames (less than SEQUENCE_LENGTH) with padding
        if sequence:
            # Pad the sequence and mask to SEQUENCE_LENGTH
            while len(sequence) < SEQUENCE_LENGTH:
                sequence.append((np.zeros_like(sequence[0][0]), sequence[0][1]))
                mask.append(False)  # Mark padded frames as invalid
            output_queue.put((trajectory_id, sequence, mask), timeout=QUEUE_TIMEOUT)

        video.release()
        output_queue.put((trajectory_id, None, None), timeout=QUEUE_TIMEOUT)

        if quit_workers_event.is_set():
            break

    output_queue.put(None)


class MinecraftDataLoader:
    def __init__(self, dataset_dir, batch_size=1, contrast=False):
        self.dataset_dir = dataset_dir
        self.batch_size = batch_size
        self.contrast = contrast
        self.num_workers = batch_size  # Set number of workers equal to batch size
        self.unique_ids = self._get_unique_ids()
        self.demonstration_tuples = self._create_demonstration_tuples()

        self.tasks_queue = mp.Queue()
        self.output_queue = mp.Queue()
        self.quit_workers_event = mp.Event()

        self.workers = []
        self._start_workers()

    def _get_unique_ids(self):
        unique_ids = glob.glob(os.path.join(self.dataset_dir, "*.mp4"))
        return_list = list(
            set([(True, os.path.basename(x).split(".")[0]) for x in unique_ids])
        )
        if self.contrast:
            fail_unique_ids = glob.glob(
                os.path.join(self.dataset_dir + "-fail", "*.mp4")
            )
            return return_list + list(
                set(
                    [
                        (False, os.path.basename(x).split(".")[0])
                        for x in fail_unique_ids
                    ]
                )
            )
        else:
            return return_list

    def _create_demonstration_tuples(self):
        demonstration_tuples = []
        for item in self.unique_ids:
            label, unique_id = item
            if label:
                video_path = os.path.abspath(
                    os.path.join(self.dataset_dir, unique_id + ".mp4")
                )
                json_path = os.path.abspath(
                    os.path.join(self.dataset_dir, unique_id + ".jsonl")
                )
            else:
                video_path = os.path.abspath(
                    os.path.join(self.dataset_dir + "-fail", unique_id + ".mp4")
                )
                json_path = os.path.abspath(
                    os.path.join(self.dataset_dir + "-fail", unique_id + ".jsonl")
                )
            demonstration_tuples.append((video_path, json_path))
        return demonstration_tuples


    def _start_workers(self):
        for _ in range(self.num_workers):
            worker = mp.Process(
                target=data_loader_worker,
                args=(self.tasks_queue, self.output_queue, self.quit_workers_event),
            )
            worker.start()
            self.workers.append(worker)

    def __iter__(self):
        for i in range(0, len(self.demonstration_tuples), self.batch_size):
            batch = self.demonstration_tuples[i : i + self.batch_size]

            # Queue tasks for the current batch
            for j, (video_path, json_path) in enumerate(batch):
                self.tasks_queue.put((i + j, video_path, json_path))

            # Process the entire batch before moving to the next one
            active_trajectories = set(range(i, i + len(batch)))
            batch_data = {j: [] for j in active_trajectories}

            while active_trajectories:
                try:
                    item = self.output_queue.get(timeout=QUEUE_TIMEOUT)
                    if item is None:
                        break
                    trajectory_id, sequence, mask = item
                    if sequence is None:
                        active_trajectories.remove(trajectory_id)
                    else:
                        batch_data[trajectory_id].append((sequence, mask))
                        # Yield complete sequences
                        yield (
                            trajectory_id,
                            batch_data[trajectory_id][0][0],  # sequence
                            batch_data[trajectory_id][0][1],  # mask
                            self.unique_ids[trajectory_id][0],  # label
                        )
                        batch_data[trajectory_id] = batch_data[trajectory_id][1:]
                except Empty:
                    if self.quit_workers_event.is_set():
                        return
            # Yield any remaining data in the batch
            for trajectory_id, data in batch_data.items():
                if data:
                    yield trajectory_id, data, self.unique_ids[trajectory_id][0]

        # Signal workers to finish
        for _ in range(self.num_workers):
            self.tasks_queue.put(None)

        # Collect any remaining output
        finished_workers = 0
        while finished_workers < self.num_workers:
            try:
                item = self.output_queue.get(timeout=QUEUE_TIMEOUT)
                if item is None:
                    finished_workers += 1
            except Empty:
                if self.quit_workers_event.is_set():
                    break

    def __len__(self):
        return len(self.demonstration_tuples)

    def close(self):
        self.quit_workers_event.set()
        for worker in self.workers:
            worker.join()

class MinecraftDataset(Dataset):
    def __init__(self, dataset_dir, contrast=False):
        self.dataset_dir = dataset_dir
        self.contrast = contrast
        self.demonstration_tuples = self._create_demonstration_tuples()

    def _get_unique_ids(self):
        unique_ids = glob.glob(os.path.join(self.dataset_dir, "*.mp4"))
        return_list = list(
            set([(True, os.path.basename(x).split(".")[0]) for x in unique_ids])
        )
        if self.contrast:
            fail_unique_ids = glob.glob(
                os.path.join(self.dataset_dir + "-fail", "*.mp4")
            )
            return return_list + list(
                set(
                    [
                        (False, os.path.basename(x).split(".")[0])
                        for x in fail_unique_ids
                    ]
                )
            )
        else:
            return return_list

    def _create_demonstration_tuples(self):
        demonstration_tuples = []
        for item in self._get_unique_ids():
            label, unique_id = item
            if label:
                video_path = os.path.abspath(
                    os.path.join(self.dataset_dir, unique_id + ".mp4")
                )
                json_path = os.path.abspath(
                    os.path.join(self.dataset_dir, unique_id + ".jsonl")
                )
            else:
                video_path = os.path.abspath(
                    os.path.join(self.dataset_dir + "-fail", unique_id + ".mp4")
                )
                json_path = os.path.abspath(
                    os.path.join(self.dataset_dir + "-fail", unique_id + ".jsonl")
                )
            demonstration_tuples.append((video_path, json_path))
        return demonstration_tuples

    def __len__(self):
        return len(self.demonstration_tuples)

    def __getitem__(self, idx):
        video_path, json_path = self.demonstration_tuples[idx]
        video = cv2.VideoCapture(video_path)

        if not os.path.isfile(json_path):
            raise FileNotFoundError(f"Json file not found: {json_path}")

        with open(json_path) as json_file:
            json_lines = json_file.readlines()
            json_data = "[" + ",".join(json_lines) + "]"
            json_data = json.loads(json_data)

        frames = []
        actions = []

        for i, action in enumerate(json_data):
            ret, frame = video.read()
            if not ret:
                break  # Exit if a frame couldn't be read

            action["is_null_action"] = is_json_action_null(action)
            action["camera"] = np.array(action["camera"])

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = np.clip(frame, 0, 255).astype(np.uint8)
            frame = resize_image(frame, AGENT_RESOLUTION)

            frames.append(frame)
            actions.append(action)

        video.release()

        if not frames:
            raise ValueError(f"No frames were found in video: {video_path}")

        return torch.tensor(frames), actions

def collate_fn(batch, chunk_size=64):
    all_chunks = []
    all_masks = []

    for frames, actions in batch:
        video_length = frames.shape[0]
        chunks = []
        masks = []

        for i in range(0, video_length, chunk_size):
            chunk = frames[i:i + chunk_size]
            mask = torch.ones(chunk.shape[0], dtype=torch.bool)

            # Padding if the chunk is smaller than chunk_size
            if chunk.shape[0] < chunk_size:
                padding = chunk_size - chunk.shape[0]
                chunk = torch.cat([chunk, torch.zeros((padding,) + chunk.shape[1:], dtype=chunk.dtype)], dim=0)
                mask = torch.cat([mask, torch.zeros(padding, dtype=torch.bool)], dim=0)

            chunks.append(chunk)
            masks.append(mask)

        all_chunks.extend(chunks)
        all_masks.extend(masks)

    # Stack all the chunks and masks for the batch
    return torch.stack(all_chunks), torch.stack(all_masks)


def data_generator(dataset, batch_size=2, step_size=64):
    data_loader = iter(DataLoader(dataset, batch_size=1, shuffle=True, num_workers=4))
    index = [0] * batch_size
    batch_data = [next(data_loader), next(data_loader)]

    while True:
        return_data = []
        mask = np.zeros((batch_size, step_size), dtype=np.int32)

        for i in range(batch_size):
            # Determine the valid range for the current batch
            start_idx = index[i]
            end_idx = min(start_idx + step_size, len(batch_data[i][0]))

            # Slice the data according to the current index and step_size
            return_data.append(batch_data[i][0][start_idx:end_idx])

            # Update the mask to reflect valid data
            mask[i, :end_idx - start_idx] = 1

            # Update the index for the next iteration
            index[i] += step_size

            # Check if we need to load new data for this batch
            if index[i] >= len(batch_data[i][0]):
                try:
                    batch_data[i] = next(data_loader)
                    index[i] = 0  # Reset index for the new data
                except StopIteration:
                    # Handle end of the dataset if needed
                    break

        # Pad return_data to have consistent shapes in each batch
        for i in range(batch_size):
            if len(return_data[i]) < step_size:
                pad_width = ((0, step_size - len(return_data[i])),) + tuple((0, 0) for _ in range(len(return_data[i].shape) - 1))
                return_data[i] = np.pad(return_data[i], pad_width, mode='constant', constant_values=0)

        # Yield the batch data along with the mask
        yield np.stack(return_data), mask

if __name__ == "__main__":
    dataset_dir = "pipeline_test_data/demonstrations/MineRLBasaltMakeWaterfall-v0"  # Replace with your actual dataset path
    # batch_size = 1  # This will also be the number of workers
    # dataloader = MinecraftDataLoader(dataset_dir, batch_size=batch_size)
    # for trajectory_id, sequence, label in dataloader:
    #     pass
    batch_size = 2  # Set the desired batch size
    data_generator()
    dataset = MinecraftDataset(dataset_dir=dataset_dir, contrast=False)
    step_size = 64
    gen = data_generator(dataset, batch_size=batch_size, step_size=step_size)

    # Fetch and print a few batches
    for _ in range(3):  # Fetch 3 batches for testing
        batch_data, mask = next(gen)
        
        print("Batch Data Shape:", batch_data.shape)
        print("Batch Data:\n", batch_data)
        print("Mask Shape:", mask.shape)
        print("Mask:\n", mask)
        print("-" * 50)