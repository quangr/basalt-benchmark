import os
import glob
import json
import cv2
import numpy as np
import multiprocessing as mp
from queue import Empty
from basalt.vpt_lib.agent import resize_image, AGENT_RESOLUTION
from torch.utils.data import Dataset, DataLoader
import torch
from basalt.vpt_lib.tree_util import tree_map
from basalt.utils.minerl_data_loader import (
    CURSOR_FILE,
    composite_images_with_alpha,
    json_action_to_env_action,
)

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


def is_json_action_null(json_action):
    """
    Determines if a JSON action is null based on the provided logic.
    """
    # Check keyboard keys
    for key in json_action:
        if key != "camera" and json_action[key] != 0:
            return False

    # Check camera movement
    camera_action = json_action["camera"]
    if camera_action[0] != 0 or camera_action[1] != 0:
        return False

    return True


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

def get_mp4_tuple(uid, dataset_dir):
    basename = os.path.basename(uid).split(".")[0]
    video_path = os.path.abspath(uid)
    json_path = os.path.abspath(os.path.join(dataset_dir, basename + ".jsonl"))
    return basename, video_path, json_path

class MinecraftDataset(Dataset):
    def __init__(self, dataset_dict, unique_ids=None):
        self.dataset_dict = dataset_dict
        if unique_ids is None:
            self.unique_ids = self._get_unique_ids()
        else:
            self.unique_ids = unique_ids
        self.demonstration_tuples = self._create_demonstration_tuples()


    def _get_unique_ids(self):
        return_list = []
        for dataset_dir, label in self.dataset_dict.items():
            unique_ids = glob.glob(os.path.join(dataset_dir, "*.mp4"))
            for uid in unique_ids:
                basename, video_path, json_path = get_mp4_tuple(uid, dataset_dir)
                return_list.append((label, basename, video_path, json_path))
        return return_list

    def _create_demonstration_tuples(self):
        demonstration_tuples = []
        for item in self.unique_ids:
            label, unique_id, video_path, json_path = item
            demonstration_tuples.append((label, video_path, json_path))
        return demonstration_tuples

    def __len__(self):
        return len(self.demonstration_tuples)

    def __getitem__(self, idx):
        label, video_path, json_path = self.demonstration_tuples[idx]
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
        return (
            np.array(frames),
            {key: np.array([a[key] for a in actions]) for key in actions[0].keys()},
            label,
            idx,
            video_path,
        )


class BasaltMinecraftDataset(MinecraftDataset):
    def __getitem__(self, idx):
        cursor_image = cv2.imread(CURSOR_FILE, cv2.IMREAD_UNCHANGED)
        # Assume 16x16
        cursor_image = cursor_image[:16, :16, :]
        cursor_alpha = cursor_image[:, :, 3:] / 255.0
        cursor_image = cursor_image[:, :, :3]

        label, video_path, json_path = self.demonstration_tuples[idx]
        video = cv2.VideoCapture(video_path)

        if not os.path.isfile(json_path):
            raise FileNotFoundError(f"Json file not found: {json_path}")

        with open(json_path) as json_file:
            json_lines = json_file.readlines()
            json_data = "[" + ",".join(json_lines) + "]"
            json_data = json.loads(json_data)

        frames = []
        actions = []

        attack_is_stuck = False
        last_hotbar = 0

        for i, step_data in enumerate(json_data):
            if i == 0:
                # Check if attack will be stuck down
                if step_data["mouse"]["newButtons"] == [0]:
                    attack_is_stuck = True
            elif attack_is_stuck:
                # Check if we press attack down, then it might not be stuck
                if 0 in step_data["mouse"]["newButtons"]:
                    attack_is_stuck = False
            # If still stuck, remove the action
            if attack_is_stuck:
                step_data["mouse"]["buttons"] = [
                    button for button in step_data["mouse"]["buttons"] if button != 0
                ]

            action, is_null_action = json_action_to_env_action(step_data)
            # Add is_null info to the action buffer
            action["is_null_action"] = is_null_action

            # Update hotbar selection
            current_hotbar = step_data["hotbar"]
            if current_hotbar != last_hotbar:
                action["hotbar.{}".format(current_hotbar + 1)] = 1
            last_hotbar = current_hotbar

            # Read frame even if this is null so we progress forward
            ret, frame = video.read()
            if ret:
                if step_data["isGuiOpen"]:
                    camera_scaling_factor = frame.shape[0] / MINEREC_ORIGINAL_HEIGHT_PX
                    cursor_x = int(step_data["mouse"]["x"] * camera_scaling_factor)
                    cursor_y = int(step_data["mouse"]["y"] * camera_scaling_factor)
                    try:
                        composite_images_with_alpha(
                            frame, cursor_image, cursor_alpha, cursor_x, cursor_y
                        )
                    except Exception as e:
                        print("Error composite images", e)
                        continue

                cv2.cvtColor(frame, code=cv2.COLOR_BGR2RGB, dst=frame)
                frame = np.asarray(np.clip(frame, 0, 255), dtype=np.uint8)
                frame = resize_image(frame, AGENT_RESOLUTION)
            else:
                break  # Exit if a frame couldn't be read
            frames.append(frame)
            actions.append(action)

        video.release()

        if not frames:
            raise ValueError(f"No frames were found in video: {video_path}")
        return (
            np.array(frames),
            {key: np.array([a[key] for a in actions]) for key in actions[0].keys()},
            label,
            idx,
            video_path,
        )


def pad_array(array, step_size):
    pad_width = ((0, step_size - len(array)),) + tuple(
        (0, 0) for _ in range(len(array.shape) - 1)
    )
    return np.pad(array, pad_width, mode="constant", constant_values=0)


def data_generator(dataloader, batch_size=2, step_size=64):
    data_loader = iter(dataloader)
    index = [0] * batch_size
    batch_data = []
    for i in range(batch_size):
        batch_data.append(next(data_loader))
    done = False
    while not done or len(batch_data) > 0:
        return_data = []
        mask = np.zeros((len(batch_data), step_size), dtype=np.int32)

        for i in range(len(batch_data)):
            # Determine the valid range for the current batch
            start_idx = index[i]
            end_idx = min(start_idx + step_size, len(batch_data[i][0]))

            # Slice the data according to the current index and step_size
            return_data.append(
                [
                    batch_data[i][0][start_idx:end_idx],
                    tree_map(lambda x: x[start_idx:end_idx], batch_data[i][1]),
                    batch_data[i][2],
                    batch_data[i][3],
                    batch_data[i][4],
                ]
            )
            if start_idx + step_size > end_idx:
                return_data[i][0] = pad_array(return_data[i][0], step_size)
                return_data[i][1] = tree_map(
                    lambda x: pad_array(x, step_size), return_data[i][1]
                )

            # Update the mask to reflect valid data
            mask[i, : end_idx - start_idx] = 1

            # Update the index for the next iteration
            index[i] += step_size

            # Check if we need to load new data for this batch
            if index[i] >= len(batch_data[i][0]) and not done:
                try:
                    batch_data[i] = next(data_loader)
                    index[i] = 0  # Reset index for the new data
                except StopIteration:
                    # Handle end of the dataset if needed
                    done = True

        # Yield the batch data along with the mask
        obs = torch.from_numpy(np.stack([data[0] for data in return_data]))
        actions = [data[1] for data in return_data]
        assert all([len(action["use"]) == step_size for action in actions])
        labels = [data[2] for data in return_data]
        batch_episode_id = [data[3] for data in return_data]
        yield (obs, actions, labels, np.array(batch_episode_id),[data[4] for data in return_data]), np.array(mask)
        not_finish = [i for i, idx in enumerate(index) if idx < len(batch_data[i][0])]
        index = [index[i] for i in not_finish]
        batch_data = [batch_data[i] for i in not_finish]


if __name__ == "__main__":
    # dataset_dict = {
    #     "pipeline_test_data/demonstrations/MineRLBasaltMakeWaterfall-v0": 0,
    #     "pipeline_test_data/demonstrations/MineRLBasaltMakeWaterfall-v0-fail": 1,
    # }  # Replace with your actual dataset path
    # # batch_size = 1  # This will also be the number of workers
    # # dataloader = MinecraftDataLoader(dataset_dir, batch_size=batch_size)
    # # for trajectory_id, sequence, label in dataloader:
    # #     pass
    # batch_size = 4  # Set the desired batch size
    # dataset = MinecraftDataset(dataset_dict=dataset_dict, contrast=False)
    # step_size = 64
    # gen = data_generator(dataset, batch_size=batch_size, step_size=step_size)
    # i = 0
    # # Fetch and print a few batches
    # for batch_data, mask in gen:  # Fetch 3 batches for testing
    #     i += 1
    #     print("Mask:\n", mask)
    #     print("i:", i)
    #     print("-" * 50)
    dataset_dict = {
        "downloads/data/demonstrations/MineRLBasaltBuildVillageHouse-v0": 0,
        "downloads/data/demonstrations/MineRLBasaltCreateVillageAnimalPen-v0": 1,
    }  # Replace with your actual dataset path
    # batch_size = 1  # This will also be the number of workers
    # dataloader = MinecraftDataLoader(dataset_dir, batch_size=batch_size)
    # for trajectory_id, sequence, label in dataloader:
    #     pass
    batch_size = 4  # Set the desired batch size
    dataset = BasaltMinecraftDataset(dataset_dict=dataset_dict, contrast=False)
    step_size = 64
    gen = data_generator(dataset, batch_size=batch_size, step_size=step_size)
    i = 0
    # Fetch and print a few batches
    for batch_data, mask in gen:  # Fetch 3 batches for testing
        i += 1
        print("Mask:\n", mask)
        print("i:", i)
        print("-" * 50)
