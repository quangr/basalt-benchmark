# This code is originally from https://github.com/openai/Video-Pre-Training/blob/main/data_loader.py

import json
import glob
import os
import random
from multiprocessing import Process, Queue, Event

import numpy as np
import cv2

from basalt.vpt_lib.agent import resize_image, AGENT_RESOLUTION

QUEUE_TIMEOUT = 10

CURSOR_FILE = os.path.join(os.path.dirname(__file__), "cursors", "mouse_cursor_white_16x16.png")

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

CURSOR_FILE = os.path.join(os.path.dirname(__file__), "cursors", "mouse_cursor_white_16x16.png")

# Mapping from JSON keyboard buttons to MineRL actions
KEYBOARD_BUTTON_MAPPING = {
    "key.keyboard.escape" :"ESC",
    "key.keyboard.s" :"back",
    "key.keyboard.q" :"drop",
    "key.keyboard.w" :"forward",
    "key.keyboard.1" :"hotbar.1",
    "key.keyboard.2" :"hotbar.2",
    "key.keyboard.3" :"hotbar.3",
    "key.keyboard.4" :"hotbar.4",
    "key.keyboard.5" :"hotbar.5",
    "key.keyboard.6" :"hotbar.6",
    "key.keyboard.7" :"hotbar.7",
    "key.keyboard.8" :"hotbar.8",
    "key.keyboard.9" :"hotbar.9",
    "key.keyboard.e" :"inventory",
    "key.keyboard.space" :"jump",
    "key.keyboard.a" :"left",
    "key.keyboard.d" :"right",
    "key.keyboard.left.shift" :"sneak",
    "key.keyboard.left.control" :"sprint",
    "key.keyboard.f" :"swapHands",
}

# Template action
NOOP_ACTION = {
    "ESC": 0,
    "back": 0,
    "drop": 0,
    "forward": 0,
    "hotbar.1": 0,
    "hotbar.2": 0,
    "hotbar.3": 0,
    "hotbar.4": 0,
    "hotbar.5": 0,
    "hotbar.6": 0,
    "hotbar.7": 0,
    "hotbar.8": 0,
    "hotbar.9": 0,
    "inventory": 0,
    "jump": 0,
    "left": 0,
    "right": 0,
    "sneak": 0,
    "sprint": 0,
    "swapHands": 0,
    "camera": np.array([0, 0]),
    "attack": 0,
    "use": 0,
    "pickItem": 0,
}

MINEREC_ORIGINAL_HEIGHT_PX = 720
# Matches a number in the MineRL Java code
# search the code Java code for "constructMouseState"
# to find explanations
CAMERA_SCALER = 360.0 / 2400.0


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


def composite_images_with_alpha(image1, image2, alpha, x, y):
    """
    Draw image2 over image1 at location x,y, using alpha as the opacity for image2.
    Modifies image1 in-place
    """
    ch = max(0, min(image1.shape[0] - y, image2.shape[0]))
    cw = max(0, min(image1.shape[1] - x, image2.shape[1]))
    if ch == 0 or cw == 0:
        return
    alpha = alpha[:ch, :cw]
    image1[y:y + ch, x:x + cw, :] = (image1[y:y + ch, x:x + cw, :] * (1 - alpha) + image2[:ch, :cw, :] * alpha).astype(np.uint8)


def data_loader_worker(tasks_queue, output_queue, quit_workers_event):
    """
    Worker for the data loader.
    """

    while True:
        task = tasks_queue.get()
        if task is None:
            break
        trajectory_id, video_path, json_path = task
        video = cv2.VideoCapture(video_path)

        if not os.path.isfile(json_path):
            print("Json file not found, skipping this sample", json_path)
            continue

        with open(json_path) as json_file:
            try:
                json_lines = json_file.readlines()
                json_data = "[" + ",".join(json_lines) + "]"
                json_data = json.loads(json_data)
            except Exception as e:
                print("Error loading json file", json_path, e)
                continue

        for i in range(len(json_data)):
            if quit_workers_event.is_set():
                break
            action = json_data[i]

            is_null_action = is_json_action_null(action)
            # Add is_null info to the action buffer
            action["is_null_action"] = is_null_action
            action["camera"] = np.array(action["camera"])
            # Update hotbar selection

            # Read frame even if this is null so we progress forward
            ret, frame = video.read()
            if ret:
                cv2.cvtColor(frame, code=cv2.COLOR_BGR2RGB, dst=frame)
                frame = np.asarray(np.clip(frame, 0, 255), dtype=np.uint8)
                frame = resize_image(frame, AGENT_RESOLUTION)
                output_queue.put((trajectory_id, frame, action), timeout=QUEUE_TIMEOUT)
            else:
                print(f"Could not read frame from video {video_path}")
                break
        video.release()
        # Tell that the trajectory ended
        output_queue.put((trajectory_id, None, None), timeout=QUEUE_TIMEOUT)
        if quit_workers_event.is_set():
            break

    # Tell that we ended
    output_queue.put(None)

class DataLoader:
    def __init__(self, dataset_dir, n_workers=8, batch_size=8, n_epochs=1, max_queue_size=16, do_not_cut_epoch_tail=False):
        assert n_workers >= batch_size, "Number of workers must be equal or greater than batch size"
        self.dataset_dir = dataset_dir
        self.n_workers = n_workers
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.max_queue_size = max_queue_size
        self.do_not_cut_epoch_tail = do_not_cut_epoch_tail
        unique_ids = glob.glob(os.path.join(dataset_dir, "*.mp4"))
        unique_ids = list(set([os.path.basename(x).split(".")[0] for x in unique_ids]))
        self.unique_ids = unique_ids
        # Create tuples of (video_path, json_path) for each unique_id
        demonstration_tuples = []
        for unique_id in unique_ids:
            video_path = os.path.abspath(os.path.join(dataset_dir, unique_id + ".mp4"))
            json_path = os.path.abspath(os.path.join(dataset_dir, unique_id + ".jsonl"))
            demonstration_tuples.append((video_path, json_path))

        assert n_workers <= len(demonstration_tuples), f"n_workers should be lower or equal than number of demonstrations {len(demonstration_tuples)}"

        # Repeat dataset for n_epochs times, shuffling the order for
        # each epoch
        self.demonstration_tuples = []
        for i in range(n_epochs):
            random.shuffle(demonstration_tuples)
            self.demonstration_tuples += demonstration_tuples

        self.task_queue = Queue()
        self.n_steps_processed = 0
        self.trajectory_id_to_mp4_path = {}
        for trajectory_id, task in enumerate(self.demonstration_tuples):
            self.task_queue.put((trajectory_id, *task))
            self.trajectory_id_to_mp4_path[trajectory_id] = task[0]
        for _ in range(n_workers):
            self.task_queue.put(None)

        self.output_queues = [Queue(maxsize=max_queue_size) for _ in range(n_workers)]
        self.quit_workers_event = Event()
        self.processes = [
            Process(
                target=data_loader_worker,
                args=(
                    self.task_queue,
                    output_queue,
                    self.quit_workers_event,
                ),
                daemon=True
            )
            for output_queue in self.output_queues
        ]
        for process in self.processes:
            process.start()

    def get_mp4_path_for_trajectory_id(self, trajectory_id):
        return self.trajectory_id_to_mp4_path[trajectory_id]

    def __iter__(self):
        return self

    def __next__(self):
        batch_frames = []
        batch_actions = []
        batch_episode_id = []

        # Avoid having same episode multiple times, so limit by workers
        for i in range(min(self.batch_size, len(self.processes))):
            workitem = self.output_queues[self.n_steps_processed % self.n_workers].get(timeout=QUEUE_TIMEOUT)
            if workitem is None:
                if self.do_not_cut_epoch_tail:
                    process = self.processes[self.n_steps_processed % self.n_workers]
                    process.terminate()
                    process.join()
                    # Remove from list
                    del self.processes[self.n_steps_processed % self.n_workers]
                    self.output_queues[self.n_steps_processed % self.n_workers].close()
                    del self.output_queues[self.n_steps_processed % self.n_workers]
                    self.n_workers -= 1
                    if self.n_workers == 0:
                        raise StopIteration()
                    continue
                else:
                    # Stop iteration when first worker runs out of work to do.
                    # Yes, this has a chance of cutting out a lot of the work,
                    # but this ensures batches will remain diverse, instead
                    # of having bad ones in the end where potentially
                    # one worker outputs all samples to the same batch.
                    raise StopIteration()
            trajectory_id, frame, action = workitem
            batch_frames.append(frame)
            batch_actions.append(action)
            batch_episode_id.append(trajectory_id)
            self.n_steps_processed += 1
        return batch_frames, batch_actions, batch_episode_id

    def __del__(self):
        for process in self.processes:
            process.terminate()
            process.join()
