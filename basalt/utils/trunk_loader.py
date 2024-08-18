import os
import glob
import json
import cv2
import numpy as np
import multiprocessing as mp
from queue import Empty
from basalt.vpt_lib.agent import resize_image, AGENT_RESOLUTION

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

SEQUENCE_LENGTH=64

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

                if len(sequence) == SEQUENCE_LENGTH:
                    output_queue.put((trajectory_id, sequence), timeout=QUEUE_TIMEOUT)
                    sequence = []
            else:
                print(f"Could not read frame from video {video_path}")
                break

        # Output any remaining frames (less than SEQUENCE_LENGTH) without padding
        if sequence:
            output_queue.put((trajectory_id, sequence), timeout=QUEUE_TIMEOUT)

        video.release()
        output_queue.put((trajectory_id, None), timeout=QUEUE_TIMEOUT)

        if quit_workers_event.is_set():
            break

    output_queue.put(None)

class MinecraftDataLoader:
    def __init__(self, dataset_dir, batch_size=1):
        self.dataset_dir = dataset_dir
        self.batch_size = batch_size
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
        return list(set([os.path.basename(x).split(".")[0] for x in unique_ids]))

    def _create_demonstration_tuples(self):
        demonstration_tuples = []
        for unique_id in self.unique_ids:
            video_path = os.path.abspath(os.path.join(self.dataset_dir, unique_id + ".mp4"))
            json_path = os.path.abspath(os.path.join(self.dataset_dir, unique_id + ".jsonl"))
            demonstration_tuples.append((video_path, json_path))
        return demonstration_tuples

    def _start_workers(self):
        for _ in range(self.num_workers):
            worker = mp.Process(target=data_loader_worker, 
                                args=(self.tasks_queue, self.output_queue, self.quit_workers_event))
            worker.start()
            self.workers.append(worker)

    def __iter__(self):
        for i in range(0, len(self.demonstration_tuples), self.batch_size):
            batch = self.demonstration_tuples[i:i+self.batch_size]
            
            # Queue tasks for the current batch
            for j, (video_path, json_path) in enumerate(batch):
                self.tasks_queue.put((i+j, video_path, json_path))

            # Process the entire batch before moving to the next one
            active_trajectories = set(range(i, i+len(batch)))
            batch_data = {j: [] for j in active_trajectories}

            while active_trajectories:
                try:
                    item = self.output_queue.get(timeout=QUEUE_TIMEOUT)
                    if item is None:
                        break
                    trajectory_id, sequence = item
                    if sequence is None:
                        active_trajectories.remove(trajectory_id)
                    else:
                        batch_data[trajectory_id].append(sequence)
                        # Yield complete sequences
                        yield trajectory_id, batch_data[trajectory_id][0]
                        batch_data[trajectory_id] = batch_data[trajectory_id][1:]
                except Empty:
                    if self.quit_workers_event.is_set():
                        return

            # Yield any remaining data in the batch
            for trajectory_id, data in batch_data.items():
                if data:
                    yield trajectory_id, data

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




class ContrastMinecraftDataLoader:
    def __init__(self, dataset_dir, batch_size=1):
        self.dataset_dir = dataset_dir
        self.batch_size = batch_size
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
        fail_unique_ids = glob.glob(os.path.join(self.dataset_dir+"-fail", "*.mp4"))
        return list(set([(1,os.path.basename(x).split(".")[0]) for x in unique_ids]))+list(set([(0,os.path.basename(x).split(".")[0]) for x in fail_unique_ids]))

    def _create_demonstration_tuples(self):
        demonstration_tuples = []
        for label,unique_id in self.unique_ids:
            if label==1:
                video_path = os.path.abspath(os.path.join(self.dataset_dir, unique_id + ".mp4"))
                json_path = os.path.abspath(os.path.join(self.dataset_dir, unique_id + ".jsonl"))
            else:
                video_path = os.path.abspath(os.path.join(self.dataset_dir+"-fail", unique_id + ".mp4"))
                json_path = os.path.abspath(os.path.join(self.dataset_dir+"-fail", unique_id + ".jsonl"))
            demonstration_tuples.append((video_path, json_path))
        return demonstration_tuples

    def _start_workers(self):
        for _ in range(self.num_workers):
            worker = mp.Process(target=data_loader_worker, 
                                args=(self.tasks_queue, self.output_queue, self.quit_workers_event))
            worker.start()
            self.workers.append(worker)

    def __iter__(self):
        for i in range(0, len(self.demonstration_tuples), self.batch_size):
            batch = self.demonstration_tuples[i:i+self.batch_size]
            
            # Queue tasks for the current batch
            for j, (video_path, json_path) in enumerate(batch):
                self.tasks_queue.put((i+j, video_path, json_path))

            # Process the entire batch before moving to the next one
            active_trajectories = set(range(i, i+len(batch)))
            batch_data = {j: [] for j in active_trajectories}

            while active_trajectories:
                try:
                    item = self.output_queue.get(timeout=QUEUE_TIMEOUT)
                    if item is None:
                        break
                    trajectory_id, sequence = item
                    if sequence is None:
                        active_trajectories.remove(trajectory_id)
                    else:
                        batch_data[trajectory_id].append(sequence)
                        # Yield complete sequences
                        yield trajectory_id, batch_data[trajectory_id][0], self.unique_ids[trajectory_id][0]
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



if __name__ == "__main__":
    dataset_dir = "pipeline_test_data/demonstrations/MineRLBasaltMakeWaterfall-v0"  # Replace with your actual dataset path
    batch_size = 1  # This will also be the number of workers
    dataloader = ContrastMinecraftDataLoader(dataset_dir, batch_size=batch_size)
    for trajectory_id, sequence,label in dataloader:
        pass