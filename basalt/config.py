from dataclasses import dataclass, field
import enum
from typing import List, Literal
import tyro


@dataclass
class BaseDataConfig:
    task_data_prefix: str
    checkpoint_prefix: str
    scale: str

    @property
    def weights_path(self) -> List[str]:
        return f"{self.checkpoint_prefix}/rl-from-foundation-{self.scale}.weights"

    @property
    def model_path(self) -> List[str]:
        return f"{self.checkpoint_prefix}/{self.scale}.model"


@dataclass
class GCloudDataConfig(BaseDataConfig):
    task_data_prefix: str = "/data/demonstrations/"
    checkpoint_prefix: str = "/data/VPT-models"
    scale: str = "2x"


@dataclass
class GpuDataConfig(BaseDataConfig):
    task_data_prefix: str = "/data/quangr/demonstrations/"
    checkpoint_prefix: str = "/data/quangr/VPT-models"
    scale: str = "1x"


class TaskType(enum.Enum):
    CaveVsWater = [
        "MineRLBasaltFindCave-v0",
        "MineRLBasaltMakeWaterfall-v0",
    ]
    MakeWaterfall = [
        "MineRLBasaltMakeWaterfall-v0",
    ]


DefaultDataConfig = GCloudDataConfig
