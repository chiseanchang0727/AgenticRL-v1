import json
from typing import Tuple, cast, List
from agentlightning.types import Dataset
from data.rooms import  Room, ROOMS, RoomStatus, AvailableRooms, RoomRequirement, RoomSelectionTask
import math

def load_room_tasks() -> Dataset[RoomSelectionTask]:
    tasks: List[RoomSelectionTask] = []
    for line in open("data/room_tasks.jsonl"):
        task = json.loads(line)
        tasks.append(RoomSelectionTask(**task))
    return cast(Dataset[RoomSelectionTask], tasks)



def load_train_val_dataset() -> Tuple[Dataset[RoomSelectionTask], Dataset[RoomSelectionTask]]:
    dataset_full = load_room_tasks()
    train_split = math.floor(len(dataset_full)*0.8)
    dataset_train = [dataset_full[i] for i in range(train_split)]
    dataset_val = [dataset_full[i] for i in range(train_split, len(dataset_full))]
    return cast(Dataset[RoomSelectionTask], dataset_train), cast(Dataset[RoomSelectionTask], dataset_val)



