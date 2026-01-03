from typing import List, Optional, Tuple, TypedDict, cast

class Room(TypedDict):
    id: str
    capacity: int
    equipment: List[str]
    accessible: bool
    distance_m: int
    booked: List[Tuple[str, str, int]]

class RoomStatus(Room):
    free: bool


class AvailableRooms(TypedDict):
    rooms: List[RoomStatus]


class RoomRequirement(TypedDict):
    date: str
    time: str
    duration_min: int
    attendees: int
    needs: List[str]
    accessible_required: bool


class RoomSelectionTask(TypedDict):
    id: str
    task_input: RoomRequirement
    expected_choice: str

ROOMS: List[Room] = [
    {
        "id": "Orion",
        "capacity": 4,
        "equipment": ["tv", "whiteboard"],
        "accessible": True,
        "distance_m": 12,
        "booked": [("2025-10-13", "10:00", 60), ("2025-10-13", "15:00", 30)],
    },
    {
        "id": "Lyra",
        "capacity": 10,
        "equipment": ["projector", "whiteboard", "confphone"],
        "accessible": True,
        "distance_m": 30,
        "booked": [("2025-10-13", "09:30", 30), ("2025-10-13", "11:00", 60)],
    },
    {
        "id": "Vega",
        "capacity": 6,
        "equipment": ["tv"],
        "accessible": False,
        "distance_m": 22,
        "booked": [("2025-10-13", "14:00", 60)],
    },
    {
        "id": "Nova",
        "capacity": 12,
        "equipment": ["ledwall", "whiteboard", "confphone"],
        "accessible": True,
        "distance_m": 45,
        "booked": [],
    },
    {
        "id": "Quark",
        "capacity": 8,
        "equipment": ["projector", "whiteboard"],
        "accessible": False,
        "distance_m": 18,
        "booked": [("2025-10-13", "10:30", 30)],
    },
    # Two extra to create harder ties
    {
        "id": "Atlas",
        "capacity": 6,
        "equipment": ["projector", "whiteboard"],
        "accessible": True,
        "distance_m": 10,
        "booked": [("2025-10-13", "09:00", 30), ("2025-10-13", "13:30", 30)],
    },
    {
        "id": "Pulse",
        "capacity": 8,
        "equipment": ["tv", "whiteboard", "confphone"],
        "accessible": True,
        "distance_m": 8,
        "booked": [("2025-10-13", "16:30", 30)],
    },
]