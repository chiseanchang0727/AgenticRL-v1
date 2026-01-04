from typing import List, Optional, Tuple, TypedDict, cast
from data.rooms import  Room, ROOMS, RoomStatus, AvailableRooms, RoomRequirement, RoomSelectionTask

def overlaps(start: str, dur: int, other_start: str, other_dur: int) -> bool:
    def tmin(t: str):
        return int(t[:2]) * 60 + int(t[3:])

    a0, a1 = tmin(start), tmin(start) + dur
    b0, b1 = tmin(other_start), tmin(other_start) + other_dur
    return max(a0, b0) < min(a1, b1)


def get_rooms_and_availability(date: str, time_str: str, duration_min: int) -> AvailableRooms:
    avail: List[RoomStatus] = []
    for r in ROOMS:
        free = all(
            not (b_date == date and overlaps(time_str, duration_min, b_time, b_dur))
            for (b_date, b_time, b_dur) in r["booked"]
        )
        item: RoomStatus = {
            **r,
            "free": free,
        }
        avail.append(item)
    return {"rooms": avail}
