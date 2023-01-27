import os
import time

def get_cpus_available() -> int:
    """Function to get the number of CPUs available on the system.

    Returns:
        int: number of CPUs available on the system
    """
    return max(1, len(os.sched_getaffinity(0)))

def seconds_to_formatted_time(seconds: float) -> str:
    return "".join(
        [
            f"{value}{unit}" for value, unit in zip(
                time.strftime('%H:%M:%S', time.gmtime(seconds)
                ).split(":"), ["h", "m", "s"]) if value != "00"
        ]
    )
