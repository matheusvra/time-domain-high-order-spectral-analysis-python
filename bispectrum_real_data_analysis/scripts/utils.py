import os

def get_cpus_available() -> int:
    """Function to get the number of CPUs available on the system.

    Returns:
        int: number of CPUs available on the system
    """
    return max(1, len(os.sched_getaffinity(0)))
