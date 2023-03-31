import os
import time
import numpy as np

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

def standardize_array(array: np.ndarray, scale_to_unit: bool = False) -> np.ndarray:
    """Function to standardize an array.

    Args:
        array (np.ndarray): array to be standardized

    Returns:
        np.ndarray: standardized array
    """
    if scale_to_unit:
        standardized_array =  (array - array.min())/(array.max() - array.min())
    
    else:
        standardized_array = (array - array.mean()) / array.std()

    return standardized_array
