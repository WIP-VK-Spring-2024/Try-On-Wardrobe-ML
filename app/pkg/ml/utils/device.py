import torch


def get_device(prefer_device: str):
    """
    Returns device closest to prefer device
    Args:
        prefer_device - str - device that would be best if this device is available
    returns device:str

    Example: prefer_device="cuda:1"
    """
    prefer_device_index = int(prefer_device[-1])

    num_of_gpus = torch.cuda.device_count()

    if num_of_gpus == 0:
        return "cpu"
    elif prefer_device_index < num_of_gpus:
        return prefer_device
    else:
        return "cuda:0"


if __name__ == "__main__":
    print(get_device("cuda:1"))