"""
Helpers for distributed training.
"""

import io
import os
import socket

import blobfile as bf
import torch
# from mpi4py import MPI #not sure why this needs to be set after torch
import torch.distributed as dist

# Change this to reflect your cluster layout.
# The GPU for a given rank is (rank % GPUS_PER_NODE).
# 全局的设备 id，由外部训练脚本传入控制，比如 0 / 1 / 2 / 3
_DEVICE_ID = 0

def set_device_id(device_id: int):
    """
    在训练脚本中调用，用来指定当前进程要使用哪块 GPU。
    """
    global _DEVICE_ID
    _DEVICE_ID = int(device_id)

def get_device_id() -> int:
    """
    返回当前使用的 GPU 序号（整型），给 DDP 用。
    """
    return _DEVICE_ID

def _get_mpi_comm():
    from mpi4py import MPI
    return MPI.COMM_WORLD


GPUS_PER_NODE = torch.cuda.device_count() if torch.cuda.is_available() else 1

SETUP_RETRY_COUNT = 3

def setup_dist():
    """
    Setup a distributed process group.
    """
    if dist.is_initialized():
        return

    # 根据全局 _DEVICE_ID 选卡，而不是 MPI 的 rank
    if torch.cuda.is_available():
        torch.cuda.set_device(_DEVICE_ID)

    comm = _get_mpi_comm()
    backend = "gloo" if not torch.cuda.is_available() else "nccl"

    if backend == "gloo":
        hostname = "localhost"
    else:
        hostname = "127.0.0.1"
    os.environ["MASTER_ADDR"] = comm.bcast(hostname, root=0)
    os.environ["RANK"] = str(comm.rank)
    os.environ["WORLD_SIZE"] = str(comm.size)

    port = comm.bcast(_find_free_port(), root=0)
    os.environ["MASTER_PORT"] = str(port)

    dist.init_process_group(backend=backend, init_method="env://")



def dev():
    """
    返回当前要使用的 torch.device。
    """
    if torch.cuda.is_available():
        return torch.device(f"cuda:{_DEVICE_ID}")
    return torch.device("cpu")



def load_state_dict(path, **kwargs):
    """
    Load a PyTorch file without redundant fetches across MPI ranks.
    """
    chunk_size = 2**30  # MPI has a relatively small size limit
    if dist.is_initialized() and int(os.environ.get("WORLD_SIZE", "1")) > 1 and "RANK" in os.environ:
        # torchrun/env:// path: every rank can read local/remote storage directly.
        return torch.load(path, **kwargs)

    comm = _get_mpi_comm()
    if comm.Get_rank() == 0:
        with bf.BlobFile(path, "rb") as f:
            data = f.read()
        num_chunks = len(data) // chunk_size
        if len(data) % chunk_size:
            num_chunks += 1
        comm.bcast(num_chunks)
        for i in range(0, len(data), chunk_size):
            comm.bcast(data[i : i + chunk_size])
    else:
        num_chunks = comm.bcast(None)
        data = bytes()
        for _ in range(num_chunks):
            data += comm.bcast(None)

    return torch.load(io.BytesIO(data), **kwargs)


def sync_params(params):
    """
    Synchronize a sequence of Tensors across ranks from rank 0.
    """
    for p in params:
        with torch.no_grad():
            dist.broadcast(p, 0)


def _find_free_port():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind(("", 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]
    finally:
        s.close()
