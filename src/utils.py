import random
import numpy as np
import torch
from torch import nn
# from src.model.spatialdual import DualSpatial
import pickle
import torch.distributed as dist
from transformers import AdamW
import os
import collections
import logging
from torch.serialization import default_restore_location

logger = logging.getLogger(__name__)
CheckpointState = collections.namedtuple("CheckpointState",
                                         ['model_dict', 'optimizer_dict', 'scheduler_dict', 'offset', 'epoch',
                                          'encoder_params'])
def pack_tensor_2D(lstlst, default, dtype, length=None):
    batch_size = len(lstlst)
    length = length if length is not None else max(len(l) for l in lstlst)
    tensor = default * torch.ones((batch_size, length), dtype=dtype)
    for i, l in enumerate(lstlst):
        tensor[i, :len(l)] = torch.tensor(l, dtype=dtype)
    return tensor
    
def get_model_obj(model: nn.Module):
    return model.module if hasattr(model, 'module') else model

def _save_checkpoint(args, model, optimizer, scheduler, step: int, loss: float) -> str:
    offset = step
    epoch = 0
    model_to_save = get_model_obj(model)
    cp = os.path.join(args.checkpoint_dir,'checkpoint-' + str(offset) + '-' + str(round(loss.item(), 4)))

    meta_params = {}

    state = CheckpointState(model_to_save.state_dict(),
                            optimizer.state_dict(),
                            scheduler.state_dict(),
                            offset,
                            epoch, meta_params
                            )
    torch.save(state._asdict(), cp)
    logger.info('Saved checkpoint at %s', cp)
    return cp

def get_optimizer(args, model: nn.Module, weight_decay: float = 0.0, ) -> torch.optim.Optimizer:
    # if not args.spatial_pretraining:
    #     for name, param in model.named_parameters():
    #         if 'spatial' not in name:
    #             param.requires_grad = False
        
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if p.requires_grad and not any(nd in n for nd in no_decay)],
        'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if p.requires_grad and any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    if args.optimizer == "adamW":
        return AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    else:
        raise Exception("optimizer {0} not recognized! Can only be lamb or adamW".format(args.optimizer))

def all_gather_list(data, group=None, max_size=16384):
    """Gathers arbitrary data from all nodes into a list.
    Similar to :func:`~torch.distributed.all_gather` but for arbitrary Python
    data. Note that *data* must be picklable.
    Args:
        data (Any): data from the local worker to be gathered on other workers
        group (optional): group of the collective
    """
    SIZE_STORAGE_BYTES = 4  # int32 to encode the payload size

    enc = pickle.dumps(data)
    enc_size = len(enc)

    if enc_size + SIZE_STORAGE_BYTES > max_size:
        raise ValueError(
            'encoded data exceeds max_size, this can be fixed by increasing buffer size: {}'.format(enc_size))

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    buffer_size = max_size * world_size

    if not hasattr(all_gather_list, '_buffer') or \
            all_gather_list._buffer.numel() < buffer_size:
        all_gather_list._buffer = torch.cuda.ByteTensor(buffer_size)
        all_gather_list._cpu_buffer = torch.ByteTensor(max_size).pin_memory()

    buffer = all_gather_list._buffer
    buffer.zero_()
    cpu_buffer = all_gather_list._cpu_buffer

    assert enc_size < 256 ** SIZE_STORAGE_BYTES, 'Encoded object size should be less than {} bytes'.format(
        256 ** SIZE_STORAGE_BYTES)

    size_bytes = enc_size.to_bytes(SIZE_STORAGE_BYTES, byteorder='big')

    cpu_buffer[0:SIZE_STORAGE_BYTES] = torch.ByteTensor(list(size_bytes))
    cpu_buffer[SIZE_STORAGE_BYTES: enc_size + SIZE_STORAGE_BYTES] = torch.ByteTensor(list(enc))

    start = rank * max_size
    size = enc_size + SIZE_STORAGE_BYTES
    buffer[start: start + size].copy_(cpu_buffer[:size])

    if group is None:
        group = dist.group.WORLD
    dist.all_reduce(buffer, group=group)

    try:
        result = []
        for i in range(world_size):
            out_buffer = buffer[i * max_size: (i + 1) * max_size]
            size = int.from_bytes(out_buffer[0:SIZE_STORAGE_BYTES], byteorder='big')
            if size > 0:
                result.append(pickle.loads(bytes(out_buffer[SIZE_STORAGE_BYTES: size + SIZE_STORAGE_BYTES].tolist())))
        return result
    except pickle.UnpicklingError:
        raise Exception(
            'Unable to unpickle data from other workers. all_gather_list requires all '
            'workers to enter the function together, so this error usually indicates '
            'that the workers have fallen out of sync somehow. Workers can fall out of '
            'sync if one of them runs out of memory, or if there are other conditions '
            'in your training script that can cause one worker to finish an epoch '
            'while other workers are still iterating over their portions of the data.'
        )
        
def is_first_worker():
    return not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0
    
def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

def poi_collate_function(batch, max_poi_length):
    length = max_poi_length
    input_ids = [x["input_ids"] for x in batch]
    token_type = [x['token_type'] for x in batch]
    attention_mask = [x["attention_mask"] for x in batch]
    data = {
        "input_ids": pack_tensor_2D(input_ids, default=0, 
            dtype=torch.int64, length=length),
        "token_type": pack_tensor_2D(token_type, default=0, 
            dtype=torch.int64, length=length),
        "attention_mask": pack_tensor_2D(attention_mask, default=0, 
            dtype=torch.int64, length=length),
        }
    coordinates = [x['coordinate'] for x in batch]        
    pids = [x['id'] for x in batch]
    return data, coordinates

def poi_collate_function_cross(batch):
    input_ids = [x["input_ids"] for x in batch]
    token_type = [x['token_type'] for x in batch]
    attention_mask = [x["attention_mask"] for x in batch]
    data = {
        "input_ids": input_ids,
        "token_type": token_type,
        "attention_mask": attention_mask,
        }
    coordinates = [x['coordinate'] for x in batch]        
    pids = [x['id'] for x in batch]
    return data, coordinates


def load_states_from_checkpoint(model_file: str) -> CheckpointState:
    logger.info('Reading saved model from %s', model_file)
    state_dict = torch.load(model_file, map_location=lambda s, l: default_restore_location(s, 'cpu'))
    logger.info('model_state_dict keys %s', state_dict.keys())
    return CheckpointState(**state_dict)