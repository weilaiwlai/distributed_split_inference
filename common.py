import msgspec
import torch
import enum

class ReqHiddenStatesMessage(msgspec.Struct, tag=True):
    request_id: str
    seq_id: str
    hidden_states: torch.Tensor

class RespTokenIdMessage(msgspec.Struct, tag=True):
    request_id: str
    seq_id: str
    predicted_token_id: torch.Tensor

class ReqEndMessage(msgspec.Struct, tag=True):
    request_id: str
    seq_id: str

class RespEndMessage(msgspec.Struct, tag=True):
    request_id: str
    seq_id: str