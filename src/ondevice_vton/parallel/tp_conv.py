"""Channel-parallel (Megatron-style) TP for the Mobile-VTON ResnetBlock2D.

Stage 1b: split the resnet's MIDDLE channels across ranks so the heavy pointwise
(1x1) convs run in parallel, not replicated like attention-only TP left them.

The resnet body is (spatial omitted):
    x -> norm1 -> act -> [up/down] -> conv1(dw1 -> pw1) -> +temb -> norm2 -> act
      -> dropout -> conv2(dw2 -> pw2) -> (+shortcut)/scale
conv1/conv2 are depthwise-separable: dw_conv (per-channel 3x3) + pw_conv (1x1
channel mixing). We split the middle-channel region [pw1 out .. pw2 in]:

  pw1   (in->middle, 1x1)  : COLUMN-parallel  (slice middle out; dw1 stays full)
  time_emb_proj (->middle) : COLUMN-parallel  (slice middle out)
  norm2 (GroupNorm/middle) : reslice to the middle half -- the split lands on
                             whole groups, so per-group stats stay LOCAL (no sync)
  dw2   (depthwise/middle) : slice per-channel (free, no comm)
  pw2   (middle->out, 1x1) : ROW-parallel     (slice middle in) -> 1 all_reduce

Every middle-channel intermediate stays consistently sliced, so the ORIGINAL
resnet forward runs unchanged -- the only cross-rank step is summing pw2's
partial output, done by an all_reduce forward-hook on conv2.pw_conv. One
all_reduce per resnet block, mirroring the attention TP.

    shard_resnet_(block, rank, world)        # slice weights + install hook
    shard_denoiser_conv_(model, rank, world) # do it for every supported block
"""
import torch
import torch.distributed as dist


def _slice_out(layer, s, e):
    """Keep output features [s:e] (column-parallel): rows of weight + bias."""
    layer.weight = torch.nn.Parameter(layer.weight.data[s:e].clone())
    if getattr(layer, "bias", None) is not None:
        layer.bias = torch.nn.Parameter(layer.bias.data[s:e].clone())
    if hasattr(layer, "out_channels"):
        layer.out_channels = e - s
    if hasattr(layer, "out_features"):
        layer.out_features = e - s


def _slice_in_conv(conv, s, e, keep_bias):
    """Keep input channels [s:e] of a 1x1 conv (row-parallel): cols (dim=1)."""
    conv.weight = torch.nn.Parameter(conv.weight.data[:, s:e].clone())
    if conv.bias is not None and not keep_bias:
        conv.bias = torch.nn.Parameter(torch.zeros_like(conv.bias.data))
    conv.in_channels = e - s


def _slice_depthwise(conv, s, e):
    """Slice a depthwise conv (groups==channels) to channels [s:e]."""
    conv.weight = torch.nn.Parameter(conv.weight.data[s:e].clone())  # [C,1,k,k]
    if conv.bias is not None:
        conv.bias = torch.nn.Parameter(conv.bias.data[s:e].clone())
    conv.in_channels = e - s
    conv.out_channels = e - s
    conv.groups = e - s


def _reslice_groupnorm(gn, s, e, world):
    """Reslice GroupNorm to channels [s:e]; split lands on whole groups -> local."""
    gn.num_channels = e - s
    gn.num_groups = gn.num_groups // world
    gn.weight = torch.nn.Parameter(gn.weight.data[s:e].clone())
    gn.bias = torch.nn.Parameter(gn.bias.data[s:e].clone())


def tp_resnet_supported(block, world):
    """True if this ResnetBlock2D can be channel-parallel sharded across `world`.

    Needs depthwise-separable conv1/conv2, the simple ('default') time-embedding
    add, and a MIDDLE-channel split that lands on whole GroupNorm groups.
    """
    if not (_is_sepconv(block.conv1) and _is_sepconv(block.conv2)):
        return False
    if getattr(block, "time_embedding_norm", "default") != "default":
        return False
    if block.time_emb_proj is None:
        return False
    mid = block.norm2.num_channels
    g = block.norm2.num_groups
    return (mid % world == 0 and g % world == 0
            and block.time_emb_proj.out_features == mid
            and (mid // world) % (mid // g) == 0)


def _is_sepconv(conv):
    return hasattr(conv, "dw_conv") and hasattr(conv, "pw_conv")


def shard_resnet_(block, rank, world):
    """In-place channel-parallel shard of one ResnetBlock2D across `world` ranks."""
    mid = block.norm2.num_channels
    local = mid // world
    s, e = rank * local, (rank + 1) * local

    _slice_out(block.conv1.pw_conv, s, e)         # column-parallel pw1
    _slice_out(block.time_emb_proj, s, e)         # column-parallel time embed
    _reslice_groupnorm(block.norm2, s, e, world)  # local per-group stats
    _slice_depthwise(block.conv2.dw_conv, s, e)   # depthwise on middle slice
    # row-parallel pw2: keep bias on rank 0 only so the all_reduce counts it once
    _slice_in_conv(block.conv2.pw_conv, s, e, keep_bias=(rank == 0))

    # sum the partial pw2 outputs across ranks (the one cross-rank step)
    def _all_reduce_hook(_module, _inp, out):
        dist.all_reduce(out, op=dist.ReduceOp.SUM)
        return out

    block.conv2.pw_conv.register_forward_hook(_all_reduce_hook)
    return block


def iter_resnet_blocks(model):
    from Mobile_VTON.models.resnet import ResnetBlock2D
    for name, m in model.named_modules():
        if isinstance(m, ResnetBlock2D):
            yield name, m


def shard_denoiser_conv_(model, rank, world):
    """Channel-parallel shard every supported ResnetBlock2D. Returns (done, skip)."""
    done, skip = [], []
    for name, block in iter_resnet_blocks(model):
        if tp_resnet_supported(block, world):
            shard_resnet_(block, rank, world)
            done.append(name)
        else:
            skip.append(name)
    return done, skip
