import os
import sys
import os.path as osp
import argparse
import gc
import ctypes
import json
from typing import Literal, Tuple


def _free_modules(*mods):
    """Drop CPU-resident modules and return memory to the OS (glibc malloc_trim)."""
    for m in mods:
        try:
            del m
        except Exception:
            pass
    gc.collect()
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass


class TiledVaeDecoder:
    """Decode a VAE latent in overlapping spatial tiles.

    The full-resolution decode (1024x768) is the pipeline's peak-RAM stage: its
    intermediate conv feature maps are huge and blow past the Pi's 8GB. Decoding
    in tiles means those intermediates only ever materialize at tile resolution,
    capping peak RAM regardless of dtype or oneDNN bf16->fp32 fallback. Tiles
    overlap and are feather-blended so the seams don't show. Drop-in for the
    `vae_decoder(latents)` call: same (sample, latent_embeds) signature/output.
    """

    def __init__(self, decoder, tile=48, overlap=16, scale=8,
                 sp_rank=0, sp_world=1, sp_group=None):
        self.decoder = decoder    # the real Decoder module
        self.tile = tile          # latent-space tile size (square)
        self.overlap = overlap    # latent-space overlap between adjacent tiles
        self.scale = scale        # decoder spatial upsample factor (8 for SD VAE)
        # SPATIAL: the tiled decode is embarrassingly parallel -- the tiles are
        # already independent and feather-blended. Each rank decodes its slice of
        # the tile list (round-robin) into a local out/acc, then the two buffers are
        # summed across ranks with a single pair of all_reduces. N ranks -> ~Nx on
        # the VAE decode (the pipeline's other big fixed cost). No halo / norm sync.
        self.sp_rank = sp_rank
        self.sp_world = sp_world
        self.sp_group = sp_group

    @staticmethod
    def _starts(total, tile, stride):
        if total <= tile:
            return [0]
        s = list(range(0, total - tile + 1, stride))
        if s[-1] != total - tile:
            s.append(total - tile)  # flush the last tile to the border
        return s

    @staticmethod
    def _window(n, ov, eps=0.02):
        # 1D feather: ramps eps->1 over the overlap on each side, 1.0 in the
        # middle. Stays strictly positive so the weight-normalized blend never
        # divides by zero at the image border (where only one tile contributes).
        w = torch.ones(n)
        if ov > 0 and n > 2 * ov:
            ramp = torch.linspace(eps, 1.0, ov)
            w[:ov] = ramp
            w[-ov:] = ramp.flip(0)
        return w

    def __call__(self, sample, latent_embeds=None):
        with torch.no_grad():
            B, C, H, W = sample.shape
            tile, ov, s = self.tile, self.overlap, self.scale
            if H <= tile and W <= tile:
                print(f"[TiledVaeDecoder] input={tuple(sample.shape)} -> 1 tile (no split)",
                      file=sys.stderr, flush=True)
                return self.decoder(sample, latent_embeds)
            n = len(self._starts(H, tile, tile - ov)) * len(self._starts(W, tile, tile - ov))
            print(f"[TiledVaeDecoder] input={tuple(sample.shape)} tile={tile} ov={ov} -> {n} tiles",
                  file=sys.stderr, flush=True)
            stride = tile - ov
            tiles = [(hi, wi) for hi in self._starts(H, tile, stride)
                     for wi in self._starts(W, tile, stride)]
            out = acc = None
            for ti, (hi, wi) in enumerate(tiles):
                # SPATIAL: this rank only decodes its round-robin share of the tiles.
                if self.sp_world > 1 and (ti % self.sp_world) != self.sp_rank:
                    continue
                z = sample[:, :, hi:hi + tile, wi:wi + tile]
                img = self.decoder(z, latent_embeds)        # [B, Co, tile*s, tile*s]
                th, tw = img.shape[2], img.shape[3]
                if out is None:
                    out = torch.zeros(B, img.shape[1], H * s, W * s, dtype=img.dtype)
                    acc = torch.zeros(1, 1, H * s, W * s, dtype=img.dtype)
                mask = (self._window(th, ov * s).to(img.dtype)[:, None]
                        * self._window(tw, ov * s).to(img.dtype)[None, :])[None, None]
                oh, ow = hi * s, wi * s
                out[:, :, oh:oh + th, ow:ow + tw] += img * mask
                acc[:, :, oh:oh + th, ow:ow + tw] += mask
                del img
                gc.collect()
            if self.sp_world > 1:
                # a rank that drew no tiles still must join the all_reduces
                if out is None:
                    out = torch.zeros(B, 3, H * s, W * s, dtype=sample.dtype)
                    acc = torch.zeros(1, 1, H * s, W * s, dtype=sample.dtype)
                import torch.distributed as _dist
                _dist.all_reduce(out, op=_dist.ReduceOp.SUM, group=self.sp_group)
                _dist.all_reduce(acc, op=_dist.ReduceOp.SUM, group=self.sp_group)
            return out / acc.clamp_min(1e-6)

import torch
import torch.utils.data as data
import torchvision
import numpy as np
from PIL import Image
from torchvision import transforms
from tqdm.auto import tqdm

from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration, set_seed

from diffusers import FlowMatchEulerDiscreteScheduler
from diffusers.models.autoencoders import AutoencoderKL
from transformers import AutoImageProcessor, AutoModel
from transformers import CLIPTextModelWithProjection, CLIPTokenizer

WORK_DIR = osp.abspath(osp.join(osp.dirname(__file__), "../.."))
if WORK_DIR not in sys.path:
    sys.path.append(WORK_DIR)

# Repo root (where parallel/ lives) on the path so `--tp` can import the TP code
# regardless of the cwd the launcher used.
_REPO_ROOT = osp.dirname(osp.abspath(__file__))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from Mobile_VTON.utils.misc import get_real_path
from Mobile_VTON.models.autoencoders.vae import Decoder

from Mobile_VTON.models.unets.unet_2d_condition_tryon import (
    UNet2DConditionModel as Unet_Tryon
)
from Mobile_VTON.models.unets.unet_2d_condition_garment import (
    UNet2DConditionModel as Unet_Garment
)

from Mobile_VTON.pipelines.tryon_pipeline_full_cat_lowmem import T2IMobilePipelineV1_3_NotLoadingT5_Decoder as TryonPipeline



class VitonHDTestDataset(data.Dataset):
    def __init__(
        self,
        dataroot_path: str,
        phase: Literal["test"] = "test",
        order: Literal["paired", "unpaired"] = "paired",
        size: Tuple[int, int] = (1024, 768),
        image_encoder_path: str = "ckpt/DINO_V2",
        person_image_name: str = "",
    ):
        super().__init__()
        self.dataroot = dataroot_path
        self.phase = phase
        self.height, self.width = size
        self.order = order

        self.descriptions = {}
        with open(os.path.join(dataroot_path, phase, "image_descriptions.txt"), "r") as f:
            for line in f:
                parts = line.strip().split(": ", 1)
                if len(parts) == 2:
                    filename, description = parts
                    self.descriptions[filename] = description

        self.transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
        )
        self.toTensor = transforms.ToTensor()

        if phase == "single_inference":
            pairs_txt = os.path.join(dataroot_path, "single_inference", f"{phase}_pairs.txt")
        else:
            pairs_txt = os.path.join(dataroot_path, f"{phase}_pairs.txt")

        im_names, c_names = [], []
        with open(pairs_txt, "r") as f:
            for line in f.readlines():
                if order == "paired":
                    im_name, _ = line.strip().split()
                    c_name = im_name
                else:
                    im_name, c_name = line.strip().split()
                    
                if person_image_name:
                    im_name = person_image_name
                im_names.append(im_name)
                c_names.append(c_name)


        self.im_names = im_names
        self.c_names = c_names

        self.image_processor = AutoImageProcessor.from_pretrained(image_encoder_path)

    def __getitem__(self, index):
        im_name = self.im_names[index]
        c_name = self.c_names[index]

        cloth_pil = Image.open(os.path.join(self.dataroot, self.phase, "cloth", c_name)).convert("RGB")
        cloth_trim = self.image_processor(images=cloth_pil, return_tensors="pt").pixel_values  # [1,3,h,w]

        # The full_cat pipeline concatenates the garment latent with the person
        # latent, so cloth_pure must share the person's (width, height). The
        # original code assumed VITON-HD inputs already at 1024x768; for
        # arbitrary-size inputs we must resize here or the cat shapes mismatch.
        cloth_pure = self.transform(cloth_pil.resize((self.width, self.height)))  # [-1,1]

        image_pil = Image.open(os.path.join(self.dataroot, self.phase, "image", im_name)).convert("RGB").resize(
            (self.width, self.height)
        )
        image = self.transform(image_pil)  # [-1,1]

        desc = self.descriptions[c_name]
        sample = {
            "im_name": im_name,
            "c_name": c_name,
            "image": image,
            "cloth": cloth_trim,
            "cloth_pure": cloth_pure,
            "caption": "Replace the upper body with " + " ".join(desc.split()[1:]),
            "caption_cloth": desc,
        }
        return sample

    def __len__(self):
        return len(self.im_names)


def pil_to_tensor(img: Image.Image) -> torch.Tensor:
    arr = np.array(img).astype(np.float32) / 255.0
    return torch.from_numpy(arr.transpose(2, 0, 1))


def parse_args():
    p = argparse.ArgumentParser("HSW Try-on: infer all test pairs (full pipeline)")
    p.add_argument("--data_dir", type=str, default="../IDM-VTON/Dataset/zalando")
    p.add_argument("--output_dir", type=str, default="output/infer_test_full")
    p.add_argument("--order", type=str, choices=["paired", "unpaired"], default="paired")
    p.add_argument("--height", type=int, default=1024)
    p.add_argument("--width", type=int, default=768)
    p.add_argument("--test_batch_size", type=int, default=8)
    p.add_argument("--num_inference_steps", type=int, default=28)
    p.add_argument("--guidance_scale", type=float, default=2.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--mixed_precision", type=str, default=None, choices=["no", "fp16", "bf16"])
    p.add_argument("--person_image_name", type=str, default=None)
    p.add_argument("--checkpoint_path", type=str, required=True,)
    p.add_argument("--scheduler_shift", type=float, default=3.0,)

    # 2-Pi tensor parallelism (see parallel/tp_bootstrap.py). When --tp is set the
    # SAME command runs on both Pis with different --rank; the main denoiser's
    # attention is sharded head-parallel across them.
    p.add_argument("--tp", action="store_true",
                   help="enable 2-way tensor parallelism across two Pis")
    p.add_argument("--rank", type=int, default=0, help="TP rank of this process")
    p.add_argument("--world_size", type=int, default=2, help="number of TP ranks")
    p.add_argument("--master_addr", type=str, default="192.168.100.1",
                   help="rank-0 eth0 address for gloo rendezvous")
    p.add_argument("--master_port", type=int, default=29500)
    p.add_argument("--tp_iface", type=str, default="eth0",
                   help="NIC gloo binds to (the direct wired link)")
    # 2-Pi SPATIAL (H-band) parallelism (see parallel/sp_*.py + SPATIAL_TP_PLAN.md).
    # DIFFERENT axis from --tp: weights stay full, the feature maps are split along
    # H across ranks. Splits the whole step (incl. the mem-bound depthwise/GroupNorm/
    # sampler region that --tp must replicate). Mutually exclusive with --tp.
    p.add_argument("--spatial", action="store_true",
                   help="enable 2-way spatial (H-band) parallelism across two Pis")
    # channels-last (NHWC) memory format: big win for the depthwise/pointwise convs
    # on the Pi A76 (2-5x on convs in microbench). GroupNorm is slightly slower in
    # NHWC but tiny vs the conv saving.
    p.add_argument("--channels_last", action="store_true",
                   help="run the denoiser convs in channels-last (NHWC) layout")

    return p.parse_args()


def main():
    args = parse_args()

    # Bring up the TP process group BEFORE Accelerator so accelerate stays a
    # single (non-distributed) process and our gloo group owns the default group
    # that TPAttnProcessor's all_reduce uses.
    if args.tp and args.spatial:
        raise SystemExit("--tp and --spatial are mutually exclusive (pick one axis)")
    tp_rank, tp_world = 0, 1
    if args.tp or args.spatial:
        from parallel.tp_bootstrap import init_tp
        tp_rank, tp_world = init_tp(
            args.rank, args.world_size, args.master_addr, args.master_port, args.tp_iface)
        tag = "sp" if args.spatial else "tp"
        print(f"[{tag}] process group up: rank {tp_rank}/{tp_world} "
              f"(master {args.master_addr}:{args.master_port}, iface {args.tp_iface})",
              file=sys.stderr, flush=True)
    is_writer = (not (args.tp or args.spatial)) or (tp_rank == 0)

    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,
        project_config=ProjectConfiguration(project_dir=args.output_dir),
    )
    if accelerator.is_main_process and args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)

    if args.seed is not None:
        set_seed(args.seed)

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16


    test_dataset = VitonHDTestDataset(
        dataroot_path=args.data_dir,
        phase="test",
        order=args.order,
        size=(args.height, args.width),
        image_encoder_path=args.checkpoint_path + "/image_encoder",
        person_image_name=args.person_image_name,
    )
    # Pi has no GPU: pin_memory is useless and worker procs each hold a copy of
    # the batch, inflating RAM and oversubscribing the 4 cores. Load in-process.
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        shuffle=False,
        batch_size=args.test_batch_size,
        num_workers=0,
        pin_memory=False
    )

    # Scheduler
    noise_scheduler = FlowMatchEulerDiscreteScheduler(
        num_train_timesteps=1000,
        shift=args.scheduler_shift,
    )

    # Tokenizer & Text Encoder
    tokenizer_one = CLIPTokenizer.from_pretrained(args.checkpoint_path, subfolder="tokenizer")
    tokenizer_two = CLIPTokenizer.from_pretrained(args.checkpoint_path, subfolder="tokenizer_2")

    text_encoder_one = CLIPTextModelWithProjection.from_pretrained(args.checkpoint_path, subfolder="text_encoder")
    text_encoder_two = CLIPTextModelWithProjection.from_pretrained(args.checkpoint_path, subfolder="text_encoder_2")

    # VAE encoder（SD3.5-large's vae）
    vae = AutoencoderKL.from_pretrained(args.checkpoint_path, subfolder="vae")

    # VAE decoder
    vd_cfg = get_real_path(args.checkpoint_path + "/vae_decoder/decoder.json")
    vd_ckpt = get_real_path(args.checkpoint_path + "/vae_decoder/decoder.pt")
    with open(vd_cfg, "r") as f:
        vd_cfg_json = json.load(f)
    vae_decoder = Decoder(**vd_cfg_json)
    vae_decoder.load_state_dict(torch.load(vd_ckpt, map_location="cpu"), strict=True)

    # DINOv2
    image_encoder = AutoModel.from_pretrained(args.checkpoint_path, subfolder="image_encoder")

    # garmentnet and tryonnet
    denoiser = Unet_Tryon.from_pretrained(args.checkpoint_path, subfolder="denoiser")
    denoiser_garment = Unet_Garment.from_pretrained(args.checkpoint_path, subfolder="denoiser_garment")

    vae.to(accelerator.device, dtype=weight_dtype)
    vae_decoder.to(accelerator.device, dtype=torch.float32)
    text_encoder_one.to(accelerator.device, dtype=weight_dtype)
    text_encoder_two.to(accelerator.device, dtype=weight_dtype)
    denoiser.to(accelerator.device, dtype=weight_dtype)
    denoiser_garment.to(accelerator.device, dtype=weight_dtype)
    image_encoder.to(accelerator.device, dtype=weight_dtype)

    for m in [vae, vae_decoder, text_encoder_one, text_encoder_two, denoiser, denoiser_garment, image_encoder]:
        m.eval().requires_grad_(False)

    # channels-last: convert the conv-heavy nets to NHWC weights AND coerce their
    # 4D `sample` input to NHWC at forward entry, so the fast layout propagates
    # through every conv (otherwise a contiguous input forces convs back to NCHW).
    if args.channels_last:
        def _to_cl_sample(module, fargs, fkwargs):
            s = fkwargs.get("sample")
            if s is not None and s.dim() == 4:
                fkwargs["sample"] = s.contiguous(memory_format=torch.channels_last)
            return fargs, fkwargs
        for _m in (denoiser, denoiser_garment):
            if _m is not None:
                _m.to(memory_format=torch.channels_last)
                _m.register_forward_pre_hook(_to_cl_sample, with_kwargs=True)
        print("[channels_last] denoiser/garment converted to NHWC", file=sys.stderr, flush=True)

    # TP: shard the main denoiser's attention head-parallel across ranks. Done
    # after .to(device,dtype)/.eval() (slicing clones in-place at the final dtype)
    # and before the pipeline runs. Only the per-step main denoiser is sharded;
    # the garment net stays replicated (it runs once, then gets cached).
    if args.tp:
        from parallel.tp_attention import shard_denoiser_attention_
        from parallel.tp_conv import shard_denoiser_conv_
        sharded, skipped = shard_denoiser_attention_(denoiser, tp_rank, tp_world)
        print(f"[tp] rank {tp_rank}: sharded {len(sharded)} / skipped {len(skipped)} "
              f"denoiser attention modules", file=sys.stderr, flush=True)
        # Stage 1b: channel-parallel the resnet pointwise convs (the heavy bulk).
        cdone, cskip = shard_denoiser_conv_(denoiser, tp_rank, tp_world)
        print(f"[tp] rank {tp_rank}: sharded {len(cdone)} / skipped {len(cskip)} "
              f"denoiser resnet blocks (channel-parallel conv)", file=sys.stderr, flush=True)
        # Stage 2: Megatron-MLP shard the transformer FFN (~48% of params).
        from parallel.tp_ffn import shard_denoiser_ffn_
        fdone, fskip = shard_denoiser_ffn_(denoiser, tp_rank, tp_world)
        print(f"[tp] rank {tp_rank}: sharded {len(fdone)} / skipped {len(fskip)} "
              f"denoiser FFN modules", file=sys.stderr, flush=True)

    # SPATIAL (H-band): patch GroupNorm (stat all_reduce), convs (halo) + samplers
    # (gather-full), and transformer attentions (self: person K/V all_gather; cross:
    # query-parallel). Then wrap the denoiser so its `sample` is scattered into this
    # rank's row band on entry and the output gathered back to full on exit -- the
    # pipeline still sees a normal full-in/full-out denoiser. All non-spatial inputs
    # (text/garment/image embeds, timestep) stay replicated.
    if args.spatial:
        from parallel.sp_groupnorm import shard_denoiser_groupnorm_
        from parallel.sp_conv import shard_denoiser_spatial_conv_
        from parallel.sp_attention import shard_denoiser_spatial_attention_
        from parallel.sp_common import scatter_rows, gather_rows
        # Spatial keeps the MAIN denoiser's weights full (each rank computes all
        # channels for its row band). To keep the model-loading peak under the Pi's
        # RAM, CHANNEL-parallel shard the GARMENT denoiser instead: it runs once, its
        # all_reduced output is full/replicated (exactly what the main self-attn keys
        # need), and halving its weights claws back the headroom spatial loses on the
        # main net. Skippable via SP_NO_GARMENT_TP=1.
        if tp_world > 1 and os.environ.get("SP_NO_GARMENT_TP") != "1":
            from parallel.tp_attention import shard_denoiser_attention_ as _cta
            from parallel.tp_conv import shard_denoiser_conv_ as _ctc
            from parallel.tp_ffn import shard_denoiser_ffn_ as _ctf
            ga, _ = _cta(denoiser_garment, tp_rank, tp_world)
            gc, _ = _ctc(denoiser_garment, tp_rank, tp_world)
            gf, _ = _ctf(denoiser_garment, tp_rank, tp_world)
            print(f"[sp] rank {tp_rank}: garment denoiser channel-TP sharded "
                  f"attn={len(ga)} conv={len(gc)} ffn={len(gf)}", file=sys.stderr, flush=True)
        ngn = shard_denoiser_groupnorm_(denoiser)
        nhalo, nsamp = shard_denoiser_spatial_conv_(denoiser)
        if os.environ.get("SP_SKIP_ATTN") == "1":
            nt, na = 0, 0
            print("[sp] SP_SKIP_ATTN=1 -> attention left replicated (bisect)", file=sys.stderr, flush=True)
        else:
            nt, na = shard_denoiser_spatial_attention_(denoiser)
        print(f"[sp] rank {tp_rank}: spatial-sharded groupnorm={ngn} halo_conv={nhalo} "
              f"samplers={nsamp} transformers={nt} attn={na}", file=sys.stderr, flush=True)

        _orig_denoiser_forward = type(denoiser).forward

        def _spatial_denoiser_forward(sample, *a, **k):
            sample = scatter_rows(sample, tp_rank, tp_world)
            out = _orig_denoiser_forward(denoiser, sample, *a, **k)
            if isinstance(out, tuple):
                return (gather_rows(out[0], tp_rank, tp_world),) + out[1:]
            out.sample = gather_rows(out.sample, tp_rank, tp_world)
            return out
        denoiser.forward = _spatial_denoiser_forward

    # Tile/slice the VAE so the encode step never materializes the full-res
    # feature map at once (big peak-memory win at high resolution).
    try:
        vae.enable_slicing()
        vae.enable_tiling()
    except AttributeError:
        pass

    pipe = TryonPipeline(
        vae=vae,
        vae_decoder=vae_decoder,
        scheduler=noise_scheduler,
        tokenizer=tokenizer_one,
        tokenizer_2=tokenizer_two,
        text_encoder=text_encoder_one,
        text_encoder_2=text_encoder_two,
        image_encoder=image_encoder,
        denoiser=denoiser,
        denoiser_garment=denoiser_garment,
    )

    # Swap the decoder for a tiled wrapper so the final full-res decode (the
    # pipeline's peak-RAM stage) never allocates the whole feature map at once.
    # object.__setattr__ bypasses the diffusers pipeline's module registration.
    # SPATIAL: distribute the decode tiles across ranks (round-robin) + all_reduce.
    _vae_sp = dict(sp_rank=tp_rank, sp_world=tp_world) if args.spatial else {}
    object.__setattr__(pipe, "vae_decoder",
                       TiledVaeDecoder(vae_decoder, tile=32, overlap=8, **_vae_sp))
    print(f"[lowmem] tiled vae_decoder installed: {type(pipe.vae_decoder).__name__}"
          + (f" (spatial tiles across {tp_world} ranks)" if args.spatial else ""),
          file=sys.stderr, flush=True)

    # Drop main()'s own references to vae / image_encoder so that when the
    # pipeline sets self.vae / self.image_encoder = None before the denoise
    # loop, the memory is actually released (pipe keeps the only other ref).
    vae = None
    image_encoder = None
    # With garment caching on, the pipeline frees self.denoiser_garment after
    # the first step; drop our ref too so that free actually releases the RAM.
    denoiser_garment = None
    _free_modules()

    generator = torch.Generator(device=accelerator.device).manual_seed(args.seed) if args.seed is not None else None
    neg_prompt = (
        "deformed, distorted, disfigured, poorly drawn, bad anatomy, wrong anatomy, extra limb, "
        "missing limb, floating limbs, mutated hands and fingers, disconnected limbs, mutation, "
        "mutated, ugly, disgusting, blurry, amputation, NSFW"
    )

    test_loader = accelerator.prepare(test_loader)
    
    pipe_device = accelerator.device
    progress = tqdm(total=len(test_loader), disable=not accelerator.is_local_main_process, desc="Inferring")

    # Batches we will actually run (loop breaks at i >= 4). Once the last one is
    # encoded, the text encoders are dead weight during the heavy denoise loop,
    # so we free them to claw back ~0.8GB (bf16) of peak RAM.
    n_proc = min(len(test_loader), 4)
    text_encoders_freed = False

    with torch.no_grad():
        autocast_dtype = None
        if accelerator.mixed_precision == "fp16":
            autocast_dtype = torch.float16
        elif accelerator.mixed_precision == "bf16":
            autocast_dtype = torch.bfloat16

        for i, batch in enumerate(test_loader):
            if i >= 4:
                break
            B = batch["cloth"].shape[0]

            ip_imgs = torch.cat([batch["cloth"][i] for i in range(B)], dim=0)  # [B,3,h,w]

            prompt = batch["caption"]
            if not isinstance(prompt, list):
                prompt = [prompt] * B
            negative_prompt = [neg_prompt] * B

            prompt_c = batch["caption_cloth"]
            if not isinstance(prompt_c, list):
                prompt_c = [prompt_c] * B
            negative_prompt_c = [neg_prompt] * B 

            with torch.autocast(device_type=pipe_device.type, dtype=autocast_dtype) if autocast_dtype else torch.no_grad():

                (
                    prompt_embeds,
                    negative_prompt_embeds,
                    pooled_prompt_embeds,
                    negative_pooled_prompt_embeds,
                ) = pipe.encode_prompt(
                    prompt=prompt,
                    prompt_2=prompt,
                    num_images_per_prompt=1,
                    do_classifier_free_guidance=True,
                    negative_prompt=negative_prompt,
                    device=pipe_device,
                )


                (
                    prompt_embeds_c,
                    negative_prompt_embeds_c,
                    pooled_prompt_embeds_c,
                    negative_pooled_prompt_embeds_c,
                ) = pipe.encode_prompt(
                    prompt=prompt_c,
                    prompt_2=prompt_c,
                    num_images_per_prompt=1,
                    do_classifier_free_guidance=False,
                    negative_prompt=negative_prompt_c,
                    device=pipe_device,
                )

                # Prompts for every batch we'll process are now encoded into the
                # *_embeds tensors above; the text encoders are no longer needed.
                # Release them before the memory-heavy denoise loop.
                if (i == n_proc - 1) and not text_encoders_freed:
                    text_encoder_one = text_encoder_two = None
                    pipe.text_encoder = pipe.text_encoder_2 = None
                    _free_modules()
                    text_encoders_freed = True

                images = pipe(
                    prompt_embeds=prompt_embeds,
                    negative_prompt_embeds=negative_prompt_embeds,
                    pooled_prompt_embeds=pooled_prompt_embeds,
                    negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
                    num_inference_steps=args.num_inference_steps,
                    generator=generator,
                    height=args.height,
                    width=args.width,
                    guidance_scale=args.guidance_scale,
                    text_embeds_cloth=prompt_embeds_c,
                    negative_text_embeds_cloth=negative_prompt_embeds_c,
                    cloth=batch["cloth_pure"].to(pipe_device), 
                    image=(batch["image"].to(pipe_device) + 1.0) / 2.0, 
                    ip_adapter_image=ip_imgs.to(pipe_device), 
                    device=pipe_device,
                )[0]

            # Both ranks produce identical images (SPMD); only the writer saves
            # so the two Pis don't race on the same output files.
            if is_writer:
                for i, pil_img in enumerate(images):
                    im_name = batch["im_name"][i]
                    c_name = batch["c_name"][i]
                    out_name = f"{im_name[:-4]}_{c_name}"
                    x = pil_to_tensor(pil_img)
                    torchvision.utils.save_image(x, os.path.join(args.output_dir, out_name))

            progress.update(1)

    progress.close()
    if args.tp or args.spatial:
        from parallel.tp_bootstrap import shutdown_tp
        shutdown_tp()
    if is_writer:
        print(f"Done. Saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
