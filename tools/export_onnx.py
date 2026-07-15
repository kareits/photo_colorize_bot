"""Export the three models to ONNX. Build-time only.

This is the *only* file in the project allowed to import torch. It runs on the
Docker build stage; the runtime image gets nothing but the resulting .onnx files
and onnxruntime. That boundary is what lets the runtime drop torch, basicsr,
facexlib and modelscope — and with them the numpy<2 / timm<1 / torchvision==0.16
pin cascade that currently forces Python 3.10.

Each export is checked numerically against the original torch model. An export
that does not reproduce torch within tolerance is not written out — a silently
wrong model would be far worse than a failed build.

Reference implementations this mirrors (read them if something looks arbitrary):
  * DDColor         -> ddcolor/pipeline.py, ColorizationPipeline.process
  * RestoreFormer++ -> RestoreFormer/RestoreFormer.py, RestoreFormer.enhance
"""
from __future__ import annotations

import argparse
import logging
import sys
import tempfile
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from onnxruntime.quantization import QuantType, quantize_dynamic
from onnxruntime.quantization.shape_inference import quant_pre_process

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger("export")

# Tolerance for the torch-vs-onnxruntime comparison, as a *fraction of the output's
# own scale*. Comparing raw absolute differences would be meaningless here: the three
# models emit wildly different ranges — Real-ESRGAN [0, 1], RestoreFormer++ [-1, 1],
# DDColor raw Lab chroma at roughly +/-110 — so the same absolute error means something
# different for each. Normalising makes one threshold honest across all of them.
#
# 0.1% of the output range is far below anything an eye can see, while still catching
# an export that genuinely diverged.
TOLERANCE = 1e-3

FACE_SIZE = 512     # both DDColor and RestoreFormer++ take a fixed 512x512 input
COLOR_SIZE = 512


def _verify(
    torch_model, onnx_path: Path, sample: torch.Tensor, take_first: bool = False
) -> tuple[float, float]:
    """Run the same input through torch and onnxruntime.

    Returns (absolute max difference, difference relative to the output's range).
    The relative figure is the one worth judging by — see TOLERANCE.
    """
    with torch.no_grad():
        expected = torch_model(sample)
    if take_first:
        expected = expected[0]
    expected = expected.cpu().numpy()

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    actual = session.run(None, {session.get_inputs()[0].name: sample.numpy()})[0]

    if expected.shape != actual.shape:
        raise SystemExit(f"{onnx_path.name}: shape mismatch torch={expected.shape} onnx={actual.shape}")

    absolute = float(np.abs(expected - actual).max())
    output_range = float(expected.max() - expected.min())
    return absolute, absolute / max(output_range, 1e-6)


def _export(torch_model, sample: torch.Tensor, out_path: Path, dynamic_hw: bool, take_first: bool) -> None:
    """Export to ONNX, then refuse to keep the file if it does not match torch."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch_model.eval()

    dynamic_axes = None
    if dynamic_hw:
        # Real-ESRGAN is fed arbitrary tiles, so its spatial dims must stay free.
        dynamic_axes = {"input": {2: "height", 3: "width"}, "output": {2: "height", 3: "width"}}

    torch.onnx.export(
        torch_model,
        sample,
        str(out_path),
        opset_version=17,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes=dynamic_axes,
        do_constant_folding=True,
    )

    absolute, relative = _verify(torch_model, out_path, sample, take_first=take_first)
    if relative > TOLERANCE:
        out_path.unlink(missing_ok=True)
        raise SystemExit(
            f"{out_path.name}: onnx deviates from torch by {relative:.2%} of the output range "
            f"(abs {absolute:.2e}, limit {TOLERANCE:.2%})"
        )

    size_mb = out_path.stat().st_size / 1e6
    log.info(
        "%s: ok, %.4f%% of output range (abs %.2e), %.0f MB",
        out_path.name, relative * 100, absolute, size_mb,
    )


def _quantize(torch_model, fp32_path: Path, sample: torch.Tensor, take_first: bool) -> None:
    """Produce an int8 sibling of an fp32 export.

    The server has 3.8 GB of RAM with ~1 GB already spent on the TTS service, so
    DDColor-large (912 MB of fp32 weights) does not comfortably fit. Quantisation
    is what makes the large model viable at all here, not a mere speed tweak.

    Dynamic quantisation only touches MatMul/Gemm, not Conv — which sounds fatal
    for a convolutional net until you notice ConvNeXt keeps most of its
    parameters in the pointwise 1x1 layers, which torch emits as Linear/MatMul.
    The transformer decoder quantises too.

    Unlike the fp32 export, this is *not* gated on matching torch: int8 changes
    the numbers by construction. We measure the deviation and report it; whether
    it is acceptable is a question for the eyes in phase 5, not for a threshold
    here.
    """
    int8_path = fp32_path.with_name(fp32_path.stem + "_int8.onnx")

    # Fold the graph before quantising. DDColor's final block uses spectral
    # normalisation, so its weights are *computed* at inference time (a Div of two
    # tensors) rather than stored as constants — and the quantiser rejects that:
    # "Expected ... to be an initializer". Constant folding evaluates those
    # subgraphs ahead of time, turning the results back into plain initializers
    # the quantiser can handle.
    with tempfile.TemporaryDirectory() as tmp:
        folded = Path(tmp) / "folded.onnx"
        quant_pre_process(str(fp32_path), str(folded), skip_symbolic_shape=False)
        quantize_dynamic(
            model_input=str(folded),
            model_output=str(int8_path),
            weight_type=QuantType.QInt8,
        )

    absolute, relative = _verify(torch_model, int8_path, sample, take_first=take_first)
    fp32_mb = fp32_path.stat().st_size / 1e6
    int8_mb = int8_path.stat().st_size / 1e6
    log.info(
        "%s: %.0f MB -> %.0f MB (x%.1f smaller), deviates %.2f%% of output range (abs %.3f) "
        "— judge this one by eye, not by the number",
        int8_path.name, fp32_mb, int8_mb, fp32_mb / max(int8_mb, 1e-9), relative * 100, absolute,
    )


# --------------------------------------------------------------------------- #
# Real-ESRGAN x2plus. Plain RRDBNet — the same architecture the current
# pipeline.py builds by hand, so there is nothing subtle here.
# --------------------------------------------------------------------------- #
def export_realesrgan_compact(weights: Path, out_dir: Path) -> None:
    """realesr-general-x4v3: the fast upscaler.

    RealESRGAN_x2plus is 23 residual-in-residual blocks (~16.7M parameters) and is
    simply expensive on a CPU — roughly 27 s per output megapixel, which dominates the
    whole pipeline. This one is a plain VGG-style stack (~1.2M parameters, 5 MB) built
    for exactly this situation.

    It only comes at x4. We upscale by 4 and resample back down to 2, which supersamples
    rather than loses anything — see imaging.upscale_tiled's `downscale_to` argument.
    """
    from tools.vendor.srvgg_arch import SRVGGNetCompact

    model = SRVGGNetCompact(
        num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=32, upscale=4, act_type="prelu"
    )
    state = torch.load(weights, map_location="cpu")
    model.load_state_dict(state.get("params") or state["params_ema"], strict=True)

    sample = torch.rand(1, 3, 64, 64)
    _export(model, sample, out_dir / "realesrgan_compact_x4.onnx", dynamic_hw=True, take_first=False)


def export_realesrgan(weights: Path, out_dir: Path) -> None:
    # Vendored rather than imported from basicsr — see tools/vendor/arch_util.py
    # for why installing basicsr would undo the whole point of this refactor.
    from tools.vendor import RRDBNet

    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
    state = torch.load(weights, map_location="cpu")
    # Real-ESRGAN checkpoints keep the weights under 'params_ema' (falling back to 'params').
    model.load_state_dict(state.get("params_ema") or state["params"], strict=True)

    # A small tile is enough to validate the graph; H/W stay dynamic anyway.
    sample = torch.rand(1, 3, 64, 64)
    _export(model, sample, out_dir / "realesrgan_x2plus.onnx", dynamic_hw=True, take_first=False)


# --------------------------------------------------------------------------- #
# DDColor. Takes a greyscale image carried as RGB in [0, 1] and returns the two
# Lab chroma channels — it never sees or reproduces luminance. That is why the
# runtime can keep the original full-resolution L and only upsample ab.
# --------------------------------------------------------------------------- #
def export_ddcolor(weights: Path, out_dir: Path, model_size: str, quantize: bool, repo: Path) -> None:
    sys.path.insert(0, str(repo))
    from ddcolor import DDColor, build_ddcolor_model

    model = build_ddcolor_model(
        DDColor,
        model_path=str(weights),
        input_size=COLOR_SIZE,
        model_size=model_size,
        device=torch.device("cpu"),
    )

    sample = torch.rand(1, 3, COLOR_SIZE, COLOR_SIZE)
    out_path = out_dir / f"ddcolor_{model_size}.onnx"
    _export(model, sample, out_path, dynamic_hw=False, take_first=False)
    if quantize:
        _quantize(model, out_path, sample, take_first=False)


# --------------------------------------------------------------------------- #
# RestoreFormer++. Restores one aligned 512x512 face crop. Weights live under a
# 'vqvae.' prefix in the checkpoint, and the module returns a tuple whose first
# element is the image — both quirks are handled here so the runtime does not
# have to know about them.
# --------------------------------------------------------------------------- #
class _FirstOutput(torch.nn.Module):
    """Unwrap the (image, ...) tuple so the ONNX graph has a single output."""

    def __init__(self, inner: torch.nn.Module):
        super().__init__()
        self.inner = inner

    def forward(self, x):
        return self.inner(x)[0]


def export_restoreformer(weights: Path, out_dir: Path, repo: Path) -> None:
    sys.path.insert(0, str(repo))
    from RestoreFormer.modules.vqvae.vqvae_arch import VQVAEGANMultiHeadTransformer

    # head_size=4, ex_multi_scale_num=1 is what upstream uses for the '++' variant.
    inner = VQVAEGANMultiHeadTransformer(head_size=4, ex_multi_scale_num=1)

    state = torch.load(weights, map_location="cpu")["state_dict"]
    state = {k.removeprefix("vqvae."): v for k, v in state.items()}
    missing, unexpected = inner.load_state_dict(state, strict=False)
    # Upstream loads with strict=False, so some keys are expected to be absent.
    # Anything missing on the *encoder/decoder* side, though, means a broken load.
    log.info("RestoreFormer++: %d missing / %d unexpected keys", len(missing), len(unexpected))

    model = _FirstOutput(inner)

    # Input is normalised to [-1, 1], so sample from that range rather than [0, 1].
    sample = torch.rand(1, 3, FACE_SIZE, FACE_SIZE) * 2.0 - 1.0
    _export(model, sample, out_dir / "restoreformer.onnx", dynamic_hw=False, take_first=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights-dir", type=Path, default=Path("weights"))
    parser.add_argument("--out-dir", type=Path, default=Path("onnx"))
    parser.add_argument(
        "--repos-dir",
        type=Path,
        default=Path("vendor_repos"),
        help="Where DDColor/ and RestoreFormerPlusPlus/ are cloned.",
    )
    parser.add_argument(
        "--only",
        choices=["realesrgan", "realesrgan-compact", "ddcolor-tiny", "ddcolor-large", "restoreformer"],
        action="append",
        help="Export just these. Repeatable. Default: all of them.",
    )
    args = parser.parse_args()

    # DDColor ships in two sizes and we export both on purpose: 'tiny' fits the
    # server's memory outright, 'large' only fits once quantised. Phase 5 decides
    # between them by looking at the pictures.
    wanted = args.only or [
        "realesrgan", "realesrgan-compact", "ddcolor-tiny", "ddcolor-large", "restoreformer",
    ]
    torch.set_grad_enabled(False)

    ddcolor_repo = args.repos_dir / "DDColor"
    restoreformer_repo = args.repos_dir / "RestoreFormerPlusPlus"

    if "realesrgan" in wanted:
        export_realesrgan(args.weights_dir / "RealESRGAN_x2plus.pth", args.out_dir)
    if "realesrgan-compact" in wanted:
        export_realesrgan_compact(args.weights_dir / "realesr-general-x4v3.pth", args.out_dir)
    if "ddcolor-tiny" in wanted:
        export_ddcolor(
            args.weights_dir / "ddcolor_paper_tiny.pth", args.out_dir, "tiny",
            quantize=False, repo=ddcolor_repo,
        )
    if "ddcolor-large" in wanted:
        # Only the large model actually needs int8 to be deployable at all.
        export_ddcolor(
            args.weights_dir / "ddcolor_modelscope.pth", args.out_dir, "large",
            quantize=True, repo=ddcolor_repo,
        )
    if "restoreformer" in wanted:
        export_restoreformer(args.weights_dir / "RestoreFormer++.ckpt", args.out_dir, restoreformer_repo)

    log.info("done: %s", ", ".join(sorted(p.name for p in args.out_dir.glob("*.onnx"))))


if __name__ == "__main__":
    main()
