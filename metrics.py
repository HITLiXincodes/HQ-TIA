import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import lpips
import pyiqa
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import DataLoader, Dataset

SUPPORTED_EXTENSIONS: Tuple[str, ...] = (
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp",
    ".tif",
    ".tiff",
    ".webp",
)


class PairedImageDataset(Dataset):
    def __init__(
        self,
        ref_dir: Path,
        gen_dir: Path,
        filenames: Sequence[str],
    ) -> None:
        self.ref_dir = ref_dir
        self.gen_dir = gen_dir
        self.filenames = list(filenames)

    def __len__(self) -> int:
        return len(self.filenames)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        filename = self.filenames[index]
        ref_path = self.ref_dir / filename
        gen_path = self.gen_dir / filename

        with Image.open(ref_path) as ref_img_pil:
            ref_img_pil = ref_img_pil.convert("RGB")
            ref_size = ref_img_pil.size
            ref_tensor = TF.to_tensor(ref_img_pil)

        with Image.open(gen_path) as gen_img_pil:
            gen_img_pil = gen_img_pil.convert("RGB")
            if gen_img_pil.size != ref_size:
                gen_img_pil = gen_img_pil.resize(ref_size, Image.Resampling.BICUBIC)
            gen_tensor = TF.to_tensor(gen_img_pil)

        return {
            "filename": filename,
            "ref": ref_tensor,
            "gen": gen_tensor,
        }


def _normalize_device(device: str) -> torch.device:
    requested = device.strip().lower()
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    normalized = torch.device(requested)
    if normalized.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    if normalized.type == "mps":
        if not getattr(torch.backends, "mps", None) or not torch.backends.mps.is_available():
            raise RuntimeError("MPS was requested but is not available.")
    return normalized


def _tensor_to_list(values: torch.Tensor) -> List[float]:
    return [float(v) for v in values.detach().reshape(-1).cpu().tolist()]


def _mean(values: List[float]) -> float:
    if not values:
        return float("nan")
    return float(sum(values) / len(values))


def _collect_filenames(directory: Path) -> List[str]:
    filenames: List[str] = []
    for path in directory.iterdir():
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
            filenames.append(path.name)
    return sorted(filenames)


def _match_filenames(ref_dir: Path, gen_dir: Path) -> List[str]:
    ref_names = set(_collect_filenames(ref_dir))
    gen_names = set(_collect_filenames(gen_dir))

    missing_in_gen = sorted(ref_names - gen_names)
    missing_in_ref = sorted(gen_names - ref_names)

    if missing_in_gen or missing_in_ref:
        details: List[str] = []
        if missing_in_gen:
            details.append(
                f"Missing in generated folder ({len(missing_in_gen)}): {missing_in_gen[:10]}"
            )
        if missing_in_ref:
            details.append(
                f"Missing in reference folder ({len(missing_in_ref)}): {missing_in_ref[:10]}"
            )
        raise ValueError("Filename mismatch between folders. " + " | ".join(details))

    if not ref_names:
        raise ValueError("No supported image files were found in the reference folder.")

    return sorted(ref_names)


def evaluate_image_folders(
    ref_dir: str,
    gen_dir: str,
    device: str = "auto",
    batch_size: int = 8,
    num_workers: int = 0,
    output_json: Optional[str] = None,
    output_txt: Optional[str] = None,
) -> Dict[str, Any]:
    if batch_size <= 0:
        raise ValueError("batch_size must be > 0")
    if num_workers < 0:
        raise ValueError("num_workers must be >= 0")

    ref_path = Path(ref_dir).expanduser().resolve()
    gen_path = Path(gen_dir).expanduser().resolve()

    if not ref_path.is_dir():
        raise ValueError(f"Reference directory does not exist: {ref_path}")
    if not gen_path.is_dir():
        raise ValueError(f"Generated directory does not exist: {gen_path}")

    device_obj = _normalize_device(device)

    ssim_metric = pyiqa.create_metric("ssim", device=str(device_obj))
    psnr_metric = pyiqa.create_metric("psnr", device=str(device_obj))
    fsim_metric = pyiqa.create_metric("fsim", device=str(device_obj))
    dists_metric = pyiqa.create_metric("dists", device=str(device_obj))
    fid_metric = pyiqa.create_metric("fid", device=str(device_obj))

    lpips_metric = lpips.LPIPS(net="alex").to(device_obj).eval()

    filenames = _match_filenames(ref_path, gen_path)
    dataset = PairedImageDataset(
        ref_dir=ref_path,
        gen_dir=gen_path,
        filenames=filenames,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device_obj.type == "cuda"),
        drop_last=False,
    )

    per_metric_values: Dict[str, List[float]] = {
        "ssim": [],
        "psnr": [],
        "lpips_alex": [],
        "fsim": [],
        "dists": [],
    }
    per_image_results: List[Dict[str, Any]] = []

    with torch.inference_mode():
        for batch in loader:
            batch_filenames: List[str] = list(batch["filename"])
            ref_cpu = batch["ref"]
            gen_cpu = batch["gen"]

            ref_tensor = ref_cpu.to(device_obj, non_blocking=True)
            gen_tensor = gen_cpu.to(device_obj, non_blocking=True)

            ssim_vals = _tensor_to_list(ssim_metric(gen_tensor, ref_tensor))
            psnr_vals = _tensor_to_list(psnr_metric(gen_tensor, ref_tensor))
            fsim_vals = _tensor_to_list(fsim_metric(gen_tensor, ref_tensor))
            dists_vals = _tensor_to_list(dists_metric(gen_tensor, ref_tensor))

            lpips_in_gen = gen_tensor * 2.0 - 1.0
            lpips_in_ref = ref_tensor * 2.0 - 1.0
            lpips_vals = _tensor_to_list(lpips_metric(lpips_in_gen, lpips_in_ref))

            for values_key, vals in (
                ("ssim", ssim_vals),
                ("psnr", psnr_vals),
                ("fsim", fsim_vals),
                ("dists", dists_vals),
                ("lpips_alex", lpips_vals),
            ):
                if len(vals) != len(batch_filenames):
                    raise RuntimeError(
                        f"Metric '{values_key}' returned {len(vals)} results for batch size {len(batch_filenames)}."
                    )
                per_metric_values[values_key].extend(vals)

            for idx, filename in enumerate(batch_filenames):
                per_image_results.append(
                    {
                        "filename": filename,
                        "ssim": ssim_vals[idx],
                        "psnr": psnr_vals[idx],
                        "lpips_alex": lpips_vals[idx],
                        "fsim": fsim_vals[idx],
                        "dists": dists_vals[idx],
                    }
                )

        fid_value = float(fid_metric(str(gen_path), str(ref_path)).item())

    mean_results = {
        "ssim": _mean(per_metric_values["ssim"]),
        "psnr": _mean(per_metric_values["psnr"]),
        "lpips_alex": _mean(per_metric_values["lpips_alex"]),
        "fsim": _mean(per_metric_values["fsim"]),
        "dists": _mean(per_metric_values["dists"]),
        "fid": fid_value,
    }

    result: Dict[str, Any] = {
        "ref_dir": str(ref_path),
        "gen_dir": str(gen_path),
        "device": str(device_obj),
        "batch_size": batch_size,
        "num_images": len(filenames),
        "mean": mean_results,
        "per_image": per_image_results,
    }

    if output_json is not None:
        output_path = Path(output_json).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)

    if output_txt is not None:
        output_txt_path = Path(output_txt).expanduser().resolve()
        output_txt_path.parent.mkdir(parents=True, exist_ok=True)
        lines = [
            "Evaluation completed",
            f"Reference folder: {result['ref_dir']}",
            f"Generated folder: {result['gen_dir']}",
            f"Device: {result['device']}",
            f"Num images: {result['num_images']}",
            "Mean metrics:",
        ]
        for key, value in result["mean"].items():
            lines.append(f"  {key}: {value:.6f}")
        with output_txt_path.open("w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")

    return result


def _build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate paired image folders with SSIM, PSNR, LPIPS-Alex, FSIM, DISTS, and FID."
    )
    parser.add_argument("ref_dir", type=str, help="Path to reference image folder")
    parser.add_argument("gen_dir", type=str, help="Path to generated image folder")
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Computation device (e.g., auto, cpu, cuda, cuda:0, mps)",
    )
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for per-image metrics")
    parser.add_argument("--num_workers", type=int, default=0, help="DataLoader worker count")
    parser.add_argument(
        "--output_json",
        type=str,
        default=None,
        help="Optional path to save full results as JSON",
    )
    parser.add_argument(
        "--output_txt",
        type=str,
        default="metrics.txt",
        help="Optional path to save mean metrics as text",
    )
    return parser

def main() -> None:
    # Default to manual settings so running `python metrics.py` needs no CLI args.
    # Set `use_cli_args = True` if you want to temporarily use command-line inputs.
    use_cli_args = False

    ref_dir = "/home/ubuntu/FR_Attack/coarse_images/target/real_AgeDB"
    gen_dir = "/home/ubuntu/FR_Attack/stage1/adaface/test/agedb/fake_agedb"
    device = "cuda:0"
    batch_size = 256
    num_workers = 0
    output_json: Optional[str] = None
    output_txt: Optional[str] = "metrics.txt"

    if use_cli_args:
        parser = _build_argparser()
        args = parser.parse_args()
        ref_dir = args.ref_dir
        gen_dir = args.gen_dir
        device = args.device
        batch_size = args.batch_size
        num_workers = args.num_workers
        output_json = args.output_json
        output_txt = args.output_txt

    results = evaluate_image_folders(
        ref_dir=ref_dir,
        gen_dir=gen_dir,
        device=device,
        batch_size=batch_size,
        num_workers=num_workers,
        output_json=output_json,
        output_txt=output_txt,
    )

    print("Evaluation completed")
    print(f"Reference folder: {results['ref_dir']}")
    print(f"Generated folder: {results['gen_dir']}")
    print(f"Device: {results['device']}")
    print(f"Num images: {results['num_images']}")
    print("Mean metrics:")
    for key, value in results["mean"].items():
        print(f"  {key}: {value:.6f}")

    if output_json:
        output_path = Path(output_json).expanduser().resolve()
        print(f"Saved JSON results to: {output_path}")
    if output_txt:
        output_txt_path = Path(output_txt).expanduser().resolve()
        print(f"Saved text metrics to: {output_txt_path}")


if __name__ == "__main__":
    main()
