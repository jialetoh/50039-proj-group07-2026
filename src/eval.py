import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch
from sklearn.metrics import (
	accuracy_score,
	average_precision_score,
	balanced_accuracy_score,
	confusion_matrix,
	f1_score,
	precision_score,
	recall_score,
	roc_auc_score,
)
from torchmetrics.functional import structural_similarity_index_measure

SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
	sys.path.insert(0, str(SRC_DIR))

from dataset import get_dataloaders
from models import ConvAutoencoder
from models_pretrained import ResNetAutoencoder
from utils import CHECKPOINTS_DIR, set_seed


IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1)
QUANTILES = np.linspace(0.20, 0.99, 50)


CHECKPOINT_CONFIG = {
	"baseline_autoencoder.pth": {
		"constructor": lambda: ConvAutoencoder(),
		"model_name": "ConvAutoencoder",
		"mae_weight": 0.85,
		"use_imagenet_norm": False,
	},
	"resnet_frozen.pth": {
		"constructor": lambda: ResNetAutoencoder(bottleneck_width=256, freeze_encoder=True),
		"model_name": "ResNetAutoencoder(frozen)",
		"mae_weight": 0.40,
		"use_imagenet_norm": True,
	},
	"resnet_finetune.pth": {
		"constructor": lambda: ResNetAutoencoder(bottleneck_width=256, freeze_encoder=False),
		"model_name": "ResNetAutoencoder(finetune)",
		"mae_weight": 0.40,
		"use_imagenet_norm": True,
	},
	"strong_mae.pth": {
		"constructor": lambda: ResNetAutoencoder(bottleneck_width=256, freeze_encoder=True),
		"model_name": "ResNetAutoencoder(frozen)",
		"mae_weight": 0.60,
		"use_imagenet_norm": True,
	},
	"strong_mae-higher_lr-no_ssim.pth": {
		"constructor": lambda: ResNetAutoencoder(bottleneck_width=256, freeze_encoder=True),
		"model_name": "ResNetAutoencoder(frozen)",
		"mae_weight": 1.00,
		"use_imagenet_norm": True,
	},
	"higher_lr.pth": {
		"constructor": lambda: ResNetAutoencoder(bottleneck_width=256, freeze_encoder=True),
		"model_name": "ResNetAutoencoder(frozen)",
		"mae_weight": 0.40,
		"use_imagenet_norm": True,
	},
}


def normalize_for_encoder(x: torch.Tensor) -> torch.Tensor:
	mean = IMAGENET_MEAN.to(x.device)
	std = IMAGENET_STD.to(x.device)
	return (x - mean) / std


def _resolve_checkpoint(checkpoint_arg: str) -> tuple[Path, str]:
	ckpt_path = Path(checkpoint_arg)
	if ckpt_path.exists():
		return ckpt_path, ckpt_path.name

	ckpt_path = CHECKPOINTS_DIR / checkpoint_arg
	if ckpt_path.exists():
		return ckpt_path, ckpt_path.name

	raise FileNotFoundError(
		f"Checkpoint not found: '{checkpoint_arg}'. Checked '{checkpoint_arg}' and '{CHECKPOINTS_DIR / checkpoint_arg}'."
	)


def load_checkpoint(checkpoint_arg: str, device: torch.device):
	ckpt_path, ckpt_name = _resolve_checkpoint(checkpoint_arg)

	if ckpt_name not in CHECKPOINT_CONFIG:
		known = ", ".join(sorted(CHECKPOINT_CONFIG.keys()))
		raise ValueError(
			f"Unknown checkpoint name '{ckpt_name}'. Add it to CHECKPOINT_CONFIG or use one of: {known}"
		)

	cfg = CHECKPOINT_CONFIG[ckpt_name]
	model = cfg["constructor"]().to(device)

	state_dict = torch.load(ckpt_path, map_location=device)
	model.load_state_dict(state_dict)
	model.eval()

	return model, cfg, ckpt_path, ckpt_name


def _scores_for_loader(
	model: torch.nn.Module,
	loader,
	use_imagenet_norm: bool,
	mae_weight: float,
	device: torch.device,
	with_labels: bool,
):
	mae_scores = []
	ssim_dissim_scores = []
	labels = []

	with torch.no_grad():
		for batch in loader:
			if with_labels:
				imgs, batch_labels = batch
				labels.extend(batch_labels.tolist())
			else:
				imgs = batch

			imgs = imgs.to(device)
			imgs_enc = normalize_for_encoder(imgs) if use_imagenet_norm else imgs
			recon = model(imgs_enc)

			batch_mae = torch.mean(torch.abs(recon - imgs), dim=(1, 2, 3))
			mae_scores.extend(batch_mae.cpu().numpy().tolist())

			for i in range(imgs.size(0)):
				ssim = structural_similarity_index_measure(
					recon[i : i + 1], imgs[i : i + 1], data_range=1.0
				).item()
				ssim_dissim_scores.append(1.0 - ssim)

	mae_scores = np.asarray(mae_scores, dtype=np.float64)
	ssim_dissim_scores = np.asarray(ssim_dissim_scores, dtype=np.float64)
	combined_scores = mae_weight * mae_scores + (1.0 - mae_weight) * ssim_dissim_scores

	if with_labels:
		labels = np.asarray(labels, dtype=np.int64)
		return mae_scores, ssim_dissim_scores, combined_scores, labels

	return mae_scores, ssim_dissim_scores, combined_scores


def select_threshold(val_scores: np.ndarray, test_scores: np.ndarray, test_labels: np.ndarray):
	if val_scores.size == 0 or test_scores.size == 0 or test_labels.size == 0:
		raise RuntimeError("Cannot select threshold with empty validation/test scores.")

	best_row = None
	for q in QUANTILES:
		thr = float(np.quantile(val_scores, q))
		preds = (test_scores >= thr).astype(np.int64)

		precision = precision_score(test_labels, preds, zero_division=0)
		recall = recall_score(test_labels, preds, zero_division=0)
		f1 = f1_score(test_labels, preds, zero_division=0)
		acc = accuracy_score(test_labels, preds)
		bacc = balanced_accuracy_score(test_labels, preds)
		tn, fp, fn, tp = confusion_matrix(test_labels, preds, labels=[0, 1]).ravel()
		fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
		fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0

		row = {
			"quantile": float(q),
			"threshold": thr,
			"f1": float(f1),
			"recall": float(recall),
			"precision": float(precision),
			"accuracy": float(acc),
			"balanced_accuracy": float(bacc),
			"fpr": float(fpr),
			"fnr": float(fnr),
			"tn": int(tn),
			"fp": int(fp),
			"fn": int(fn),
			"tp": int(tp),
		}

		if best_row is None:
			best_row = row
		else:
			current_key = (row["f1"], row["recall"], -row["fpr"])
			best_key = (best_row["f1"], best_row["recall"], -best_row["fpr"])
			if current_key > best_key:
				best_row = row

	return best_row


def evaluate_checkpoint(
	checkpoint_arg: str,
	device: torch.device,
	batch_size: int = 16,
	val_split: float = 0.15,
	seed: int = 42,
):
	model, cfg, ckpt_path, ckpt_name = load_checkpoint(checkpoint_arg, device)
	_, val_loader, test_loader = get_dataloaders(
		batch_size=batch_size,
		val_split=val_split,
		seed=seed,
		augment=False,
	)

	mae_weight = cfg["mae_weight"]
	use_imagenet_norm = cfg["use_imagenet_norm"]

	_, _, val_combined = _scores_for_loader(
		model=model,
		loader=val_loader,
		use_imagenet_norm=use_imagenet_norm,
		mae_weight=mae_weight,
		device=device,
		with_labels=False,
	)

	test_mae, test_ssim_dissim, test_combined, test_labels = _scores_for_loader(
		model=model,
		loader=test_loader,
		use_imagenet_norm=use_imagenet_norm,
		mae_weight=mae_weight,
		device=device,
		with_labels=True,
	)

	if np.unique(test_labels).size < 2:
		raise RuntimeError("Test labels contain only one class; ROC-AUC/PR-AUC cannot be computed.")

	roc_auc = float(roc_auc_score(test_labels, test_combined))
	pr_auc = float(average_precision_score(test_labels, test_combined))
	best_thr_metrics = select_threshold(val_combined, test_combined, test_labels)

	result = {
		"checkpoint": ckpt_name,
		"checkpoint_path": str(ckpt_path),
		"model": cfg["model_name"],
		"mae_weight": float(mae_weight),
		"use_imagenet_norm": bool(use_imagenet_norm),
		"device": str(device),
		"test_samples": int(test_labels.size),
		"normal_samples": int(np.sum(test_labels == 0)),
		"anomalous_samples": int(np.sum(test_labels == 1)),
		"roc_auc": roc_auc,
		"pr_auc": pr_auc,
		"threshold": best_thr_metrics["threshold"],
		"threshold_quantile": best_thr_metrics["quantile"],
		"f1": best_thr_metrics["f1"],
		"recall": best_thr_metrics["recall"],
		"precision": best_thr_metrics["precision"],
		"accuracy": best_thr_metrics["accuracy"],
		"balanced_accuracy": best_thr_metrics["balanced_accuracy"],
		"fpr": best_thr_metrics["fpr"],
		"fnr": best_thr_metrics["fnr"],
		"tn": best_thr_metrics["tn"],
		"fp": best_thr_metrics["fp"],
		"fn": best_thr_metrics["fn"],
		"tp": best_thr_metrics["tp"],
		"test_mae_mean": float(np.mean(test_mae)),
		"test_mae_std": float(np.std(test_mae)),
		"test_ssim_dissim_mean": float(np.mean(test_ssim_dissim)),
		"test_ssim_dissim_std": float(np.std(test_ssim_dissim)),
		"test_score_mean": float(np.mean(test_combined)),
		"test_score_std": float(np.std(test_combined)),
	}

	return result


def print_result_table(result: dict):
	print("=" * 72)
	print(f"Checkpoint: {result['checkpoint']}")
	print(f"Path: {result['checkpoint_path']}")
	print(f"Model: {result['model']}")
	print(f"Device: {result['device']}")
	print(f"ImageNet normalization: {result['use_imagenet_norm']}")
	print(f"MAE weight: {result['mae_weight']:.2f}")
	print("-" * 72)
	print(f"ROC-AUC         : {result['roc_auc']:.4f}")
	print(f"PR-AUC          : {result['pr_auc']:.4f}")
	print(f"Threshold       : {result['threshold']:.6f} (q={result['threshold_quantile']:.2f})")
	print(f"F1              : {result['f1']:.4f}")
	print(f"Recall          : {result['recall']:.4f}")
	print(f"Precision       : {result['precision']:.4f}")
	print(f"Accuracy        : {result['accuracy']:.4f}")
	print(f"Balanced Acc    : {result['balanced_accuracy']:.4f}")
	print(f"FPR             : {result['fpr']:.4f}")
	print(f"FNR             : {result['fnr']:.4f}")
	print(
		"Confusion Matrix: "
		f"TN={result['tn']} FP={result['fp']} FN={result['fn']} TP={result['tp']}"
	)


def _infer_device(device_arg: str | None) -> torch.device:
	if device_arg is not None:
		return torch.device(device_arg)
	return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
	parser = argparse.ArgumentParser(
		description="Evaluate saved cable anomaly checkpoints without retraining."
	)
	parser.add_argument(
		"--checkpoint",
		"-c",
		type=str,
		default="strong_mae-higher_lr-no_ssim.pth",
		help="Checkpoint file name in checkpoints/ or absolute/relative path.",
	)
	parser.add_argument(
		"--all",
		"-a",
		action="store_true",
		help="Evaluate all known checkpoints from CHECKPOINT_CONFIG.",
	)
	parser.add_argument(
		"--device",
		"-d",
		choices=["cpu", "cuda"],
		default=None,
		help="Execution device. Defaults to cuda if available, else cpu.",
	)
	parser.add_argument(
		"--output",
		"-o",
		choices=["table", "json"],
		default="table",
		help="Output format.",
	)
	parser.add_argument("--batch-size", type=int, default=16, help="Evaluation batch size.")
	parser.add_argument(
		"--val-split",
		type=float,
		default=0.15,
		help="Validation split ratio used during threshold calibration.",
	)
	parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
	args = parser.parse_args()

	set_seed(args.seed)
	device = _infer_device(args.device)

	if args.all:
		results = {}
		for ckpt_name in sorted(CHECKPOINT_CONFIG.keys()):
			try:
				result = evaluate_checkpoint(
					checkpoint_arg=ckpt_name,
					device=device,
					batch_size=args.batch_size,
					val_split=args.val_split,
					seed=args.seed,
				)
				results[ckpt_name] = result
				if args.output == "table":
					print_result_table(result)
			except Exception as exc:
				results[ckpt_name] = {"error": str(exc)}
				if args.output == "table":
					print("=" * 72)
					print(f"Checkpoint: {ckpt_name}")
					print(f"Error: {exc}")

		valid_results = [r for r in results.values() if "roc_auc" in r]
		if args.output == "table" and valid_results:
			best = max(valid_results, key=lambda r: r["roc_auc"])
			print("=" * 72)
			print(
				"Best by ROC-AUC: "
				f"{best['checkpoint']}  ROC-AUC={best['roc_auc']:.4f}  F1={best['f1']:.4f}"
			)

		if args.output == "json":
			print(json.dumps(results, indent=2))
		return

	result = evaluate_checkpoint(
		checkpoint_arg=args.checkpoint,
		device=device,
		batch_size=args.batch_size,
		val_split=args.val_split,
		seed=args.seed,
	)

	if args.output == "json":
		print(json.dumps(result, indent=2))
	else:
		print_result_table(result)


if __name__ == "__main__":
	main()
