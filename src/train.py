import torch
import torch.optim as optim
from torchmetrics.image import StructuralSimilarityIndexMeasure

from dataset import get_dataloaders
from masking import generate_cable_mask, masked_mae_loss, masked_ssim_loss
from models import ConvAutoencoder
from utils import CHECKPOINTS_DIR, set_seed


NUM_EPOCHS = 50
LR = 1e-3
BATCH_SIZE = 16
VAL_SPLIT = 0.15
SEED = 42
MAE_WEIGHT = 0.85
USE_MASKING = True


def main():
	set_seed(SEED)
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	print(f"Using device: {device}")

	train_loader, val_loader, _ = get_dataloaders(
		batch_size=BATCH_SIZE,
		val_split=VAL_SPLIT,
		seed=SEED,
	)

	model = ConvAutoencoder().to(device)
	mae_criterion = torch.nn.L1Loss()  # used only when USE_MASKING=False
	optimizer = optim.Adam(model.parameters(), lr=LR)
	ssim_fn = StructuralSimilarityIndexMeasure(data_range=1.0, kernel_size=11).to(device)

	train_losses = []
	val_losses = []

	for epoch in range(1, NUM_EPOCHS + 1):
		model.train()
		running_loss = 0.0
		for imgs in train_loader:
			imgs = imgs.to(device)
			optimizer.zero_grad()
			recon = model(imgs)
			if USE_MASKING:
				mask = generate_cable_mask(imgs)
				loss_mae = masked_mae_loss(recon, imgs, mask)
				loss_ssim = masked_ssim_loss(recon, imgs, mask)
			else:
				loss_mae = mae_criterion(recon, imgs)
				loss_ssim = 1.0 - ssim_fn(recon, imgs)
			loss = MAE_WEIGHT * loss_mae + (1.0 - MAE_WEIGHT) * loss_ssim
			loss.backward()
			optimizer.step()
			running_loss += loss.item() * imgs.size(0)
		train_loss = running_loss / len(train_loader.dataset)

		model.eval()
		running_val = 0.0
		with torch.no_grad():
			for imgs in val_loader:
				imgs = imgs.to(device)
				recon = model(imgs)
				if USE_MASKING:
					mask = generate_cable_mask(imgs)
					val_mae = masked_mae_loss(recon, imgs, mask)
					val_ssim = masked_ssim_loss(recon, imgs, mask)
				else:
					val_mae = mae_criterion(recon, imgs)
					val_ssim = 1.0 - ssim_fn(recon, imgs)
				val_loss = MAE_WEIGHT * val_mae + (1.0 - MAE_WEIGHT) * val_ssim
				running_val += val_loss.item() * imgs.size(0)
		val_loss = running_val / len(val_loader.dataset)

		train_losses.append(train_loss)
		val_losses.append(val_loss)

		if epoch % 10 == 0 or epoch == 1:
			print(
				f"Epoch {epoch:3d}/{NUM_EPOCHS}  "
				f"train_loss={train_loss:.6f}  val_loss={val_loss:.6f}"
			)

	print("Training complete.")

	CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
	ckpt_name = "masked_autoencoder.pth" if USE_MASKING else "baseline_autoencoder.pth"
	ckpt_path = CHECKPOINTS_DIR / ckpt_name
	torch.save(model.state_dict(), ckpt_path)
	print(f"Checkpoint saved to {ckpt_path}")


if __name__ == "__main__":
	main()
