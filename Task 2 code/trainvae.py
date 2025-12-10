# trainvae.py
import os
import time
import argparse
import csv

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image

from vae import VAE


# ----------------- Dataset -----------------


class CelebADataset(Dataset):
    """
    Very simple dataset: loads all *.jpg files from a folder.
    Unsupervised -> returns only images (no labels).
    """

    def __init__(self, root_dir, transform=None, max_images=None):
        self.root_dir = root_dir
        self.transform = transform

        # Collect all jpg files
        files = []
        for fname in os.listdir(root_dir):
            if fname.lower().endswith(".jpg"):
                files.append(os.path.join(root_dir, fname))
        files.sort()
        if max_images is not None:
            files = files[:max_images]

        self.image_paths = files

        if len(self.image_paths) == 0:
            raise RuntimeError(f"No .jpg files found in {root_dir}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        img = Image.open(path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img


# ----------------- Loss -----------------


def vae_loss(recon_x, x, mu, logvar, beta=1.0):
    """
    Reconstruction + KL divergence loss for VAE.
    - recon_x, x are in [0,1], so we use BCE.
    - KL term matches the standard VAE derivation.
    """
    # BCE per batch
    bce = torch.nn.functional.binary_cross_entropy(
        recon_x, x, reduction="sum"
    ) / x.size(0)

    # KL divergence between q(z|x) and N(0, I)
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)

    loss = bce + beta * kl
    return loss, bce, kl


# ----------------- Training loop -----------------


def train(args):
    device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    )
    print(f"Using device: {device}")
    if device.type == "cuda":
        print("GPU name:", torch.cuda.get_device_name(0))

    # Transforms: center crop → resize to 64x64 → tensor in [0,1]
    transform = transforms.Compose(
        [
            transforms.CenterCrop(148),  # works for standard CelebA 178x218
            transforms.Resize(64),
            transforms.ToTensor(),  # scales to [0,1]
        ]
    )

    dataset = CelebADataset(
        root_dir=args.data_dir,
        transform=transform,
        max_images=args.max_images,
    )
    print(f"Total images: {len(dataset)}")

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    model = VAE(latent_dim=args.latent_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    os.makedirs(args.out_dir, exist_ok=True)
    samples_dir = os.path.join(args.out_dir, "samples")
    os.makedirs(samples_dir, exist_ok=True)

    log_path = os.path.join(args.out_dir, "train_log.csv")
    log_file = open(log_path, "w", newline="")
    log_writer = csv.writer(log_file)
    log_writer.writerow(["epoch", "loss", "recon_loss", "kl_loss"])

    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        epoch_recon = 0.0
        epoch_kl = 0.0
        n_batches = 0

        for batch_idx, batch in enumerate(dataloader, start=1):
            batch = batch.to(device, non_blocking=(device.type == "cuda"))

            optimizer.zero_grad()
            recon_batch, mu, logvar = model(batch)
            loss, recon_loss, kl_loss = vae_loss(
                recon_batch, batch, mu, logvar, beta=args.beta
            )
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            epoch_recon += recon_loss.item()
            epoch_kl += kl_loss.item()
            n_batches += 1

            # progress print every 100 batches (and on last batch)
            if batch_idx % 100 == 0 or batch_idx == len(dataloader):
                print(
                    f"Epoch {epoch}/{args.epochs} | "
                    f"Batch {batch_idx}/{len(dataloader)} | "
                    f"Loss {loss.item():.4f}",
                    end="\r",
                )
        print()  # newline after epoch progress

        avg_loss = epoch_loss / n_batches
        avg_recon = epoch_recon / n_batches
        avg_kl = epoch_kl / n_batches

        print(
            f"Epoch [{epoch}/{args.epochs}] "
            f"Loss: {avg_loss:.4f}  Recon: {avg_recon:.4f}  KL: {avg_kl:.4f}"
        )

        # log to CSV for plotting later
        log_writer.writerow([epoch, avg_loss, avg_recon, avg_kl])
        log_file.flush()

        # Save a small grid of original vs reconstructed images
        model.eval()
        with torch.no_grad():
            sample_batch = next(iter(dataloader)).to(
                device, non_blocking=(device.type == "cuda")
            )
            recon_batch, _, _ = model(sample_batch)

        # Stack originals and reconstructions vertically
        grid = torch.cat([sample_batch[:8], recon_batch[:8]], dim=0)
        save_image(
            grid,
            os.path.join(samples_dir, f"recon_epoch_{epoch:03d}.png"),
            nrow=8,
        )

        # Save checkpoint
        ckpt_path = os.path.join(args.out_dir, f"vae_model_epoch_{epoch:03d}.pth")
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "args": vars(args),
            },
            ckpt_path,
        )

    total_time = time.time() - start_time
    log_file.close()
    print(f"Training finished in {total_time/60:.2f} minutes")
    print(f"Log saved to: {log_path}")
    print(f"Checkpoints and samples saved to: {args.out_dir}")


# ----------------- CLI -----------------


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a simple VAE on CelebA (Task 2)"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Folder with CelebA images (e.g. data/celeba/img_align_celeba)",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="outputs",
        help="Where to save logs, checkpoints, and sample images",
    )
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--latent-dim", type=int, default=128)
    parser.add_argument("--beta", type=float, default=1.0, help="KL weight (beta-VAE)")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Optional: limit number of images for quick tests",
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force CPU even if CUDA is available",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
