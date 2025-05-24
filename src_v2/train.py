import torch
from tqdm import tqdm

from loss import v8DetectionLoss, ContrastiveLoss, AVLoss
from main_model import MainModel

from dataset import AVADataset, AVADataLoader
from config import T, N_MFCC, NUM_CLASSES, C, H, W


class Trainer:
    def __init__(self, model, optimizer, scheduler, device):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device

        self.contrastive_loss = ContrastiveLoss()
        self.av_loss = AVLoss()
        self.detection_loss = v8DetectionLoss(model, tal_topk=10)

        self.lambda_nce = 0.1
        self.lambda_av = 0.1
        self.lambda_det = 1.0
        self.lambda_box = 1.0
        self.lambda_cls = 1.0
        self.lambda_dfl = 1.0

    def do_one_epoch(self, dataloader):
        self.model.train()
        total_loss = 0.0
        bar = tqdm(range(len(dataloader)), desc="Training")
        for i, (audio, video, targets, bboxes) in enumerate(dataloader):
            # Audio: (B, 4T, N_MFCC)
            # Video: (B, T, C, H, W)
            # Targets: (B, T, max_speakers)
            # Labels: (B, T, max_speakers, 4)

            audio = audio.to(self.device)
            video = video.to(self.device)
            targets = targets.to(self.device)
            bboxes = bboxes.to(self.device)

            # Forward pass

            self.optimizer.zero_grad()

            # Audio and visual encoders to get embeddings
            audio_features = self.model.forward_audio_encoder(audio)
            del audio
            video_features = self.model.forward_visual_encoder(video)
            del video

            # Compute contrastive loss
            # Select the audio and video features at frames where at least one
            # speaker is active (i.e., the sum of the target labels is greater than 0)
            # active_frames = bboxes.sum(dim=-1) > 0  # (B, T)
            loss_nce = 0
            # self.contrastive_loss(
            #     audio_features, video_features, active_frames)

            # Multimodal fusion
            fused_features = self.model.forward_fusion(
                video_features, audio_features)

            # Detection head
            head_out = self.model.forward_head(fused_features)

            # Reshape targets to match the expected input shape for the detection loss
            # Get the targets that are not padded
            non_padded = targets >= 0
            batch_idx = torch.arange(
                targets.size(0)*targets.size(1), device=targets.device)
            batch_idx = batch_idx.repeat_interleave(targets.size(2))
            targets = targets[non_padded].flatten()
            batch_idx = batch_idx[non_padded.flatten()]
            bboxes = bboxes[non_padded].flatten(end_dim=-2)
            batch = {
                "batch_idx": batch_idx,
                "cls": targets,
                "bboxes": bboxes
            }

            # Compute detection loss
            loss_det, fg_mask = \
                self.detection_loss(head_out, batch)
            # Weighted sum of detection loss:
            loss_det = self.lambda_box * \
                loss_det[0] + self.lambda_cls * \
                loss_det[1] + self.lambda_dfl * loss_det[2]
            # Compute AV loss
            loss_av = self.av_loss(
                audio_features, fused_features, fg_mask)
            # Total loss
            total_loss = self.lambda_nce * loss_nce + \
                self.lambda_av * loss_av + self.lambda_det * loss_det

            # Backward pass
            total_loss.backward()
            self.optimizer.step()
            self.scheduler.step()
            total_loss += total_loss.item()

            # Print loss in the progress bar
            bar.set_postfix(
                loss=total_loss.item(), av=loss_av.item(), det=loss_det.item()
            )
            bar.update(1)
        bar.close()
        return total_loss / len(dataloader)

    def do_train(self, train_loader, val_loader, epochs, save_dir):
        train_loss = []
        val_loss = []
        for epoch in range(epochs):
            print(f"Epoch {epoch+1}/{epochs}")
            loss_ = self.do_one_epoch(train_loader)
            print(f"Epoch {epoch+1}/{epochs}, Loss: {loss_:.4f}")
            # Save model checkpoint
            torch.save(self.model.state_dict(),
                       f"{save_dir}/model_epoch_{epoch+1}.pth")
            if epoch % 2 == 0:
                # Validate the model
                self.model.eval()
                with torch.no_grad():
                    val_loss_ = self.do_one_epoch(val_loader)
                    print(f"Validation Loss: {val_loss_:.4f}\n")
                    val_loss.append(val_loss_)
            train_loss.append(loss_)

        # Save training and validation loss
        with open(f"{save_dir}/train_loss.txt", "w") as f:
            for l in train_loss:
                f.write(f"{l}\n")
        with open(f"{save_dir}/val_loss.txt", "w") as f:
            for l in val_loss:
                f.write(f"{l}\n")


def main():
    print("Loading dataset...")
    train_dataset = AVADataset(
        "train", N_MFCC,
        C, H, W, T
    )
    train_loader = AVADataLoader(
        train_dataset, batch_size=2, shuffle=True, num_workers=0)
    val_dataset = AVADataset(
        "val", N_MFCC,
        C, H, W, T
    )
    val_loader = AVADataLoader(
        val_dataset, batch_size=2, shuffle=False, num_workers=0)
    print("Done.")

    print("Initializing model...")
    model = MainModel(
        T, C, H, W, N_MFCC, NUM_CLASSES, max_det=10)
    print("Done.")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on device: {device}")
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=10, gamma=0.1)

    trainer = Trainer(model, optimizer, scheduler, device)

    print("Starting training...")
    trainer.do_train(train_loader, val_loader,
                     epochs=50, save_dir="checkpoints")


if __name__ == "__main__":
    main()
