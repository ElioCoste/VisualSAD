import torch

from loss import v8DetectionLoss, ContrastiveLoss, AVLoss
from main_model import MainModel

from dataset import AVADataset, AVADataLoader
from config import resnet_cfg, resnet_input_shape, T, N_MFCC, NUM_CLASSES, C, H, W


class Trainer:
    def __init__(self, model, optimizer, scheduler, device):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device

        self.contrastive_loss = ContrastiveLoss()
        self.av_loss = AVLoss()
        self.detection_loss = v8DetectionLoss(model, tal_topk=10)

    def do_one_epoch(self, dataloader):
        self.model.train()
        total_loss = 0.0
        for i, (audio, video, targets, labels) in enumerate(dataloader):
            # Audio: (B, 4T, N_MFCC)
            # Video: (B, T, C, H, W)
            # Targets: (B, T, max_speakers)
            # Labels: (B, T, max_speakers, 4)

            audio = audio.to(self.device)
            video = video.to(self.device)
            targets = targets.to(self.device)

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
            active_frames = targets.sum(dim=-1) > 0  # (B, T)
            loss_nce = self.contrastive_loss(
                audio_features, video_features, active_frames)

            # Multimodal fusion
            fused_features = self.model.forward_fusion(
                video_features, audio_features)
            del video_features
            # Compute audio-visual loss
            loss_av = self.av_loss(
                audio_features, fused_features, targets, labels)
            del audio_features

            # Detection head
            head_out = self.model.forward_head(fused_features)
            del fused_features
            
            # Compute detection loss
            loss_det, _ = self.detection_loss(head_out, targets, labels)
            # Weighted sum of detection loss:
            loss_det = self.lambda_box * \
                loss_det[:, 0] + self.lambda_cls * \
                loss_det[:, 1] + self.lambda_dfl * loss_det[:, 2]

            # Total loss
            total_loss = self.lambda_nce * loss_nce + \
                self.lambda_av * loss_av + self.lambda_det * loss_det

            # Backward pass
            total_loss.backward()
            self.optimizer.step()
            self.scheduler.step()
            total_loss += total_loss.item()

        return total_loss / len(dataloader)

    def do_train(self, dataloader, epochs, save_dir):
        for epoch in range(epochs):
            total_loss = self.do_one_epoch(dataloader)
            print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss:.4f}")
            # Save model checkpoint
            torch.save(self.model.state_dict(),
                       f"{save_dir}/model_epoch_{epoch+1}.pth")


def main():
    train_dataset = AVADataset(
        "train", N_MFCC,
        C, H, W, T
    )

    train_loader = AVADataLoader(
        train_dataset, batch_size=2, shuffle=True, num_workers=0)

    model = MainModel(
        resnet_cfg, resnet_input_shape,
        T, N_MFCC, NUM_CLASSES, max_det=10)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=10, gamma=0.1)

    trainer = Trainer(model, optimizer, scheduler, device)
    trainer.do_train(train_loader, epochs=50, save_dir="checkpoints")


if __name__ == "__main__":
    main()
