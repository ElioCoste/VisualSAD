import torch
import torch.nn as nn


class AudioEncoder(nn.Module):
    """
    CNN feature extractor for audio data.
    """

    def __init__(self, input_shape, output_dim, n_filters=32):
        """
        Audio feature extractor for the VisualSAD model.

        input_shape: tuple
            Shape of the input audio data (4T, C_mel)
        output_dim: int
            Dimension of the output embedding
        """
        super(AudioEncoder, self).__init__()
        # Input shape is (batch_size, 4T, C_mel)
        # where T is the number of time frames and C_mel is the number of Mel bands.
        self.input_shape = input_shape
        self.output_dim = output_dim
        self.n_filters = n_filters
        T = input_shape[0] // 4
        C_mel = input_shape[1]

        # Convolutional layers with batch normalization
        # The convolutions have to preserve the time resolution
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, n_filters, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm2d(n_filters),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(n_filters, n_filters*2,
                      kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(n_filters*2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(n_filters*2, n_filters*3,
                      kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(n_filters*3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

        self.conv_layers = nn.Sequential(
            self.conv1,
            self.conv2,
            self.conv3
        )

        self.fc = nn.Linear(
            n_filters*3 * (T // 2) * (C_mel // 8), output_dim * T)

    def forward(self, x):
        """
        The input is a Mel-spectrogram of shape (batch_size, 4T, C_mel)
        where T is the number of time frames and C_mel is the number of Mel bands.

        Use the 4T frames of audio data to compute T embeddings of size output_dim.
        The output is a tensor of shape (batch_size, T, output_dim)
        """
        # Reshape the input to (batch_size, 1, 4T, C_mel)
        # print("Forward pass through the audio encoder")
        # print(f"Input shape before reshaping: {x.shape}")
        x = x.view(x.size(0), 1, self.input_shape[0], self.input_shape[1])
        # print(f"Input shape after reshaping: {x.shape}")
        for layer in self.conv_layers:
            x = layer(x)
            # print(f"Shape after {layer}: {x.shape}")
        # The output shape of the conv layers is
        # (batch_size, n_filters*4, T, C_mel//8)

        # Reshape to (batch_size, T*n_filters*4*C_mel//4)
        x = x.view(x.size(0), -1)
        # print(f"Shape after flattening: {x.shape}")
        # Apply a fully connected layer to get the final embedding
        # The output shape is (batch_size, T*output_dim)
        x = self.fc(x)
        # print(f"Shape after fully connected layer: {x.shape}")

        # Reshape to (batch_size, T, output_dim)
        x = x.view(x.size(0), -1, self.output_dim)
        # print(f"Output shape: {x.shape}")
        return x


class VisualEncoder(nn.Module):
    """
    CNN feature extractor for visual data.
    Processes T frames of size H x W for each speaker.
    """

    def __init__(self, input_shape, output_dim, n_filters=16):
        """
        Visual feature extractor for the VisualSAD model.

        input_shape: tuple
            Shape of the input visual data (C, T, H, W)
        output_dim: int
            Dimension of the output embedding
        """
        super(VisualEncoder, self).__init__()
        self.C, self.T, self.H, self.W = input_shape
        self.output_dim = output_dim  # Embedding dimension

        self.conv1 = nn.Sequential(
            nn.Conv3d(self.C, n_filters*self.C,
                      kernel_size=7, stride=1, padding=3),
            nn.BatchNorm3d(n_filters*self.C),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1, 4, 4), stride=(1, 4, 4))
        )
        self.conv2 = nn.Sequential(
            nn.Conv3d(n_filters*self.C, 2*n_filters*self.C,
                      kernel_size=5, stride=1, padding=2),
            nn.BatchNorm3d(2*n_filters*self.C),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
        )
        self.conv3 = nn.Sequential(
            nn.Conv3d(2*n_filters*self.C, 4*n_filters*self.C,
                      kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(4*n_filters*self.C),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
        )

        self.conv_layers = nn.Sequential(
            self.conv1,
            self.conv2,
            self.conv3
        )

        # Fully connected output layer
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(
            4 * n_filters * self.T * (self.H // 16) * (self.W // 16), output_dim*self.T)

    def forward(self, x):
        """
        Forward pass through the feature extractor.
        x: tensor of shape (batch_size, C, T, H, W)
        Returns:
            x: tensor of shape (batch_size, T, output_dim)
                Embedding for each time frame
        """
        # print("Forward pass through the visual encoder")
        # print(f"Input shape {x.shape}")

        # Apply the convolutional layers
        for layer in self.conv_layers:
            x = layer(x)
            # print(f"Shape after {layer}: {x.shape}")

        # Flatten the features
        x = self.flatten(x)
        # print(f"Shape after flattening: {x.shape}")

        # Apply the fully connected layer to get the final embedding
        # for each time frame
        x = self.fc(x)
        # print(f"Shape after fully connected layer: {x.shape}")
        # Reshape to (batch_size, T, output_dim)
        x = x.view(x.size(0), -1, self.output_dim)
        # print(f"Output shape: {x.shape}")
        return x


class AudioVisualFusion(nn.Module):
    """
    Fusion module for combining audio and visual embeddings.
    Simple concatenation of the two embeddings followed by a fully connected layer
    is used here, but other fusion methods can be implemented (e.g., cross-attention).
    """

    def __init__(self, embedding_dim_audio, embedding_dim_visual, output_dim):
        """
        Fusion module for combining audio and visual embeddings.
        embedding_dim_audio: int
            Dimension of the audio embedding
        embedding_dim_visual: int
            Dimension of the visual embedding

        output_dim: int
            Dimension of the output embedding
            """
        super(AudioVisualFusion, self).__init__()
        self.embedding_dim_audio = embedding_dim_audio
        self.embedding_dim_visual = embedding_dim_visual
        self.output_dim = output_dim
        self.fc = nn.Linear(
            embedding_dim_audio + embedding_dim_visual, output_dim)

    def forward(self, audio_embedding, visual_embedding):
        """
        Forward pass through the fusion module.
        audio_embedding: tensor
            Audio embedding of shape (batch_size, embedding_dim_audio)
        visual_embedding: tensor
            Visual embedding of shape (batch_size, n_speakers, embedding_dim_visual)

        Returns:
            x: tensor
                output embedding of shape (batch_size, n_speakers, output_dim)
        """
        # Reshape the visual embedding to (batch_size*T, embedding_dim_visual)
        batch_size = audio_embedding.size(0)
        
        # print("Forward pass through the fusion module")
        # print(f"Audio embedding shape: {audio_embedding.shape}")
        # print(f"Visual embedding shape: {visual_embedding.shape}")
        visual_embedding = visual_embedding.view(
            visual_embedding.size(0) * visual_embedding.size(1), -1)
        audio_embedding = audio_embedding.view(
            audio_embedding.size(0) * audio_embedding.size(1), -1)
        # print(
        #    f"Visual embedding shape after reshaping: {visual_embedding.shape}")
        # print(
        #    f"Audio embedding shape after reshaping: {audio_embedding.shape}")

        # Concatenate the audio and visual embeddings along the last dimension
        x = torch.cat((audio_embedding, visual_embedding), dim=-1)
        # print(f"Shape after concatenation: {x.shape}")

        # Apply the fully connected layer to get the final embedding
        x = self.fc(x)
        # print(f"Shape after fully connected layer: {x.shape}")

        # Reshape the output to (batch_size, output_dim)
        x = x.view(batch_size, -1, self.output_dim)
        # print(f"Output shape: {x.shape}")
        return x


class TemporalModel(nn.Module):
    """
    Temporal model for processing the audio and visual embeddings.
    This implementation uses a simple LSTM and thus models each speaker
    independently. 
    An attention mechanism can be added to model the interactions
    between speakers and would likely improve the performance.
    """

    def __init__(self, input_dim, output_dim):
        super(TemporalModel, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.lstm = nn.LSTM(
            input_dim, output_dim, batch_first=True)

    def forward(self, x):
        """
        Forward pass through the temporal model.
        x: tensor
            Input tensor of shape (batch_size, n_speakers, input_dim)
        Returns:
            x: tensor
                Output tensor of shape (batch_size, n_speakers, output_dim)
        """
        # print("Forward pass through the temporal model")
        # print(f"Input shape: {x.shape}")
        x = x.view(-1, self.input_dim)
        # print(f"Input shape after reshaping: {x.shape}")
        x, _ = self.lstm(x)
        # print(f"Shape after LSTM: {x.shape}")
        return x


class ContrastiveLoss(nn.Module):
    """
    Contrastive loss function for training the model.
    Based on the TalkNCE loss (see references)
    """

    def __init__(self, tau=1):
        super(ContrastiveLoss, self).__init__()
        self.tau = tau

    def forward(self, s):
        """
        Compute the contrastive loss.

        T_act: duration of the active speaking region for each item in the batch
        s: dot product of the audio and visual embeddings
        """
        exp_s = torch.exp(s / self.tau)
        # Compute the sum of the exponentials for i != j
        exp_diag = exp_s.diagonal(offset=0, dim1=-2, dim2=-1)
        sum_exp = exp_s.sum(dim=1, keepdim=True) - exp_diag
        # Compute the contrastive loss
        loss = -torch.log(exp_diag / sum_exp).mean()
        return loss


class VisualSAD(nn.Module):
    """
    Main model class for VisualSAD.
    Input shape is (batch_size, max_n_speakers, C, T, H, W) for visual data
    and (batch_size, 4*T, C_mel) for audio data.
    """

    def __init__(self,
                 C, T, H, W, C_mel,
                 embedding_dim_audio,
                 embedding_dim_visual,
                 embedding_dim,
                 target_dim,
                 lmbda=0.3):
        super(VisualSAD, self).__init__()

        self.C = C  # Number of channels in the visual data
        self.T = T  # Number of time frames in the visual data
        self.H = H  # Height of the visual data
        self.W = W  # Width of the visual data
        self.C_mel = C_mel  # Number of mel bands in the audio data
        self.embedding_dim_audio = embedding_dim_audio
        self.embedding_dim_visual = embedding_dim_visual
        self.embedding_dim = embedding_dim  # Dimension of the final embedding
        self.target_dim = target_dim  # Number of classes for classification

        # Lambda parameter to balance the influence of the contrastive loss
        self.lmbda = lmbda

        input_shape_audio = (4 * T, C_mel)
        input_shape_visual = (C, T, H, W)

        self.audio_encoder = AudioEncoder(
            input_shape=input_shape_audio,
            output_dim=embedding_dim_audio)
        self.visual_encoder = VisualEncoder(
            input_shape=input_shape_visual,
            output_dim=embedding_dim_visual)

        self.fusion = AudioVisualFusion(
            embedding_dim_audio=embedding_dim_audio,
            embedding_dim_visual=embedding_dim_visual,
            output_dim=embedding_dim)

        self.temporal_model = TemporalModel(
            input_dim=embedding_dim,
            output_dim=target_dim)

        # Contrastive loss function
        self.contrastive_loss = ContrastiveLoss()
        # Final loss function for classification of speakers
        self.classification_loss = nn.CrossEntropyLoss()

    def forward(self, audio, visual, targets):
        """Forward pass through the model."""
        batch_size = audio.size(0)
        n_speakers = visual.size(1)

        audio_embedding = self.audio_encoder(audio)

        # Collapse the first two dimensions of the visual input
        # (batch_size * n_speakers, C, T, H, W)
        visual = visual.view(-1, self.C, self.T, self.H, self.W)
        visual_embedding = self.visual_encoder(visual)

        # print(
        #    f"Audio embedding shape after encoder: {audio_embedding.shape}")
        # print(
        #    f"Visual embedding shape after encoder: {visual_embedding.shape}")

        # Compute the dot product of the audio and visual embeddings
        # Broadcast the audio embedding to match the visual embedding shape
        # Shape is (batch_size, n_speakers, T, embedding_dim_audio)
        audio_embedding = audio_embedding.unsqueeze(1)
        audio_embedding = audio_embedding.expand(
            -1, n_speakers, -1, -1)
        # Collapse the first two dimensions of the audio embedding
        # (batch_size * n_speakers, T, embedding_dim_audio)
        audio_embedding = audio_embedding.contiguous().view(
            -1, self.T, self.embedding_dim_audio)

        # print(
        #    f"Audio embedding shape after expanding: {audio_embedding.shape}")
        # print(
        #    f"Visual embedding shape after reshaping: {visual_embedding.shape}")
        
        # Select only the active frames i.e. the frames where at least one
        # speaker is speaking (targets[..., t, :])
        # and compute the contrastive loss using the
        # corresponding embeddings
        targets = targets.view(batch_size*n_speakers, self.T, -1)
        act_idx = targets.sum(dim=-1) > 0
        if act_idx.sum() > 0:
            act_embedding_audio = audio_embedding[act_idx]
            act_embedding_visual = visual_embedding[act_idx]       
            # Shape is (batch_size * n_speakers, T_act, embedding_dim_audio)
            # Compute the dot product along the 2 last dimensions
            s = torch.matmul(
                act_embedding_audio, act_embedding_visual.transpose(1, 2))
            # Compute the contrastive loss
            loss_c = self.contrastive_loss(s)
        else:
            # If no active frames are present, set the loss to 0
            loss_c = torch.tensor(0.0).to(audio.device)

        # Apply the fusion module to get the final embedding
        # Shape is (batch_size, T, embedding_dim)
        x = self.fusion(audio_embedding, visual_embedding)

        # Apply the temporal model to get the predictions
        x = self.temporal_model(x)
        x = x.view(batch_size*n_speakers, self.T, x.size(-1))
        # print(f"Shape after temporal model: {x.shape}")

        # Compute the classification loss
       
        # print(f"Targets shape: {targets.shape}")
        targets = targets.view(batch_size*n_speakers, self.T, -1)
        loss_cls = self.classification_loss(x, targets)
        total_loss = loss_cls + self.lmbda * loss_c

        # Reshape the output to (batch_size, n_speakers, output_dim)
        return total_loss, x.view(-1, x.size(1), x.size(2)), loss_cls, loss_c
