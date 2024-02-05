import torch
import torch.nn as nn

from model.residual_vq import ResidualVQ


class SemanticEncoder(nn.Module):
    """
    a DNN encoder that encodes the input semantic embedding into a latent representation.
    The encoder has three intermediate layers of size 512, 256 and 128 with ReLU activation
    with a final latent representation dimension of 32.
    """

    def __init__(self, input_dim=768, latent_dim=32):
        super(SemanticEncoder, self).__init__()
        self.encoder_layers = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim),
        )

    def forward(self, semantic_embedding):
        encoded_latent = self.encoder_layers(semantic_embedding)
        return encoded_latent


class ResidualQuantizer(nn.Module):
    def __init__(self):
        super(ResidualQuantizer, self).__init__()

    def forward(self, x):
        residual_vq = ResidualVQ(
            dim=32,
            num_quantizers=3,
            codebook_size=256,
            stochastic_sample_codes=True,
            sample_codebook_temp=0.1,
            shared_codebook=False
        )
        quantized, indices, commit_loss = residual_vq(x)
        return quantized, indices, commit_loss


class SemanticDecoder(nn.Module):
    """
    a DNN decoder that decodes the quantized representation back to the semantic input embedding
    """

    def __init__(self, latent_dim=32, output_dim=768):
        super(SemanticDecoder, self).__init__()
        self.decoder_layers = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim),
        )

    def forward(self, quantized_representation):
        decoded_semantic_embedding = self.decoder_layers(quantized_representation)
        return decoded_semantic_embedding


if __name__ == "__main__":
    # 定义模型参数
    input_dim = 768  # 假设输入的语义嵌入维度为768
    latent_dim = 32  # 最终潜在表示维度为32

    # 创建编码器实例
    encoder = SemanticEncoder(input_dim)

    # 示例使用
    semantic_input = torch.randn(10, input_dim)  # 假设我们有10个样本，每个样本的语义嵌入维度为768
    latent_representations = encoder(semantic_input)

    print(f"Encoded latent representations shape: {latent_representations.shape}")

    # 初始化编码后的潜在表示（假设维度与codebook_dim一致）
    latent_representations = torch.randn(10, 32)  # 假设有10个样本，每个样本的潜在表示维度为32

    # 创建并实例化残差量化器
    quantizer = ResidualQuantizer()

    # 对潜在表示进行量化
    quantized, indices, commit_loss = quantizer(latent_representations)
    print(f"Quantized representations shape: {quantized.shape}")
    print(f"Indices shape: {indices.shape}")
    print(f"Commit loss: {commit_loss}")
