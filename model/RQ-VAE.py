import torch
import torch.nn as nn


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
    def __init__(self, codebook_cardinality=256, codebook_dim=32):
        super(ResidualQuantizer, self).__init__()
        # 在实际应用中，这些码本可能通过聚类或其他方式预先计算和存储
        self.codebooks = [torch.randn(codebook_cardinality, codebook_dim) for _ in range(3)]

    def forward(self, latent_representation):
        residuals = [latent_representation]
        for i, codebook in enumerate(self.codebooks):
            distances = torch.cdist(latent_representation.unsqueeze(1), codebook.unsqueeze(0))
            _, nearest_indices = distances.min(dim=1)
            quantized_residual = codebook[nearest_indices]
            if i < len(self.codebooks) - 1:
                residual = latent_representation - quantized_residual
                residuals.append(residual)
                latent_representation = residual
            else:
                quantized_representation = sum(residuals) + quantized_residual
        return quantized_representation


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
    quantized_latent_representations = quantizer(latent_representations)

    print(f"Quantized latent representations shape: {quantized_latent_representations.shape}")
