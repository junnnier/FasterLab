import torch
from torch import nn
from torchvision import models
from transformers import DistilBertModel


class ImageEncoder(nn.Module):
    def __init__(self, trainable=True):
        super().__init__()
        self.model = nn.Sequential()
        # 图像编码网络使用resnet50
        self.basemodel = models.resnet50()
        state_dict = torch.load("resnet50-11ad3fa6.pth")
        self.basemodel.load_state_dict(state_dict)
        # 去除最后一层
        self.basemodel = nn.Sequential(*list(self.basemodel.children())[:-1])
        # 设置是否可训练参数
        for p in self.basemodel.parameters():
            p.requires_grad = trainable
        self.idi = torch.nn.Identity()

    def forward(self, x):
        x = self.basemodel(x)
        x = x.squeeze(-1).squeeze(-1)
        x = self.idi(x)
        return x


class TextEncoder(nn.Module):
    def __init__(self, model_name, trainable=False):
        super().__init__()
        # 加载Bert模型作为文本的Encoder
        self.model = DistilBertModel.from_pretrained(model_name)
        for p in self.model.parameters():
            p.requires_grad = trainable
        self.target_token_idx = 0

    def forward(self, input_ids, attention_mask):
        output = self.model(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = output.last_hidden_state  # (10, 100, 768)
        # return last_hidden_state[:, self.target_token_idx, :]
        return last_hidden_state[:, 1:4, :].mean(dim=1)


class ProjectionHead(nn.Module):
    def __init__(self, embedding_dim, projection_dim, dropout):
        super().__init__()
        self.projection = nn.Linear(embedding_dim, projection_dim)
        self.gelu = nn.GELU()
        self.fc = nn.Linear(projection_dim, projection_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(projection_dim)

    def forward(self, x):
        projected = self.projection(x)
        x = self.gelu(projected)
        x = self.fc(x)
        x = self.dropout(x)
        x = x + projected
        x = self.layer_norm(x)
        return x


def cross_entropy(preds, targets, reduction='none'):
    log_softmax = nn.LogSoftmax(dim=-1)
    loss = (-targets * log_softmax(preds)).sum(1)
    if reduction == "none":
        return loss
    elif reduction == "mean":
        return loss.mean()


class CLIPModel(nn.Module):
    def __init__(self, text_model_name, trainable, image_embedding, text_embedding, projection_dim, dropout):
        super().__init__()
        self.image_encoder = ImageEncoder(trainable=trainable)
        self.text_encoder = TextEncoder(text_model_name, trainable=trainable)
        self.image_projection = ProjectionHead(image_embedding, projection_dim, dropout)
        self.text_projection = ProjectionHead(text_embedding, projection_dim, dropout)

    def forward(self, batch):
        # 图片编码
        image_features = self.image_encoder(batch["image"])
        # 文本编码
        text_features = self.text_encoder(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
        # 图片特征和文本特征映射到相同维度上
        image_embeddings = self.image_projection(image_features)
        text_embeddings = self.text_projection(text_features)

        return [image_embeddings, text_embeddings]


def create_model(param):
    return CLIPModel(param["text_model_name"],
                     param["trainable"],
                     param["image_embedding"],
                     param["text_embedding"],
                     param["projection_dim"],
                     param["dropout"])


if __name__ == '__main__':
    from torchsummary import summary
    param = {"temperature": 1.0,
             "image_embedding": 2048,
             "text_embedding": 768}
    net = create_model(param)
    result = summary(net,input_size=[(2048, 512, 512),(768, 100)])
    print(result)