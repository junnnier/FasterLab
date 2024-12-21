import torch.nn as nn


class LossFunction(nn.Module):
    def __init__(self, model, param):
        super(LossFunction, self).__init__()
        self.temperature = param["temperature"]
        self.log_softmax = nn.LogSoftmax(dim=-1)

    def __call__(self, output):
        image_embeddings, text_embeddings = output
        # 得到每张图片与每个文本的相似程度矩阵
        logits = (text_embeddings @ image_embeddings.T) / self.temperature
        # 得到target
        images_similarity = image_embeddings @ image_embeddings.T
        texts_similarity = text_embeddings @ text_embeddings.T
        targets = nn.functional.softmax((images_similarity + texts_similarity) / 2 * self.temperature, dim=-1)
        # 计算loss
        texts_loss = self.cross_entropy(logits, targets, reduction='none')
        images_loss = self.cross_entropy(logits.T, targets.T, reduction='none')
        loss = (images_loss + texts_loss) / 2.0  # shape: (batch_size)
        loss = loss.mean()
        return loss, [loss.item()]

    def cross_entropy(self, preds, targets, reduction='none'):
        log_softmax = nn.LogSoftmax(dim=-1)
        loss = (-targets * log_softmax(preds)).sum(1)
        if reduction == "none":
            return loss
        elif reduction == "mean":
            return loss.mean()