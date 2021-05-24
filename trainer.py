import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from gcn import GCNClassifier


class GCNTrainer(object):
    def __init__(self, args, emb_matrix=None):
        self.args = args
        self.emb_matrix = emb_matrix
        self.model = GCNClassifier(args, emb_matrix=emb_matrix).cuda()
        self.parameters = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.Adam(self.parameters, lr=args.lr,weight_decay=args.l2reg)

    # load model
    def load(self, filename):
        try:
            checkpoint = torch.load(filename)
        except BaseException:
            print("Cannot load model from {}".format(filename))
            exit()
        self.model.load_state_dict(checkpoint['model'])
        self.args = checkpoint['config']

    # save model
    def save(self, filename):
        params = {
                'model': self.model.state_dict(),
                'config': self.args,
                }
        try:
            torch.save(params, filename)
            print("model saved to {}".format(filename))
        except BaseException:
            print("[Warning: Saving failed... continuing anyway.]")

    def different_loss(self, Z, ZC):
        diff_loss = torch.mean(torch.matmul(Z.permute(0, 2, 1), ZC) ** 2)
        return diff_loss

    def similarity_loss(self, ZCSY, ZCSE):
        ZCSY = F.normalize(ZCSY, p=2, dim=1)
        ZCSE = F.normalize(ZCSE, p=2, dim=1)
        similar_loss = torch.mean((ZCSY - ZCSE) ** 2)
        return similar_loss

    def update(self, batch):
        inputs = batch[0:9]
        label = batch[-1]

        # step forward
        self.model.train()
        self.optimizer.zero_grad()
        logits, gcn_outputs, h_sy, h_se, h_csy, h_cse= self.model(inputs)

        diff_loss = self.args.beta * (self.different_loss(h_sy, h_csy) + self.different_loss(h_se, h_cse))
        similar_loss = self.args.theta * self.similarity_loss(h_csy, h_cse)

        loss = F.cross_entropy(logits, label, reduction='mean') + diff_loss + similar_loss
        corrects = (torch.max(logits, 1)[1].view(label.size()).data == label.data).sum()
        acc = 100.0 * np.float(corrects) / label.size()[0]
        
        # backward
        loss.backward()
        self.optimizer.step()
        return loss.data, acc

    def predict(self, batch):
        inputs = batch[0:9]
        label = batch[-1]

        # forward
        self.model.eval()
        logits, gcn_outputs, h_sy, h_se, h_csy, h_cse = self.model(inputs)

        diff_loss = self.args.beta * (self.different_loss(h_sy, h_csy) + self.different_loss(h_se, h_cse))
        similar_loss = self.args.theta * self.similarity_loss(h_csy, h_cse)

        loss = F.cross_entropy(logits, label, reduction='mean') + diff_loss + similar_loss
        corrects = (torch.max(logits, 1)[1].view(label.size()).data == label.data).sum()
        acc = 100.0 * np.float(corrects) / label.size()[0]
        predictions = np.argmax(logits.data.cpu().numpy(), axis=1).tolist()
        predprob = F.softmax(logits, dim=1).data.cpu().numpy().tolist()

        return loss.data, acc, predictions, label.data.cpu().numpy().tolist(), predprob, gcn_outputs.data.cpu().numpy()
