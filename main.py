import json

import cv2
import torch
from PIL import Image
from torchvision import transforms as T

from net import get_model


def preprocess_image(img):
    img = Image.fromarray(img)  # Convert numpy.array to PIL image object

    transforms = T.Compose([
        T.Resize(size=(288, 144)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    img = transforms(img).unsqueeze(dim=0)  # Add new dimension (1 x channel x height x width)
    return img


class PAR:
    dataset_dict = {
        'market': 'Market-1501',
        'duke': 'DukeMTMC-reID',
    }
    num_cls_dict = {'market': 30, 'duke': 23}
    num_ids_dict = {'market': 751, 'duke': 702}

    def __init__(self, dataset, backbone, weight_file_path, use_id=False):
        self.dataset = dataset
        self.backbone = backbone
        self.weight_file_path = weight_file_path
        self.use_id = use_id

        self.model_name = '{}_nfc_id'.format(self.backbone) if self.use_id else '{}_nfc'.format(self.backbone)
        self.num_label = self.num_cls_dict[self.dataset]
        self.num_id = self.num_ids_dict[self.dataset]

        # Create model
        self.model = get_model(self.model_name, self.num_label, use_id=self.use_id, num_id=self.num_id)
        self.model.load_state_dict(torch.load(self.weight_file_path))
        self.model.eval()

    def predict_decode(self, pred):
        with open('doc/label.json', 'r') as f:
            label_list = json.load(f)[self.dataset]
        with open('doc/attribute.json', 'r') as f:
            attribute_dict = json.load(f)[self.dataset]

        result = {}
        pred = pred.squeeze(dim=0)
        for idx in range(self.num_label):
            name, choice = attribute_dict[label_list[idx]]
            if choice[pred[idx]]:
                result[name] = choice[pred[idx]]

        return result

    def inference(self, img):
        img = preprocess_image(img)

        if not self.use_id:
            out = self.model.forward(img)
        else:
            out, _ = self.model.forward(img)

        pred = torch.gt(out, torch.ones_like(out) / 2)  # threshold=0.5
        return self.predict_decode(pred)


def main():
    par = PAR(dataset="market", backbone="resnet50", weight_file_path="asset/model/market/resnet50_nfc/net_last.pth")

    img = cv2.imread("asset/img/test_market.jpg")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Must convert to RGB channel
    attribute_result = par.inference(img=img)

    print(attribute_result)


if __name__ == '__main__':
    main()
