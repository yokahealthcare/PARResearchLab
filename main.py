import cv2

from PAR.par import PAR


def main():
    par = PAR(dataset="market", backbone="resnet50", weight_file_path="asset/model/market/resnet50_nfc/net_last.pth")

    img = cv2.imread("asset/img/test_market.jpg")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Must convert to RGB channel
    attribute_result = par.inference(img=img)

    print(attribute_result)


if __name__ == '__main__':
    main()
