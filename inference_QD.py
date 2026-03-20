import cv2
import numpy as np
import torch
import torch.nn as nn
import argparse
from model_CNN_QuickDraw import CNN

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-size", "-i", type=int, default=28)
    parser.add_argument("--image-path", "-p", type=str, required=True)
    parser.add_argument("--checkpoint_path", "-t", type=str, default="./train_model_QuickDraw/quickdraw")
    return parser.parse_args()

def preprocess_for_quickdraw(img_path, size):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("Không tìm thấy file ảnh!")

    if np.mean(img) > 127:
        img = cv2.bitwise_not(img)

    _, img = cv2.threshold(img, 50, 255, cv2.THRESH_BINARY)

    coords = cv2.findNonZero(img)
    if coords is not None:
        x, y, w, h = cv2.boundingRect(coords)
        img = img[y:y+h, x:x+w]
        pad = int(max(w, h) * 0.1)
        img = cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)

    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    
    return img

def inference(args):
    categories = ["airplane", "angel", "apple", "axe", "bat", "book", "boomerang", "camera", "cup", "fish", "flower", "mushroom", "radio", "sun", "sword"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    processed_img = preprocess_for_quickdraw(args.image_path, args.image_size)
    
    input_data = processed_img.astype(np.float32) / 255.0
    input_tensor = torch.from_numpy(input_data).reshape(1, 1, 28, 28).to(device)

    try:
        model = torch.load(args.checkpoint_path, map_location=device, weights_only=False)
        model.eval()
    except Exception as e:
        print(f"Lỗi load model: {e}")
        return

    with torch.no_grad():
        output = model(input_tensor)
        prob = nn.Softmax(dim=1)(output)
        
    conf, idx = torch.max(prob, dim=1)
    label = categories[idx.item()]

    print(f"\nDỰ ĐOÁN: {label.upper()}")
    print(f"ĐỘ TỰ TIN: {conf.item()*100:.2f}%")

    debug_view = cv2.resize(processed_img, (280, 280), interpolation=cv2.INTER_NEAREST)
    cv2.imshow(f"Model View: {label}", debug_view)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    args = get_args()
    inference(args)

# python inference_QD.py -p "test\sun.png" -t "./train_model_QuickDraw/quickdraw"
# python -m tensorboard.main --logdir tensor_board_quickdraw/quickdraw