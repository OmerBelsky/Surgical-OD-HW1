import torch

model = torch.hub.load('yolov5', 'custom', path='Lib/best.pt', source='local')

def model_predict(img):
   res = model(img)
   return res.pandas().xyxy[0]
