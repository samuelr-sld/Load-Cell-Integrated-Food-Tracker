from ultralytics import YOLO

model = YOLO(r'runs/segment/food_seg_11s_v2/weights/best.pt')

# Test using local file path
results = model.predict(source='C:/Users/Samuel/Downloads/meal 5.jpg', save=True, conf=0.25)