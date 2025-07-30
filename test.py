

from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO(r'.\sabe-yolo.pt')
    model.predict(source=r'./test_images/',save=True, device=0)

