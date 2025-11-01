from ultralytics import YOLO
data_path = r"D:\Vishaal R\VS Code\Datasets\data.yaml"
model = YOLO("yolov8n.pt")
model.train(
    data=data_path,     
    epochs=50,          
    imgsz=640,          
    batch=8,            
    name="accident_yolov8",  
    project="runs",    
)

print("Training completed")
