from ultralytics import YOLO

model = YOLO("yolov8n.pt")

if __name__ == '__main__':
    ruta_yaml = "reto_manchester.v3i.yolov8/data.yaml"

    resultados = model.train(
        data=ruta_yaml,                     
        epochs=100,                         
        patience=20,                        
        lr0=0.001,                          
        val=True,                           
        imgsz=640,                          
        batch=4,                          
        
  
        device=0,                           
        
        hsv_v=0.4,                          
        degrees=15.0,                       
        translate=0.1,                      
        scale=0.5,                          
        fliplr=0.0,                         
        mosaic=1.0                          
    )
