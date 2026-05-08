from ultralytics import YOLO
import multiprocessing

def main():

    # Load a pretrained YOLO11n model
    # model = YOLO("yolo11n-seg.pt")
    #
    # # Train the model on the COCO8 dataset for 100 epochs
    # train_results = model.train(
    #     data="mn-seg.yaml",  # Path to dataset configuration file
    #     epochs=100,  # Number of training epochs
    #     imgsz=640,  # Image size for training
    #     device="0",  # Device to run on (e.g., 'cpu', 0, [0,1,2,3])
    #     # plots=True,  # 保存训练曲线图
    # )

    model = YOLO("yolo11n-seg.pt")

    # Train the model on the COCO8 dataset for 100 epochs
    train_results = model.train(
        data="coco8-seg.yaml",  # Path to dataset configuration file
        epochs=50,  # Number of training epochs
        imgsz=512,  # Image size for training
        device="0",  # Device to run on (e.g., 'cpu', 0, [0,1,2,3])
        workers=0,  # 减少 DataLoader workers 数量，降低内存占用
        batch=8,  # 减小 batch size，降低内存占用
        project="D:/PYTHON_PROJECT/rocket_engine_defect_detect/runs/segment",
        # plots=True,  # 保存训练曲线图
    )

    # Evaluate the model's performance on the validation set
    # metrics = model.val()

    # Perform object detection on an image
    # results = model(r"C:\Users\hy\Desktop\3.png")  # Predict on an image
    # results[0].show()  # Display results

    # Export the model to ONNX format for deployment
    path = model.export(format="onnx")  # Returns the path to the exported model

if __name__ == "__main__":
    # 设置多进程启动方法，避免 Windows 上的内存问题
    multiprocessing.freeze_support()
    main()
