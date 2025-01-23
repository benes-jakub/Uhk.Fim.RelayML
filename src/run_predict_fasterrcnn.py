from models.fasterrcnn.predict import fasterrcnn_predict

# result = fasterrcnn_predict("../dataset/test/merged/image_637873439631799832.bmp")
result = fasterrcnn_predict("../dataset/test/merged/")
print(str(result.time) + "ms")
