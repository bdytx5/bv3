# # from potassium import Potassium, Request, Response
# # from transformers import pipeline
# # import torch

# # app = Potassium("my_app")

# # # @app.init runs at startup, and loads models into the app's context
# # @app.init
# # def init():
# #     device = 0 if torch.cuda.is_available() else -1
# #     model = pipeline('fill-mask', model='bert-base-uncased', device=device)
   
# #     context = {
# #         "model": model
# #     }

# #     return context

# # # @app.handler runs for every call
# # @app.handler("/")
# # def handler(context: dict, request: Request) -> Response:
# #     prompt = request.json.get("prompt")
# #     model = context.get("model")
# #     outputs = model(prompt)

# #     return Response(
# #         json = {"outputs": outputs[0]}, 
# #         status=200
# #     )

# # if __name__ == "__main__":
# #     app.serve()


# from PIL import Image
# import time
# import onnxruntime
# import numpy as np

# from potassium import Potassium, Request, Response
# from transformers import pipeline
# import torch

# app = Potassium("my_app")

# # Load ONNX model and example input
# onnx_filename = "timesformer_modelOrig.onnx"
# ort_session = onnxruntime.InferenceSession(onnx_filename)
# example_input = np.random.randn(1, 3, 32, 224, 224).astype(np.float32)

# # @app.init runs at startup, and loads models into the app's context
# @app.init
# def init():
#     device = 0 if torch.cuda.is_available() else -1
#     model = pipeline('fill-mask', model='bert-base-uncased', device=device)

#     context = {
#         "model": model,
#         "onnx_session": ort_session
#     }

#     return context

# # @app.handler runs for every call
# @app.handler("/")
# def handler(context: dict, request: Request) -> Response:
#     prompt = request.json.get("prompt")
#     model = context.get("model")
#     outputs = model(prompt)

#     # ONNX model inference
#     start_time = time.time()
#     ort_inputs = {'input': example_input}
#     ort_outputs = context.get("onnx_session").run(None, ort_inputs)
#     end_time = time.time()
#     onnx_inference_time = end_time - start_time
#     print(f"ONNX Inference Time: {onnx_inference_time} seconds")
#     print(ort_outputs)

#     return Response(
#         json={"outputs": outputs[0]},
#         status=200
#     )

# if __name__ == "__main__":
#     app.serve()


# from PIL import Image
# import time
# import onnxruntime
# import numpy as np

# from potassium import Potassium, Request, Response

# app = Potassium("my_app")

# # Load ONNX model and example input
# onnx_filename = "timesformer_modelOrig.onnx"
# ort_session = onnxruntime.InferenceSession(onnx_filename)

# example_input = np.random.randn(1, 3, 32, 224, 224).astype(np.float32)
# print(onnx_filename.get_device())
# # @app.init runs at startup, and loads models into the app's context
# @app.init
# def init():
#     context = {
#         "onnx_session": ort_session
#     }

#     return context

# # @app.handler runs for every call
# @app.handler("/")
# def handler(context: dict, request: Request) -> Response:
#     # ONNX model inference
#     start_time = time.time()
#     ort_inputs = {'input': example_input}
#     ort_outputs = context.get("onnx_session").run(None, ort_inputs)
#     end_time = time.time()
#     onnx_inference_time = end_time - start_time
#     print(f"ONNX Inference Time: {onnx_inference_time} seconds")

#     return Response(
#         json={"outputs": ort_outputs[0].tolist()}, # Convert numpy array to list for serialization
#         status=200
#     )

# if __name__ == "__main__":
#     app.serve()




import torch
from PIL import Image
import time
import numpy as np
import torch
import sys
sys.path.append('./TimeSformer')

from timesformer.models.vit import TimeSformer

from potassium import Potassium, Request, Response

app = Potassium("my_app")

# Check CUDA availability and set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load PyTorch model
model = TimeSformer(img_size=224, num_classes=3, num_frames=32, attention_type='divided_space_time', pretrained_model='./32frame.pyth')
model = model.to(device)
model.eval()  

example_input = torch.randn(1, 3, 32, 224, 224).to(device)

# @app.init runs at startup, and loads models into the app's context
@app.init
def init():
    context = {
        "torch_model": model
    }

    return context

# @app.handler runs for every call
@app.handler("/")
def handler(context: dict, request: Request) -> Response:
    # PyTorch model inference
    start_time = time.time()
    with torch.no_grad():
        outputs = context.get("torch_model")(example_input)
    end_time = time.time()
    torch_inference_time = end_time - start_time
    print(f"PyTorch Inference Time: {torch_inference_time} seconds")

    return Response(
        json={"outputs": outputs[0].cpu().numpy().tolist()}, # Convert torch tensor to numpy array then to list for serialization
        status=200
    )

if __name__ == "__main__":
    app.serve()
