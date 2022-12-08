import torch
import json

device = "cuda" if torch.cuda.is_available() else "cpu"


def model_fn(model_dir):
    model = torch.jit.load(f"{model_dir}/model.scripted.pt")

    model.to(device).eval()
    
    return model

# data preprocessing
def input_fn(request_body, request_content_type):
    assert request_content_type == "application/json"
    data = json.loads(request_body)["inputs"]
    data = torch.tensor(data, dtype=torch.float32, device=device)
    return data


# inference
def predict_fn(input_object, model):
    with torch.no_grad():
        prediction = model(input_object)
    return prediction


# postprocess
def output_fn(predictions, content_type):
    assert content_type == "application/json"
    res = predictions.cpu().numpy().tolist()
    return json.dumps(res)
