import torch
import pytest

from pathlib import Path
from backend.models.brain_tumor_bot import predict, load_model, load_img


BASE_DIR = Path(__file__).resolve().parents[2]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
best_model = BASE_DIR / "checkpoint" / "model_4_retrain.pt"

@pytest.fixture
def model():
    model = load_model()
    model.load_state_dict(torch.load(best_model, map_location=device))
    model.to(device)
    model.eval()
    return model



def test_glioma(model):
    image_path = BASE_DIR / "data" / "dataset_1" / "Testing" / "glioma" / "Te-gl_1.jpg"
    image_path2 = BASE_DIR / "data" / "dataset_1" / "Testing" / "glioma" / "Te-gl_50.jpg"
    with torch.no_grad():
        image = load_img(image_path).to(device)
        image2 = load_img(image_path2).to(device)
        result = predict(model, image)
        result2 = predict(model, image2)
    
    assert result == "Glioma"
    assert result2 == "Glioma"


def test_meningioma(model):
    image_path = BASE_DIR / "data" / "dataset_1" / "Testing" / "meningioma" / "Te-aug-me_1.jpg"
    image_path2 = BASE_DIR / "data" / "dataset_1" / "Testing" / "meningioma" / "Te-aug-me_50.jpg"
    with torch.no_grad():
        image = load_img(image_path).to(device)
        image2 = load_img(image_path2).to(device)
        result = predict(model, image)
        result2 = predict(model, image2)
    
    assert result == "Meningioma"
    assert result2 == "Meningioma"


def test_notumor(model):
    image_path = BASE_DIR / "data" / "dataset_1" / "Testing" / "notumor" / "Te-no_1.jpg"
    image_path2 = BASE_DIR / "data" / "dataset_1" / "Testing" / "notumor" / "Te-no_50.jpg"
    with torch.no_grad():
        image = load_img(image_path).to(device)
        image2 = load_img(image_path2).to(device)
        result = predict(model, image)
        result2 = predict(model, image2)
    
    assert result == "No Tumor"
    assert result2 == "No Tumor"


def test_pituitary(model):
    image_path = BASE_DIR / "data" / "dataset_1" / "Testing" / "pituitary" / "Te-pi_1.jpg"
    image_path2 = BASE_DIR / "data" / "dataset_1" / "Testing" / "pituitary" / "Te-pi_50.jpg"
    with torch.no_grad():
        image = load_img(image_path).to(device)
        image2 = load_img(image_path2).to(device)
        result = predict(model, image)
        result2 = predict(model, image2)
    
    assert result == "Pituitary"
    assert result2 == "Pituitary"

    

