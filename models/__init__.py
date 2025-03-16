from models.DPDNet2D import DPDNet_2D
from models.DPDNet3D import DPDNet_3D
from models.submodule import model_loss

__models__ = {
    "DPDNet2D": DPDNet_2D,
    "DPDNet3D": DPDNet_3D
}
