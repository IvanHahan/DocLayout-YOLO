from huggingface_hub import PyTorchModelHubMixin

from doclayout_yolo.engine.model import Model
from doclayout_yolo.nn.tasks import YOLOv10DetectionModel, YOLOv10WithELDetectionModel

from .predict import YOLOv10DetectionPredictor
from .train import YOLOv10DetectionTrainer, YOLOv10WithELDetectionTrainer
from .val import YOLOv10DetectionValidator


class YOLOv10(
    Model,
    PyTorchModelHubMixin,
    repo_url="https://github.com/opendatalab/DocLayout-YOLO",
    pipeline_tag="object-detection",
    license="agpl-3.0",
):

    def __init__(self, model="yolov10n.pt", task=None, verbose=False):
        super().__init__(model=model, task=task, verbose=verbose)

    @property
    def task_map(self):
        """Map head to model, trainer, validator, and predictor classes."""
        return {
            "detect": {
                "model": YOLOv10DetectionModel,
                "trainer": YOLOv10DetectionTrainer,
                "validator": YOLOv10DetectionValidator,
                "predictor": YOLOv10DetectionPredictor,
            },
        }


class YOLOv10WithEL(
    Model,
    PyTorchModelHubMixin,
    repo_url="https://github.com/opendatalab/DocLayout-YOLO",
    pipeline_tag="object-detection",
    license="agpl-3.0",
):

    def __init__(self, model="yolov10n.pt", task=None, verbose=False):
        super().__init__(model=model, task=task, verbose=verbose)

    @property
    def task_map(self):
        """Map head to model, trainer, validator, and predictor classes."""
        return {
            "detect": {
                "model": YOLOv10WithELDetectionModel,
                "trainer": YOLOv10WithELDetectionTrainer,
                "validator": YOLOv10DetectionValidator,
                "predictor": YOLOv10DetectionPredictor,
            },
        }
