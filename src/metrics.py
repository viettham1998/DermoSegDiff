import torchmetrics
import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def get_metrics(config, *args, **kwargs):
    metrics = torchmetrics.MetricCollection(
        [
            torchmetrics.F1Score(
                task="multiclass", num_classes=config["dataset"]["number_classes"]
            ),
            torchmetrics.Accuracy(
                task="multiclass", num_classes=config["dataset"]["number_classes"]
            ),
            torchmetrics.Dice(),
            torchmetrics.Precision(
                task="multiclass", num_classes=config["dataset"]["number_classes"]
            ),
            torchmetrics.Specificity(
                task="multiclass", num_classes=config["dataset"]["number_classes"]
            ),
            torchmetrics.Recall(
                task="multiclass", num_classes=config["dataset"]["number_classes"]
            ),
            # IoU
            torchmetrics.JaccardIndex(
                task="multiclass", num_classes=config["dataset"]["number_classes"]
            ),
        ],
        prefix="metrics/",
    )

    # test_metrics
    test_metrics = metrics.clone(prefix="").to(device)


# {'metrics/BinaryF1Score': tensor(0.5072, device='cuda:0'), 'metrics/BinaryAccuracy': tensor(0.4850, device='cuda:0'),
#  'metrics/Dice': tensor(0.5072, device='cuda:0'), 'metrics/BinaryPrecision': tensor(0.4953, device='cuda:0'),
#  'metrics/BinarySpecificity': tensor(0.4490, device='cuda:0'), 'metrics/BinaryRecall': tensor(0.5196, device='cuda:0'),
#  'metrics/BinaryAUROC': tensor(0.4843, device='cuda:0'),
#  'metrics/BinaryAveragePrecision': tensor(0.5024, device='cuda:0'),
#  'metrics/BinaryJaccardIndex': tensor(0.3397, device='cuda:0')}


def get_binary_metrics(*args, **kwargs):
    metrics = torchmetrics.MetricCollection(
        [
            torchmetrics.F1Score(task="binary"),
            torchmetrics.Accuracy(task="binary"),
            torchmetrics.Dice(multiclass=False),
            torchmetrics.Precision(task="binary"),
            torchmetrics.Specificity(task="binary"),
            torchmetrics.Recall(task="binary"),
            torchmetrics.AUROC(task="binary"),
            torchmetrics.AveragePrecision(task="binary"),
            # IoU
            torchmetrics.JaccardIndex(task="binary", num_labels=2, num_classes=2),
        ],
        prefix="metrics/",
    )

    # test_metrics
    test_metrics = metrics.clone(prefix="").to(device)
    return test_metrics
