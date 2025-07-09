import glob
import os
import random


custom_classes = ["custom"]

# By default the custom dataset is expected under the repository's
# `datasets/custom` directory as described in docs/custom_dataset.md.
CUSTOM_DIR = os.path.join(os.path.dirname(__file__), "custom")


def _load_from_dir(folder, label):
    """Load images from a directory.

    Parameters
    ----------
    folder: str
        Directory containing images.
    label: int
        0 for normal images, 1 for abnormal images.

    Returns
    -------
    (list, list, list, list)
        Image paths, dummy mask paths, labels and type strings.
    """
    if not os.path.isdir(folder):
        return [], [], [], []

    img_paths = sorted(glob.glob(os.path.join(folder, "*")))
    gt_paths = [0 for _ in img_paths]
    labels = [label for _ in img_paths]
    types = ["good" if label == 0 else "anomaly" for _ in img_paths]
    return img_paths, gt_paths, labels, types


def load_custom(category, k_shot, experiment_indx):
    """Load a simple folder based dataset.

    The directory ``CUSTOM_DIR`` should contain::

        train/good/      # normal images
        test/anomaly/    # abnormal images for evaluation

    Ground truth masks are optional. If not found, zero masks are used.
    """

    train_good = os.path.join(CUSTOM_DIR, "train", "good")
    test_anomaly = os.path.join(CUSTOM_DIR, "test", "anomaly")

    train_img_paths, train_gt_paths, train_labels, train_types = _load_from_dir(
        train_good, 0)
    test_img_paths, test_gt_paths, test_labels, test_types = _load_from_dir(
        test_anomaly, 1)

    if k_shot > 0 and len(train_img_paths) > k_shot:
        random.seed(experiment_indx)
        indices = random.sample(range(len(train_img_paths)), k_shot)
        train_img_paths = [train_img_paths[i] for i in indices]
        train_gt_paths = [train_gt_paths[i] for i in indices]
        train_labels = [train_labels[i] for i in indices]
        train_types = [train_types[i] for i in indices]

    return (train_img_paths, train_gt_paths, train_labels, train_types), \
           (test_img_paths, test_gt_paths, test_labels, test_types)

