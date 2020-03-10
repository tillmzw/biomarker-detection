#!/usr/bin/env python3

import subprocess
import logging
import os
import datetime
import numpy as np


logger = logging.getLogger(__name__)


def git_hash(check_dirty=True, directory=None):
    try:
        sha = subprocess.check_output([
            "git", 
            "--git-dir=%s/.git" % (directory or "."), 
            "rev-parse", 
            "--short", 
            "HEAD"]).strip().decode("ascii")
        if not check_dirty:
            return sha
        else:
            # check for uncommitted changes
            retval = subprocess.call([
                "git", 
                "--git-dir=%s/.git" % (directory or "."), 
                "diff-index", 
                "--quiet", 
                "HEAD", 
                "--"])
            return "%s%s" % (sha, "DIRTY" if retval > 0 else "")
    except subprocess.CalledProcessError as e:
        logger.warning(e)
        return None


def hostname():
    try:
        host = subprocess.check_output(["hostname", "--fqdn"]).strip().decode("utf8")
    except subprocess.CalledProcessError as e:
        logger.warning(e)
        return None
    else:
        return host


def norm_mat(mat, norm="all"):
    if norm not in ("rows", "cols", "all"):
        raise SyntaxError("Invalid argument to `norm_mat`: %s" % norm)

    if norm == "rows":
        # normalize such that the sum of every row is 1
        mat = mat / mat.sum(axis=1, keepdims=True)
    elif norm == "cols":
        mat = mat / mat.sum(axis=0, keepdims=True)
    elif norm == "all":
        mat = mat / mat.sum()

    return mat


def convert_multilabel_confusion_mat(mat):
    """
     the confusion matrix is expected to be for every class
    [
        [true_negative, false_negative],
        [true_positive, false_positive]
    ]
    so the final confusion matrix has the shape (5, 2, 2) when calculated for 5 classes:
    [
        [
            [TN_c0, FN_c0],
            [TP_c0, TN_c0]
        ],
        [
            [TN_c1, FN_c1],
            [TP_c1, TN_c1]
        ],
        ...
    ]
    the output will look like:
    [
        [TN_c0, FN_c0, TP_c0, TN_c0],
        [TN_c1, FN_c1, TP_c1, TN_c1],
        ...
    ]
    """
    cmat = mat.reshape((5, 4, -1)).squeeze()
    return cmat


def plot_confusion_matrix(confusion, classes=None):
    """
    Create a matplotlib plot from the 2d matrix `confusion`.
    Can normalize to "rows" (ground truth), "cols" (predictions), "all" (all elements sum to 1).
    FIXME: this operates on matplotlib's global pyplot object
    """

    if len(confusion.shape) == 3:
        return plot_multilabel_confusion_matrix(confusion)

    # this is a reproduction of the plotting functionality in sklearn's confusion_matrix:
    # https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/metrics/_plot/confusion_matrix.py
    import numpy as np
    import matplotlib.pyplot as plt
    import itertools

    if confusion.min() != confusion.max():
        # this guard prevents `nan`s produced by normlization for an all-0 tensor
        confusion_norm = norm_mat(confusion, norm="rows")
    else:
        confusion_norm = confusion
    assert confusion_norm.max() <= 1, "Confusion matrix contains values > 1; color map does not accomodate such values"

    n_classes = confusion.shape[0]
    text = np.empty_like(confusion, dtype=object)
    if classes:
        if n_classes != len(classes):
            logger.warning("Bad class legend")
            display_labels = ("?" for _ in range(n_classes))
        else:
            display_labels = classes
    else:
        display_labels = ("class %d" % c for c in range(n_classes))
    values_format = "{row_norm:.2g}\nn={abs_val:d}"

    fig, ax = plt.subplots()

    im = ax.imshow(confusion_norm, interpolation="nearest", cmap="viridis", vmin=0, vmax=1)
    cmap_min, cmap_max = im.cmap(0), im.cmap(256)
    # choose an appropriate color for the text, based on background color
    thresh = (confusion_norm.max() + confusion_norm.min()) / 2.0
    for i, j in itertools.product(range(n_classes), range(n_classes)):
        color = cmap_max if confusion_norm[i, j] < thresh else cmap_min
        text[i, j] = ax.text(j, i,
                             values_format.format(row_norm=confusion_norm[i, j], abs_val=confusion[i, j]),
                             ha="center", va="center",
                             fontsize=8,
                             color=color)
    fig.colorbar(im, ax=ax)
    # FIXME: labels are hidden on x axis
    ax.set(xticks=np.arange(n_classes),
           yticks=np.arange(n_classes),
           xticklabels=display_labels,
           yticklabels=display_labels,
           ylabel="True label",
           xlabel="Predicted label")

    ax.set_title("Confusion Matrix")
    ax.set_ylim((n_classes - 0.5, -0.5))
    plt.setp(ax.get_xticklabels(), rotation="horizontal")

    return plt


def plot_multilabel_confusion_matrix(confusion):
    assert len(confusion.shape) == 3, "Not a multilabel confusion matrix"

    import numpy as np
    import matplotlib.pyplot as plt
    import itertools
    import dataset

    confusion = convert_multilabel_confusion_mat(confusion)
    confusion_norm = norm_mat(confusion, norm="rows")
    n_classes = confusion.shape[0]

    fig, ax = plt.subplots()
    text = np.empty_like(confusion, dtype=object)
    values_format = "{row_norm:.2g}\nn={abs_val:d}"
    im = ax.imshow(confusion, interpolation="nearest", cmap="viridis")

    cmap_min, cmap_max = im.cmap(0), im.cmap(256)
    # choose an appropriate color for the text, based on background color
    thresh = (confusion_norm.max() + confusion_norm.min()) / 2.0
    for i, j in itertools.product(range(n_classes), range(4)):
        color = cmap_max if confusion_norm[i, j] < thresh else cmap_min
        text[i, j] = ax.text(j, i,
                             values_format.format(row_norm=confusion_norm[i, j], abs_val=confusion[i, j]),
                             ha="center", va="center",
                             fontsize=8,
                             color=color)
    fig.colorbar(im, ax=ax)

    ax.set(
        xticks=np.arange(4),
        yticks=np.arange(n_classes),
        xticklabels=("TN", "FN", "TP", "FP"),
        yticklabels=dataset.IDRIDDataset.CLASSES,
        ylabel="Class"
    )
    return plt


def plot_to_pil(plt, format="png"):
    import io
    from PIL import Image
    buf = io.BytesIO()
    plt.savefig(buf, format=format, bbox_inches="tight")
    plt.close()  # make sure matplotlib can GC the plot
    buf.seek(0)
    pil = Image.open(buf)
    pil.load()  # copy the buffer's contents so we can close the buffer
    buf.close()
    return pil


def compound_img_hist(img, colorbar=None):
    """Add a histogram below PIL Image `img` and return a PIL image"""
    from matplotlib import pyplot as plt
    import numpy as np
    fig = plt.figure()
    fig.add_subplot(2, 1, 1)
    plt.imshow(img)
    plt.axis('off')
    if colorbar:
        from matplotlib import cm
        plt.colorbar(cm.ScalarMappable(cmap=colorbar), orientation='vertical')

    fig.add_subplot(2, 1, 2)
    for channel, color in zip(img.split(), ("r", "g", "b")):
        logger.debug("Calculating histogram for channel %s" % color)
        col_hist = channel.histogram()
        plt.plot(col_hist, color=color, linestyle="-" if color == "r" else ":")
        if color == "r":
            # focus on the red channel
            plt.ylim(np.min(col_hist), np.max(col_hist) * 1.1)

    return plot_to_pil(plt)


def save_pil_to_scratch(pil, target_dir, name, overwrite=False):
    if not os.path.isdir(target_dir) and os.path.isfile(target_dir):
        logger.error("%s exists and is not a directory" % target_dir)
        return
    target_file = os.path.join(target_dir, name)
    if os.path.exists(target_file) and not overwrite:
        raise FileExistsError("%s exists already" % target_file)
    elif os.path.exists(target_file):
        mtime = os.stat(target_file).st_mtime
        mdelta = datetime.datetime.now() - datetime.datetime.fromtimestamp(mtime)
        logger.info("Overwriting %s (last changed %s ago)" % (target_file, mdelta))

    pil.save(target_file)
