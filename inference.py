"""
Модуль предсказания категорий по названиям товаров.
Используется ботом и может вызываться из ноутбука.
"""
import os
import re
import json
from typing import Any, Dict, List, Set, Optional

import numpy as np
import joblib


_ws = re.compile(r"\s+")
_ctrl = re.compile(r"[\x00-\x1f\x7f]")


def normalize_text(s: Any) -> str:
    s = "" if s is None else str(s)
    s = s.strip()
    s = _ctrl.sub(" ", s)
    s = s.replace("\u00a0", " ")
    s = _ws.sub(" ", s)
    return s


def title_key(s: Any) -> str:
    return normalize_text(s).lower()


def _get_clf(model: Any) -> Any:
    return (
        model.named_steps["clf"]
        if hasattr(model, "named_steps") and "clf" in model.named_steps
        else model
    )


def decision_scores_and_classes(model: Any, X) -> tuple:
    clf = _get_clf(model)

    if hasattr(model, "predict_proba"):
        proba = np.asarray(model.predict_proba(X))
        classes = np.asarray(getattr(clf, "classes_", np.arange(proba.shape[1])))
        return proba, classes

    raw = np.asarray(model.decision_function(X))
    classes = getattr(clf, "classes_", None)
    classes = np.asarray(classes) if classes is not None else None

    if raw.ndim == 1:
        raw = raw.reshape(-1)
        scores = np.stack([-raw, raw], axis=1)
        if classes is None or classes.shape[0] != 2:
            classes = np.array([0, 1], dtype=int)
        return scores, classes

    scores = raw
    if classes is None or classes.shape[0] != scores.shape[1]:
        classes = np.arange(scores.shape[1], dtype=int)
    return scores, classes


def load_artifacts(
    out_dir: str = "models",
) -> tuple[Any, Dict[int, str], dict, Optional[Set[str]]]:
    """Загружает модель, id2label, metrics и опционально seen_titles."""
    base = os.path.dirname(os.path.abspath(__file__))
    models_path = os.path.join(base, out_dir) if not os.path.isabs(out_dir) else out_dir

    model = joblib.load(os.path.join(models_path, "model.joblib"))
    with open(
        os.path.join(models_path, "labels.json"), "r", encoding="utf-8"
    ) as f:
        labels = json.load(f)

    id2label = {int(k): v for k, v in labels["id2label"].items()}

    metrics = {}
    mpath = os.path.join(models_path, "metrics.json")
    if os.path.exists(mpath):
        with open(mpath, "r", encoding="utf-8") as f:
            metrics = json.load(f)

    seen_titles = None
    if "seen_titles" in labels:
        seen_titles = set(labels["seen_titles"])

    return model, id2label, metrics, seen_titles


def predict_smart(
    titles: List[str],
    model: Any,
    id2label: Dict[int, str],
    *,
    margin_threshold: float = 0.25,
    proba_threshold: float = 0.55,
    seen_titles: Optional[Set[str]] = None,
) -> List[dict]:
    """Предсказание с проверкой уверенности (proba или margin)."""
    results: List[dict] = []

    for t in titles:
        t_norm = normalize_text(t)
        is_seen = seen_titles is not None and title_key(t_norm) in seen_titles

        clf = _get_clf(model)

        if hasattr(model, "predict_proba"):
            proba = np.asarray(model.predict_proba([t_norm]))[0]
            classes = np.asarray(
                getattr(clf, "classes_", np.arange(proba.shape[0]))
            )

            best_pos = int(np.argmax(proba))
            best_class = int(classes[best_pos])
            conf = float(proba[best_pos])

            if conf < proba_threshold:
                msg = (
                    f"Low confidence (proba={conf:.3f} < {proba_threshold:.2f}). Not assigned."
                )
                if seen_titles is not None and not is_seen:
                    msg += " (title not in training set)"
                results.append(
                    {
                        "title": t,
                        "status": "OUT_OF_CATEGORY_LIST",
                        "category": None,
                        "message": msg,
                    }
                )
                continue

            msg = f"proba={conf:.3f}"
            if seen_titles is not None and not is_seen:
                msg += " (new title, classified by features)"
            results.append(
                {
                    "title": t,
                    "status": "OK",
                    "category": id2label[best_class],
                    "message": msg,
                }
            )
            continue

        scores2d, classes = decision_scores_and_classes(model, [t_norm])
        scores = scores2d[0]

        best_pos = int(np.argmax(scores))
        best_class = int(classes[best_pos])
        best_score = float(scores[best_pos])
        second_score = (
            float(np.partition(scores, -2)[-2]) if scores.size > 1 else float("-inf")
        )
        margin = best_score - second_score

        if margin < margin_threshold:
            msg = (
                f"Low confidence (margin={margin:.3f} < {margin_threshold:.2f}). Not assigned."
            )
            if seen_titles is not None and not is_seen:
                msg += " (title not in training set)"
            results.append(
                {
                    "title": t,
                    "status": "OUT_OF_CATEGORY_LIST",
                    "category": None,
                    "message": msg,
                }
            )
            continue

        msg = f"margin={margin:.3f}"
        if seen_titles is not None and not is_seen:
            msg += " (new title, classified by features)"
        results.append(
            {
                "title": t,
                "status": "OK",
                "category": id2label[best_class],
                "message": msg,
            }
        )

    return results
