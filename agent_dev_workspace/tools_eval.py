from typing import Optional
import numpy as np
from langchain_core.tools import tool
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score


def make_tools(session):
    @tool
    def evaluate_model_tool(
        average: str = "macro",
    ) -> dict:
        """Evaluates stored model predictions against true labels, computing classification metrics and confusion matrix.

        Args:
            average: Averaging strategy for multi-class precision, recall, and F1 ('macro', 'micro', or 'weighted'). Default is 'macro'.

        Returns:
            dict: Summary containing result_id, accuracy, precision, recall, f1_score, and confusion matrix.
        """
        if "predictions" not in session.artifacts:
            return {"error": "No predictions found in session.artifacts. Please run a model prediction tool first."}

        y_pred = session.artifacts["predictions"]

        y_true = None
        if "train_test_split" in session.artifacts:
            y_true = session.artifacts["train_test_split"]["y_test"]
        elif session.dataframe is not None and "category" in session.dataframe.columns:
            y_true = session.dataframe["category"].values
        elif session.dataframe is not None and "category_name" in session.dataframe.columns:
            y_true = session.dataframe["category_name"].values
        elif session.labels is not None:
            y_true = session.labels

        if y_true is None:
            return {"error": "No true ground-truth labels found in session."}

        acc = float(accuracy_score(y_true, y_pred))
        prec = float(precision_score(y_true, y_pred, average=average, zero_division=0))
        rec = float(recall_score(y_true, y_pred, average=average, zero_division=0))
        f1 = float(f1_score(y_true, y_pred, average=average, zero_division=0))
        cm = confusion_matrix(y_true, y_pred).tolist()

        result_id = session.next_result_id("evaluate_model")
        summary = {
            "result_id": result_id,
            "accuracy": round(acc, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1_score": round(f1, 4),
            "average_metric": average,
            "confusion_matrix": cm,
            "n_samples": len(y_true),
        }

        session.artifacts["evaluation_results"] = summary
        session.store_result("evaluate_model_tool", {"average": average}, summary)

        return summary

    return [evaluate_model_tool]
