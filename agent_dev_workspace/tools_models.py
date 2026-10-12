from typing import Optional
import numpy as np
from langchain_core.tools import tool
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB


def make_tools(session):
    @tool
    def train_test_split_tool(
        test_size: float = 0.2,
        random_state: int = 42,
        stratify: bool = True,
    ) -> dict:
        """Splits session feature matrix and labels into training and test sets.

        Args:
            test_size: Proportion of the dataset to include in the test split. Default is 0.2.
            random_state: Seed for reproducible dataset shuffling. Default is 42.
            stratify: Whether to stratify split according to category labels. Default is True.

        Returns:
            dict: Summary containing result_id, train_size, test_size, and n_features.
        """
        if session.feature_matrix is None:
            return {"error": "No feature_matrix found in session. Please run build_dtm_tool first."}

        X = session.feature_matrix

        y = None
        if session.dataframe is not None and "category" in session.dataframe.columns:
            y = session.dataframe["category"].values
        elif session.dataframe is not None and "category_name" in session.dataframe.columns:
            y = session.dataframe["category_name"].values
        elif session.labels is not None:
            y = session.labels

        if y is None:
            return {"error": "No target labels found in session. Please load dataset first."}

        stratify_y = y if (stratify and len(np.unique(y)) > 1) else None

        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state, stratify=stratify_y
            )
        except ValueError:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state, stratify=None
            )

        session.artifacts["train_test_split"] = {
            "X_train": X_train,
            "X_test": X_test,
            "y_train": y_train,
            "y_test": y_test,
        }

        result_id = session.next_result_id("train_test_split")
        summary = {
            "result_id": result_id,
            "n_train_samples": int(X_train.shape[0]),
            "n_test_samples": int(X_test.shape[0]),
            "n_features": int(X_train.shape[1]),
            "test_size": test_size,
        }

        session.store_result("train_test_split_tool", {"test_size": test_size, "random_state": random_state}, summary)

        return summary

    @tool
    def naive_bayes_tool(
        alpha: float = 1.0,
        use_split: bool = True,
    ) -> dict:
        """Trains a Multinomial Naive Bayes classifier on the feature matrix and evaluates accuracy.

        Args:
            alpha: Additive (Laplace/Lidstone) smoothing parameter. Default is 1.0.
            use_split: Whether to use existing train/test split from artifacts. Default is True.

        Returns:
            dict: Summary containing result_id, accuracy, model parameters, and predictions.
        """
        if "train_test_split" in session.artifacts and use_split:
            split = session.artifacts["train_test_split"]
            X_train, X_test = split["X_train"], split["X_test"]
            y_train, y_test = split["y_train"], split["y_test"]
        elif session.feature_matrix is not None:
            X_train = X_test = session.feature_matrix
            if session.dataframe is not None and "category" in session.dataframe.columns:
                y_train = y_test = session.dataframe["category"].values
            elif session.dataframe is not None and "category_name" in session.dataframe.columns:
                y_train = y_test = session.dataframe["category_name"].values
            elif session.labels is not None:
                y_train = y_test = session.labels
            else:
                return {"error": "No labels found in session."}
        else:
            return {"error": "No feature_matrix or train_test_split found in session."}

        model = MultinomialNB(alpha=alpha)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        accuracy = float(accuracy_score(y_test, y_pred))

        session.artifacts["naive_bayes_model"] = model
        session.artifacts["predictions"] = y_pred.tolist()

        result_id = session.next_result_id("naive_bayes")
        summary = {
            "result_id": result_id,
            "model": "MultinomialNB",
            "alpha": alpha,
            "accuracy": round(accuracy, 4),
            "n_eval_samples": len(y_test),
        }

        session.store_result("naive_bayes_tool", {"alpha": alpha, "use_split": use_split}, summary)

        return summary

    return [train_test_split_tool, naive_bayes_tool]
