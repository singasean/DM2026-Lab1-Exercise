"""Pattern mining tools using PAMI algorithms for frequent itemset discovery."""

import os
import tempfile
from typing import List, Dict, Any, Optional
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from langchain_core.tools import tool


def make_tools(session):
    """Factory creating pattern mining tools bound to the given session state."""

    @tool
    def mine_patterns_tool(
        category_name: str,
        filtering_method: str = "tfidf",
        algorithm: str = "fpgrowth",
        min_sup: Optional[int] = None,
        k: Optional[int] = None
    ) -> Dict[str, Any]:
        """Mines frequent itemsets/word patterns for a single document category using PAMI algorithms.

        Args:
            category_name: Category label to mine patterns from.
            filtering_method: Feature selection approach before mining ('variance', 'tfidf', or 'term_frequency'). Defaults to 'tfidf'.
            algorithm: PAMI mining algorithm ('fpgrowth', 'topk', or 'maxfpgrowth'). Defaults to 'fpgrowth'.
            min_sup: Minimum support count for 'fpgrowth' and 'maxfpgrowth'.
            k: Number of top patterns to keep for 'topk' algorithm.

        Returns:
            Dict containing mined patterns, support counts, and execution metadata.
        """
        if session.dataframe is None:
            return {"status": "error", "message": "No dataset loaded in session."}

        # Filter dataframe to requested category
        cat_df = session.dataframe[session.dataframe["category_name"] == category_name]
        if cat_df.empty:
            return {"status": "error", "message": f"Category '{category_name}' not found or has no documents."}

        texts = cat_df["text"].dropna().tolist()
        if not texts:
            return {"status": "error", "message": f"No valid text found for category '{category_name}'."}

        # 1. Build category-local DTM dropping stopwords
        vectorizer = CountVectorizer(stop_words="english")
        try:
            X = vectorizer.fit_transform(texts).toarray()
            vocab = np.array(vectorizer.get_feature_names_out())
        except ValueError:
            # Empty vocabulary after stopword removal
            return {
                "category_name": category_name,
                "filtering_method": filtering_method,
                "algorithm": algorithm,
                "patterns": [],
                "note": "no terms survived stopword removal"
            }

        if X.shape[1] == 0:
            return {
                "category_name": category_name,
                "filtering_method": filtering_method,
                "algorithm": algorithm,
                "patterns": [],
                "note": "no terms survived stopword removal"
            }

        # 2. Vocabulary filtering
        if filtering_method == "variance":
            metrics = np.var(X, axis=0)
            p5, p95 = np.percentile(metrics, [5, 95])
            keep_idx = np.where((metrics > p5) & (metrics < p95))[0]
        elif filtering_method == "term_frequency":
            metrics = X.sum(axis=0)
            p5, p95 = np.percentile(metrics, [5, 95])
            keep_idx = np.where((metrics > p5) & (metrics < p95))[0]
        elif filtering_method == "tfidf":
            transformer = TfidfTransformer()
            tfidf_matrix = transformer.fit_transform(X).toarray()
            metrics = tfidf_matrix.mean(axis=0)
            p20 = np.percentile(metrics, 20)
            keep_idx = np.where(metrics >= p20)[0]
        else:
            return {"status": "error", "message": f"Unknown filtering method '{filtering_method}'."}

        # 3. Handle zero surviving terms edge case
        if len(keep_idx) == 0:
            result_id = session.next_result_id("mine_patterns")
            summary = {
                "result_id": result_id,
                "category_name": category_name,
                "filtering_method": filtering_method,
                "algorithm": algorithm,
                "patterns": [],
                "note": "no terms survived filtering"
            }
            session.store_result("mine_patterns_tool", {"category_name": category_name, "filtering_method": filtering_method, "algorithm": algorithm}, summary)
            return summary

        surviving_vocab = vocab[keep_idx]
        surviving_X = X[:, keep_idx]

        # 4. Convert surviving terms to transactional database file for PAMI
        transactions = []
        for row in surviving_X:
            items = [surviving_vocab[j] for j in range(len(surviving_vocab)) if row[j] >= 1]
            if items:
                transactions.append("\t".join(items))

        if not transactions:
            result_id = session.next_result_id("mine_patterns")
            summary = {
                "result_id": result_id,
                "category_name": category_name,
                "filtering_method": filtering_method,
                "algorithm": algorithm,
                "patterns": [],
                "note": "no items present in transactions"
            }
            session.store_result("mine_patterns_tool", {"category_name": category_name, "filtering_method": filtering_method, "algorithm": algorithm}, summary)
            return summary

        temp_db = tempfile.NamedTemporaryFile(mode="w+", delete=False, suffix=".txt")
        try:
            for t in transactions:
                temp_db.write(t + "\n")
            temp_db.flush()
            temp_db.close()

            # 5. Execute requested PAMI algorithm
            patterns_list = []
            if algorithm == "fpgrowth":
                if min_sup is None:
                    return {"status": "error", "message": "min_sup is required for fpgrowth."}
                from PAMI.frequentPattern.basic import FPGrowth
                miner = FPGrowth.FPGrowth(iFile=temp_db.name, minSup=min_sup, sep="\t")
                miner.startMine()
                patterns_df = miner.getPatternsAsDataFrame()

            elif algorithm == "topk":
                if k is None:
                    return {"status": "error", "message": "k is required for topk."}
                from PAMI.frequentPattern.topk import FAE
                miner = FAE.FAE(iFile=temp_db.name, k=k, sep="\t")
                miner.startMine()
                patterns_df = miner.getPatternsAsDataFrame()

            elif algorithm == "maxfpgrowth":
                if min_sup is None:
                    return {"status": "error", "message": "min_sup is required for maxfpgrowth."}
                from PAMI.frequentPattern.maximal import MaxFPGrowth
                miner = MaxFPGrowth.MaxFPGrowth(iFile=temp_db.name, minSup=min_sup, sep="\t")
                miner.startMine()
                patterns_df = miner.getPatternsAsDataFrame()

            else:
                return {"status": "error", "message": f"Unknown algorithm '{algorithm}'."}

            if patterns_df is not None and not patterns_df.empty:
                for _, row in patterns_df.iterrows():
                    pattern_str = str(row["Patterns"])
                    supp = int(row["Support"])
                    items = [item.strip() for item in pattern_str.split("\t") if item.strip()]
                    patterns_list.append({"pattern": items, "support": supp})

        finally:
            if os.path.exists(temp_db.name):
                os.remove(temp_db.name)

        # Store artifacts in session
        artifact_key = f"patterns_{category_name}_{filtering_method}_{algorithm}"
        session.artifacts[artifact_key] = patterns_list

        result_id = session.next_result_id("mine_patterns")
        summary = {
            "result_id": result_id,
            "category_name": category_name,
            "filtering_method": filtering_method,
            "algorithm": algorithm,
            "patterns_count": len(patterns_list),
            "patterns": patterns_list
        }

        session.store_result("mine_patterns_tool", {"category_name": category_name, "filtering_method": filtering_method, "algorithm": algorithm}, summary)
        return summary

    return [mine_patterns_tool]
