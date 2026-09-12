import numpy as np
import pandas as pd


def eval_open_set(distmat, q_pids, g_pids, fpir_target=0.01, tpr_target=0.95):
    """
    Evaluates Open-Set Performance: 
    1. DIR @ FPIR (Fix False Positives, measure Detection & ID)
    2. TNR @ TPR  (Fix True Positives, measure True Negatives)
    
    Args:
        distmat: (num_query, num_gallery) matrix. LOWER is better (Distance).
        q_pids:  Target identities for query images.
        g_pids:  Target identities for gallery images.
        fpir_target: Target False Positive Identification Rate (default 1%).
        tpr_target:  Target True Positive Rate for TNR calculation (default 95%).
        
    Returns:
        dir_score:      Detection and Identification Rate at the target FPIR.
        fpir_threshold: The similarity threshold determined by FPIR.
        tnr_score:      True Negative Rate at the target TPR.
        tpr_threshold:  The similarity threshold determined by TPR.
    """

    # 1. Convert Distance back to Similarity (Higher is Better)
    sim_mat = -distmat
    sim_mat = np.asarray(sim_mat, dtype=np.float64)
    sim_mat = np.nan_to_num(
        sim_mat,
        neginf=-1e6,
        posinf=1e6,
        nan=-1e6
    )

    # 2. Separate Known Queries vs. Impostor Queries
    # Check which query IDs actually exist in the gallery
    g_pids_set = set(g_pids)
    is_known_mask = np.array([pid in g_pids_set for pid in q_pids])
    
    known_indices = np.where(is_known_mask)[0]
    impostor_indices = np.where(~is_known_mask)[0]
    
    if len(impostor_indices) == 0:
        print("Warning: No impostors found in query set. Cannot calculate FPIR.")
        return 1.0, 0.0, 0.0, 0.0
    if len(known_indices) == 0:
        print("Warning: No knowns found in query set. Cannot calculate TNR.")
        return 0.0, 0.0, 1.0, 0.0
    
    # --------------------------------------------------------
    # PART A: DIR @ FPIR
    # --------------------------------------------------------
    
    # We only care about the highest score an impostor achieves against the gallery
    impostor_max_scores = np.max(sim_mat[impostor_indices], axis=1)
    impostor_max_scores = impostor_max_scores[np.isfinite(impostor_max_scores)]

    if len(impostor_max_scores) == 0:
        raise ValueError("All impostor scores are non-finite. Check similarity construction.")

    impostor_sorted_desc = np.sort(impostor_max_scores)[::-1]
    
    # Find threshold where top X% of impostors are falsely accepted
    num_impostors = len(impostor_sorted_desc)
    fpir_index = int(np.ceil(num_impostors * fpir_target)) - 1
    fpir_index = max(0, fpir_index)
    
    fpir_threshold = impostor_sorted_desc[fpir_index]

    # Calculate DIR for Knowns    
    known_sims = sim_mat[known_indices]
    
    pred_indices = np.argmax(known_sims, axis=1)
    pred_scores = np.max(known_sims, axis=1)
    
    pred_pids = g_pids[pred_indices]
    actual_pids = q_pids[known_indices]
    
    # DIR Conditions: Correct ID AND Score >= FPIR Threshold
    is_correct_id = (pred_pids == actual_pids)
    is_above_threshold = (pred_scores >= fpir_threshold)
    
    # DIR = Fraction of knowns that pass both checks
    correct_detections = np.sum(is_correct_id & is_above_threshold)
    dir_score = correct_detections / len(known_indices)
    
    # --------------------------------------------------------
    # PART B: TNR @ TPR
    # --------------------------------------------------------
    
    # For TNR@TPR, we treat this as a binary detection task (Known vs Unknown).
    # We use the max scores of the Known queries (Positives).
    
    # Sort Known scores Low to High to find the cutoff
    known_scores_sorted_asc = np.sort(pred_scores)
    num_knowns = len(known_scores_sorted_asc)

    # We want to keep TPR fraction of knowns (e.g., top 95%)
    # So we reject the bottom (1 - TPR) fraction.
    tpr_index = int(np.floor(num_knowns * (1 - tpr_target)))
    tpr_index = max(0, min(tpr_index, num_knowns - 1))
    
    tpr_threshold = known_scores_sorted_asc[tpr_index]

    # TNR: Fraction of Impostors that are BELOW this TPR threshold
    # (True Negatives = Impostors correctly rejected)
    true_negatives = np.sum(impostor_max_scores < tpr_threshold)
    tnr_score = true_negatives / num_impostors

    return dir_score, fpir_threshold, tnr_score, tpr_threshold

def eval_func(distmat, q_pids, g_pids, q_camids, g_camids, max_rank=50):
    num_q, num_g = distmat.shape      # dimension of the distance matrix (query x gallery)

    # distmat g
    #    q    1 3 2 4
    #         4 1 2 3
    if num_g < max_rank:
        max_rank = num_g
        print("Note: number of gallery samples is quite small, got {}".format(num_g))

    indices = np.argsort(distmat, axis=1)
    #  0 2 1 3
    #  1 2 3 0
    matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)

    # compute cmc curve for each query
    all_cmc = []
    all_AP = []
    per_query = [] # collect per-query details
    num_valid_q = 0.    # number of valid query
    for q_idx in range(num_q):
        # get query pid and camid
        q_pid = q_pids[q_idx]
        q_camid = q_camids[q_idx]

        # compute cmc curve
        # binary vector, positions with value 1 are correct matches
        #orig_cmc = matches[q_idx][keep]
        orig_cmc = matches[q_idx]
        
        if not np.any(orig_cmc):
            # this condition is true when query identity does not appear in gallery
            per_query.append({
                "q_idx": q_idx,
                "q_pid": int(q_pid),
                "q_camid": int(q_camid),
                "valid": False,
                "AP": 0.0,
                "rank_of_first_match": -1,
                "ranks_of_all_matches": [],
            })
            continue

        cmc = orig_cmc.cumsum()
        cmc[cmc > 1] = 1

        all_cmc.append(cmc[:max_rank])
        num_valid_q += 1.

        # compute average precision
        # reference: https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Average_precision
        num_rel = orig_cmc.sum()
        tmp_cmc = orig_cmc.cumsum()
        
        #tmp_cmc = [x / (i + 1.) for i, x in enumerate(tmp_cmc)]
        y = np.arange(1, tmp_cmc.shape[0] + 1) * 1.0
        tmp_cmc = tmp_cmc / y
        tmp_cmc = np.asarray(tmp_cmc) * orig_cmc
        AP = tmp_cmc.sum() / num_rel

        all_AP.append(AP)
        ranks = np.where(orig_cmc == 1)[0]
        first_rank = int(ranks[0]) if ranks.size > 0 else -1
        
        per_query.append({
            "q_idx": q_idx,
            "q_pid": int(q_pid),
            "q_camid": int(q_camid),
            "valid": True,
            "AP": float(AP),
            "rank_of_first_match": first_rank,
            "ranks_of_all_matches": ranks.tolist(),
        })

    assert num_valid_q > 0, "Error: all query identities do not appear in gallery"

    all_cmc = np.asarray(all_cmc).astype(np.float32)
    all_cmc = all_cmc.sum(0) / num_valid_q
    mAP = np.mean(all_AP)

    return all_cmc, mAP, per_query, indices


def prepare_bins(df):
    """Prepare time bins for temporal analysis."""
    df = df.copy()
    year = 365
    max_days = int(df['abs_diff_days'].max())
    max_edge = ((max_days // year + 1) * year + 1)
    
    # Create bin edges: [0, 1), [1, 366), [366, 731), ..., [N, inf)
    edges = [0, 1] + list(range(year + 1, max_edge, year))
    edges.append(np.inf)
    
    # Create labels: 0, 1, 2, ..., N+
    labels = []
    for i in range(len(edges) - 1):
        left, right = edges[i], edges[i + 1]
        if left == 0 and right == 1:
            labels.append("0")
        else:
            yrs = int(np.ceil((left - 1) / year)) + 1
            labels.append(f"{yrs}")
    
    df['time_bin'] = pd.cut(
        df['abs_diff_days'],
        bins=edges,
        right=False,
        labels=labels,
        include_lowest=True
    )
    
    return df


class CustomMetric:
    """Custom metrics for wildlife re-identification."""
    
    def __init__(self, agg_method="by_id", id_col="identity", time_bin_col="time_bin"):
        self.agg_method = agg_method
        self.id_col = id_col
        self.time_bin_col = time_bin_col
    
    def compute_standard_map(self, df):
        """Compute standard mean Average Precision."""
        return df["AP"].mean()
    
    def compute_t_ndcg(self, df):
        """
        Compute time-weighted normalized Discounted Cumulative Gain (t-nDCG).
        
        This metric weights performance by temporal distance, giving higher
        importance to more recent queries.
        """
        # Step 1: Calculate per-bin scores
        if self.agg_method == "by_id":
            s = df.groupby([self.id_col, self.time_bin_col])["AP"].mean().dropna()
            mean_per_time_bin = s.groupby(level=self.time_bin_col).mean()
        elif self.agg_method == "by_query":
            s = df.groupby([self.time_bin_col])["AP"].mean().dropna()
            mean_per_time_bin = s
        else:
            raise ValueError(f"Unknown aggregation method: {self.agg_method}")
        
        # Step 2: Get counts and define weights
        count_per_time_bin = s.groupby(level=self.time_bin_col).size()
        
        # Weights decay logarithmically with time
        present_bins = count_per_time_bin.index.astype(int)
        per_time_bin_weight = 1 / np.log(present_bins + 2)
        
        # Step 3: Calculate DCG (Discounted Cumulative Gain)
        dcg = np.sum(count_per_time_bin * mean_per_time_bin * per_time_bin_weight)
        
        # Step 4: Calculate IDCG (Ideal DCG with perfect AP=1.0)
        idcg = np.sum(count_per_time_bin * 1.0 * per_time_bin_weight)
        
        if idcg == 0:
            return 0.0
        
        t_ndcg = dcg / idcg
        return t_ndcg
    
    def compute_robustness_score(self, df):
        """
        Compute robustness score measuring performance degradation over time.
        
        Note: This metric requires multiple timeframes per identity and may
        not be applicable to all datasets.
        """
        if self.agg_method == "by_id":
            s = (
                df
                .groupby(["identity", "time_bin"])["AP"]
                .mean()
                .sort_index(level=["identity", "time_bin"])
            )
            
            # Compute relative change from first bin
            robustness = s - s.groupby(level="identity").transform("first")
            denom = s.groupby(level="identity").transform("first") + 1e-3
            robustness = robustness / denom
            robustness = robustness.clip(lower=0, upper=1).rename("AP").reset_index().dropna()
        else:
            raise ValueError(f"Robustness score only supported for agg_method='by_id'")
        
        # Compute t-nDCG of robustness values
        robustness_score = self.compute_t_ndcg(
            robustness.set_index([self.id_col, self.time_bin_col])
        )
        return robustness_score