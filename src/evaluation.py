import numpy as np
import pandas as pd
from loguru import logger

from src.data import SubsettableImageDataset
from src.metrics import CustomMetric, eval_func, eval_open_set


class EvaluationPipeline:
    """Handles model evaluation."""
    
    def __init__(self, cfg):
        self.cfg = cfg
        self.metric_computer = CustomMetric(agg_method=cfg.get("agg_method", "by_id"))
        self.all_results = []
        
    def evaluate(self, model, full_df, data_manager, fpir_targets, imposter_df=None, sim_out_filename=""):
        df = data_manager.prepare_gq_split(full_df)
        
        if df.empty or df[df["split"] == "gallery"]["identity"].nunique() < 5:
            logger.warning(f"Skipping: insufficient data")
            return {}

        # Add imposters if needed
        if fpir_targets and imposter_df is not None:
            df = pd.concat([df, imposter_df])

        df = df.reset_index(drop=True)
        dataset = SubsettableImageDataset(
            metadata=df, root="", col_path="path", col_label="identity"
        )
        
        gallery_indices = df.index[df["split"] == "gallery"].tolist()
        query_indices = df.index[df["split"] == "query"].tolist()
        
        dataset_gallery = dataset.subset(gallery_indices)
        dataset_query = dataset.subset(query_indices)
        
        # Compute similarities
        similarity_scores = model(dataset_query, dataset_gallery)

        if sim_out_filename:
            # Create a Mapping: Animal ID -> Gallery Column Index
            # This tells us: "Turtle ID 55 is located at Column 3 of the matrix"
            gid_to_col = {pid: i for i, pid in enumerate(dataset_gallery.labels)}

            query_metadata: pd.DataFrame = dataset_query.metadata.copy()
            final_scores = np.zeros(len(dataset_query))
            
            # --- HANDLE GENUINE ---
            # Logic: Get the score of the SPECIFIC ground truth match
            gen_mask = query_metadata["role"] == "genuine"
            gen_pids = dataset_query.labels[gen_mask]
            # Get the correct column index for each genuine query
            target_cols = [gid_to_col[pid] for pid in gen_pids]
            final_scores[gen_mask] = similarity_scores[gen_mask, target_cols]

            # --- HANDLE IMPOSTOR ---
            # Logic: Get the MAX score (nearest neighbor)
            imp_mask = query_metadata["role"] != "genuine"
            final_scores[imp_mask] = similarity_scores[imp_mask].max(axis=1)

            query_metadata["sim_analysis"] = final_scores
            
            best_match_indices = similarity_scores.argmax(axis=1)
            # labels is the integer label, labels_map is the string label
            pred_int_labels = dataset_gallery.labels[best_match_indices]
            query_metadata["pred_identity"] = dataset_gallery.labels_map[pred_int_labels]

            query_metadata.to_csv(sim_out_filename, index=False)

        distmat = -similarity_scores
        
        # Evaluate
        q_pids = dataset_query.labels
        g_pids = dataset_gallery.labels
        q_camids = np.zeros(len(dataset_query))
        g_camids = np.zeros(len(dataset_gallery))
        # g_pids[indices] == q_pids[:, np.newaxis]
        result = {}
        # Open-set evaluation
        if fpir_targets:
            for fpir_target in fpir_targets:
                dir_at_fpir, fpir_threshold, tnr_at_tpr, tpr_threshold = eval_open_set(distmat, q_pids, g_pids, fpir_target=fpir_target)
                result[f"DIR@FPIR={fpir_target:.2f}"] = float(round(dir_at_fpir, 4))
                result[f"Thresh@FPIR={fpir_target:.2f}"] = float(round(fpir_threshold, 4))
                result[f"TNR@TPR"] = float(round(tnr_at_tpr, 4))
                result[f"Thresh@TPR"] = float(round(tpr_threshold, 4))
        
        # Closed-set evaluation
        g_pids_set = set(g_pids)
        is_known = np.array([pid in g_pids_set for pid in q_pids])
        distmat_known = distmat[is_known]
        q_pids_known = q_pids[is_known]
        q_camids_known = q_camids[is_known]
        
        cmc, mAP, per_query, indices = eval_func(
            distmat_known, q_pids_known, g_pids, q_camids_known, g_camids)
        
        # debug: save images of some gallery-query match for checking
        if self.cfg.get("save_match_images", False):
            self._save_match_images(
                distmat, dataset_query, dataset_gallery, indices, "debug")
        
        result.update({
            "mAP": float(round(mAP, 4)),
            "Q": len(dataset_query),
            "G": len(dataset_gallery),
            "identity": df["identity"].nunique(),
            "Q(C)": ((df["split"] == "query") & (df["role"] == "genuine")).sum(),
            "Q(O)": ((df["split"] == "query") & (df["role"].str.startswith("imposter"))).sum(),
        })
        
        for r in [1, 5, 10]:
            if r <= len(cmc):
                result[f"CMC@{r}"] = float(round(cmc[r - 1], 4))
        
        self.all_results.append(result)

        return result
    
    def evaluate_per_view(self, model, full_df, data_manager, fpir_targets, imposter_df=None):
        """Evaluation by view."""
        logger.info("Running per-view evaluation")

        for orientation in sorted(full_df["orientation"].unique()):
            logger.info(f"Evaluating orientation: {orientation}")

            df = full_df[full_df["orientation"] == orientation].copy()
            impostor_df_view = imposter_df
            # ort = orientation[0].upper()
            # if orientation == "topleft":
            #     ort = "L"
            # elif orientation == "topright":
            #     ort = "R"

            # impostor_df_view = imposter_df[imposter_df["orientation"] == ort].copy()
            # print(f"Ori num: {len(imposter_df)}")
            # print(f"Num impostor views: {len(impostor_df_view)}")
            
            result = self.evaluate(
                model=model,
                full_df=df,
                data_manager=data_manager,
                fpir_targets=fpir_targets,
                imposter_df=impostor_df_view,
                sim_out_filename=f"query_metadata_{orientation}.csv"
            )
            if not result:
                continue

            result["orientation"] = orientation
            self.all_results[-1] = result

            self._log_results(result)
        
    def complete_evaluation(self, output_dir):
        # Compute overall results
        overall_results = self._compute_overall_results(self.all_results)
        
        logger.info("=" * 80)
        logger.info("Overall Results")
        logger.info("=" * 80)
        self._log_results(overall_results)
        
        # Save results
        results_df = pd.DataFrame(self.all_results)
        overall_results_df = pd.DataFrame([overall_results])
        overall_results_df["orientation"] = "overall"
        results_df = pd.concat([results_df, overall_results_df], ignore_index=True)

        # arrange the sequence to CMC@1, mAP, TNR@TPR, DIR@FPIR=x + the rest 
        columns_order = ["CMC@1", "mAP", "TNR@TPR"]
        columns_order += [x for x in overall_results_df.columns if x.startswith("DIR@FPIR")]
        columns_order += [col for col in overall_results_df.columns if col not in columns_order]
        results_df = results_df[columns_order]

        results_df.to_csv(output_dir / "overall_results.csv", index=False)
        
        return overall_results

    def _compute_overall_results(self, all_results):
        """Compute weighted overall results."""
        result_df = pd.DataFrame(all_results)
        
        overall_results = {}
        for col in result_df.columns:
            # skip non-float columns
            if not isinstance(result_df[col].iloc[0], float) or col in ["G", "Q(C)", "Q(O)"]:
                continue
            elif col == "TNR@TPR" or col.startswith("Thresh@FPIR"):
                overall_results[col] = (result_df[col] * result_df["Q(O)"]).sum() / result_df["Q(O)"].sum()
            else: # CMC@r, DIR@FPIR=x, mAP
                overall_results[col] = (result_df[col] * result_df["Q(C)"]).sum() / result_df["Q(C)"].sum()
        
        overall_results["G"] = result_df["G"].sum()
        overall_results["Q(C)"] = result_df["Q(C)"].sum()
        overall_results["Q(O)"] = result_df["Q(O)"].sum()
        
        return overall_results
    
    def _log_results(self, results):
        """Log results in a formatted way."""
        logger.info("-" * 80)
        for key, value in results.items():
            if isinstance(value, float):
                if key.startswith("Thresh"):
                    logger.info(f"{key}: {value:.4f}")
                else:
                    logger.info(f"{key}: {value:.1%}")
            else:
                logger.info(f"{key}: {value}")
        logger.info("-" * 80)

    def _save_match_images(self, distmat, dataset_query, dataset_gallery, indices, output_dir):
        import os
        import torchvision.transforms as T
        """Save images to check gallery-query label"""
        os.makedirs(output_dir, exist_ok=True)
        
        for q_idx, g_indices in enumerate(indices):
            # return rgb image
            q_img, q_pid = dataset_query[q_idx]
            # denormalize
            q_img = T.Normalize(mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
                                std=[1/0.229, 1/0.224, 1/0.225])(q_img)
            q_img = T.ToPILImage()(q_img)
            
            
            for g_idx in g_indices[:5]:  # Save top 5 matches
                g_img, g_pid = dataset_gallery[g_idx]
                # denormalize
                g_img = T.Normalize(mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
                                    std=[1/0.229, 1/0.224, 1/0.225])(g_img)
                g_img = T.ToPILImage()(g_img)
                
                if q_pid == g_pid:
                    filename = f"q_{q_idx}_g_{g_idx}_d{distmat[q_idx, g_idx]:.4f}_q.jpg"
                    q_img.save(os.path.join(output_dir, filename))
                    filename = filename.replace("_q.jpg", "_g.jpg")
                    g_img.save(os.path.join(output_dir, filename))
