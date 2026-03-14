from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
from omegaconf import OmegaConf

from train_pictime.eval.embed import extract_embeddings, load_paths
from train_pictime.eval.metrics_views import evaluate_views_pack
from train_pictime.eval.metrics_rank import embedding_variance_and_effective_rank
from train_pictime.eval.metrics_geometry import geometry_pack
from train_pictime.eval.metrics_prototype import prototype_utilization

from train_pictime.wandb_logger import log_wandb, log_paired, log_paired_variant


@dataclass
class EvalConfig:
    cfg: Any  # OmegaConf DictConfig


def load_eval_config(path: str) -> EvalConfig:
    return EvalConfig(cfg=OmegaConf.load(path))


class Evaluator:
    """
    In-process evaluator. Call maybe_eval(model, it) from the training loop.
    It decides what to run based on iteration, reuses embeddings across metrics,
    and logs everything to W&B.
    """

    def __init__(self, eval_cfg: EvalConfig, train_cfg: Any, wandb_run: Any):
        self.ecfg = eval_cfg.cfg
        self.tcfg = train_cfg
        self.run = wandb_run

        self.device = str(self.ecfg.get("device", "cuda"))
        self.bs = int(self.ecfg.get("batch_size", 64))

        # Resolve UVAL path relative to repo root
        self.uval_paths = load_paths(str(self._resolve_repo_path(self.ecfg.uval.path)))

        self.cascade = bool(self.ecfg.get("cascade", {}).get("enabled", True))

        self.freq = self.ecfg.frequency
        self.sizes = self.ecfg.sizes

        self.geom_cfg = self.ecfg.geom
        self.rank_cfg = self.ecfg.rank

    def _resolve_repo_path(self, p: str) -> Path:
        """
        Resolve relative paths against repo root (.../dinov3).
        """
        here = Path(__file__).resolve()
        repo_root = here.parents[2]  # train_pictime/eval/evaluator.py -> .../dinov3
        pp = Path(p)
        return pp if pp.is_absolute() else (repo_root / pp).resolve()

    def _due(self, key: str, it: int) -> bool:
        every = int(self.freq.get(key, 0))
        return every > 0 and (it % every == 0) and it>0

    def _plan(self, it: int) -> Dict[str, bool]:
        """
        Decide what to run this iteration, with optional cascading.
        """
        do_views = self._due("views_pack", it)
        do_rank  = self._due("rank_pack", it)
        do_proto = self._due("proto_pack", it)
        do_geom  = self._due("geom_pack", it)
        any_do = do_views or do_rank or do_proto or do_geom
        if self.cascade and any_do:
            max_pack_size = max(int(self.sizes.geom), int(self.sizes.rank), int(self.sizes.proto), int(self.sizes.views))
            do_views = do_views or (int(self.sizes.views) <= max_pack_size)
            do_rank  = do_rank  or (int(self.sizes.rank)  <= max_pack_size)
            do_proto = do_proto or (int(self.sizes.proto) <= max_pack_size)
            do_geom  = do_geom  or (int(self.sizes.geom)  <= max_pack_size)

        return {"views": do_views, "rank": do_rank, "proto": do_proto, "geom": do_geom}

    def _embed_need(self, plan: Dict[str, bool]) -> int:
        """
        Determine how many single-view embeddings we need (shared by rank+geom).
        """
        need = 0
        for key in plan.keys():
            if plan[key] and int(self.sizes[key]) > need:
                need = int(self.sizes[key])
        return need

    @torch.inference_mode()
    def maybe_eval(self, model: Any, it: int) -> None:
        """
        Called from training loop after optimizer.step(). Does nothing unless due.
        """
        plan = self._plan(it)
        if not any(plan.values()):
            return

        prefix = str(self.ecfg.get("logging", {}).get("prefix", "eval/"))

        # EMBEDDINGS
        need_embed = self._embed_need(plan)
        # compute once at max required size; smaller metrics use prefixes
        E_t = extract_embeddings(model, self.uval_paths[:need_embed], which="teacher", batch_size=self.bs, device=self.device)
        E_s = extract_embeddings(model, self.uval_paths[:need_embed], which="student", batch_size=self.bs, device=self.device)

        # PROTO
        if plan["proto"]:
            Et = E_t[:self.sizes.proto]
            Es = E_s[:self.sizes.proto]
            proto_t = prototype_utilization(model, Et, which="teacher", batch_size=self.bs, device=self.device,)
            proto_s = prototype_utilization(model, Es, which="student", batch_size=self.bs, device=self.device,)
            log_paired(self.run, step=it, prefix=f"{prefix}proto/", teacher_dict=proto_t, student_dict=proto_s)

        # RANK
        if plan["rank"]:
            Et = E_t[:self.sizes.rank]
            Es = E_s[:self.sizes.rank]

            rank_t_raw = embedding_variance_and_effective_rank(Et, center_and_renorm=False)
            rank_s_raw = embedding_variance_and_effective_rank(Es, center_and_renorm=False)

            if bool(self.rank_cfg.get("centered", True)):
                rank_t_ctr = embedding_variance_and_effective_rank(Et, center_and_renorm=True)
                rank_s_ctr = embedding_variance_and_effective_rank(Es, center_and_renorm=True)

                log_paired_variant(self.run, step=it, prefix=f"{prefix}rank/",
                                   teacher_variants={"raw": rank_t_raw, "ctr": rank_t_ctr},
                                   student_variants={"raw": rank_s_raw, "ctr": rank_s_ctr})

            else:
                log_paired(self.run, step=it, prefix=f"{prefix}rank/", teacher_dict=rank_t_raw, student_dict=rank_s_raw)

        # GEOM
        if plan["geom"]:
            Et = E_t[:self.sizes.geom]
            Es = E_s[:self.sizes.geom]

            ks = tuple(int(x) for x in self.geom_cfg.get("ks", [1, 5, 10]))
            geom_t_raw = geometry_pack(Et, num_pairs=self.geom_cfg.num_pairs, ks=ks, device=self.device, center_and_renorm=False)
            geom_s_raw = geometry_pack(Es, num_pairs=self.geom_cfg.num_pairs, ks=ks, device=self.device, center_and_renorm=False)

            if bool(self.geom_cfg.get("centered", True)):
                geom_t_ctr = geometry_pack(Et, num_pairs=self.geom_cfg.num_pairs, ks=ks, device=self.device, center_and_renorm=True)
                geom_s_ctr = geometry_pack(Es, num_pairs=self.geom_cfg.num_pairs, ks=ks, device=self.device, center_and_renorm=True)

                log_paired_variant(self.run, step=it, prefix=f"{prefix}geom/",
                                   teacher_variants={"raw": geom_t_raw, "ctr": geom_t_ctr},
                                   student_variants={"raw": geom_s_raw, "ctr": geom_s_ctr})

            else:
                log_paired(self.run, step=it, prefix=f"{prefix}geom/", teacher_dict=geom_t_raw, student_dict=geom_s_raw)

        # VIEWS (separate forward passes; doesn’t reuse E_t/E_s because it uses 2 views (crop of the original img))
        if plan["views"]:
            vt = evaluate_views_pack(model, self.uval_paths[:self.sizes.views], which="teacher")
            vs = evaluate_views_pack(model, self.uval_paths[:self.sizes.views], which="student")
            log_paired(self.run, step=it, prefix=f"{prefix}views/", teacher_dict=vt, student_dict=vs)

        # marker that eval ran
        log_wandb(self.run, {"eval/iter": it}, step=it)