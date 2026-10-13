"""
Unit tests for the PeerMatcher subsystem.
Verifies self-exclusion, reference pool restriction, determinism, and configurable k.
"""

import json
import unittest
import numpy as np
import pandas as pd

from src.peers.peer_matcher import PeerMatcher


class TestPeerMatching(unittest.TestCase):
    """Test suite verifying correctness and constraints of the peer matching algorithm."""

    @classmethod
    def setUpClass(cls):
        cls.df_model = pd.read_csv("data/processed/modeling_dataset.csv")
        cls.df_peers = pd.read_csv("data/processed/peer_features.csv")
        with open("data/splits/splits_benchmark.json") as f:
            cls.splits = json.load(f)
        cls.train_set = set(cls.splits["train"])
        cls.primary_test_set = set(cls.splits["primary_test"])
        cls.secondary_test_set = set(cls.splits["secondary_test"])

    def test_battery_never_matches_itself(self):
        """Rule 1: A battery cell must NEVER match itself as a peer."""
        for _, row in self.df_peers.iterrows():
            cid = str(row["cell_id"])
            peer_ids_str = str(row.get("peer_ids", ""))
            if pd.isna(peer_ids_str) or not peer_ids_str or peer_ids_str == "nan":
                continue
            peer_list = [p.strip() for p in peer_ids_str.split(",") if p.strip()]
            self.assertNotIn(
                cid,
                peer_list,
                f"Self-match violation! Battery {cid} matched itself at cutoff {row['cutoff']}",
            )

    def test_test_cells_use_only_training_reference_pool(self):
        """Rule 2: Primary and secondary test batteries must select peers EXCLUSIVELY from training pool."""
        test_rows = self.df_peers[
            self.df_peers["benchmark_partition"].isin(["primary_test", "secondary_test"])
        ]
        for _, row in test_rows.iterrows():
            cid = str(row["cell_id"])
            peer_ids_str = str(row.get("peer_ids", ""))
            if not peer_ids_str or pd.isna(peer_ids_str):
                continue
            peer_list = [p.strip() for p in peer_ids_str.split(",") if p.strip()]
            for pid in peer_list:
                self.assertIn(
                    pid,
                    self.train_set,
                    f"Leakage violation! Test cell {cid} selected non-training peer {pid} (partition: {row['benchmark_partition']})",
                )

    def test_training_cells_use_only_other_training_cells(self):
        """Training cells must select peers from training reference pool (excluding themselves)."""
        train_rows = self.df_peers[self.df_peers["benchmark_partition"] == "train"]
        for _, row in train_rows.iterrows():
            cid = str(row["cell_id"])
            peer_ids_str = str(row.get("peer_ids", ""))
            peer_list = [p.strip() for p in peer_ids_str.split(",") if p.strip()]
            self.assertEqual(len(peer_list), 5)
            for pid in peer_list:
                self.assertIn(pid, self.train_set)
                self.assertNotEqual(pid, cid)

    def test_deterministic_matching(self):
        """Rule 8: Identical inputs and configuration must produce identical peer selections."""
        df_cut100 = self.df_model[self.df_model["cutoff"] == 100].copy().reset_index(drop=True)
        train_ref = df_cut100[df_cut100["benchmark_partition"] == "train"].copy().reset_index(drop=True)

        m1 = PeerMatcher(k=5).fit(train_ref)
        m2 = PeerMatcher(k=5).fit(train_ref)

        sample_cell = df_cut100[df_cut100["cell_id"] == "b1c0"].iloc[0]
        res1 = m1.find_peers_for_cell(sample_cell)
        res2 = m2.find_peers_for_cell(sample_cell)

        self.assertEqual(res1["peer_ids"], res2["peer_ids"])
        np.testing.assert_allclose(res1["peer_distances"], res2["peer_distances"], rtol=1e-7)

    def test_configurable_k(self):
        """PeerMatcher must correctly support configurable peer group sizes (e.g. k=3, k=7)."""
        df_cut100 = self.df_model[self.df_model["cutoff"] == 100].copy().reset_index(drop=True)
        train_ref = df_cut100[df_cut100["benchmark_partition"] == "train"].copy().reset_index(drop=True)
        sample_cell = df_cut100[df_cut100["cell_id"] == "b1c0"].iloc[0]

        for test_k in [3, 7]:
            matcher = PeerMatcher(k=test_k).fit(train_ref)
            res = matcher.find_peers_for_cell(sample_cell)
            self.assertEqual(res["peer_count"], test_k)
            self.assertEqual(len(res["peer_ids"]), test_k)
            self.assertEqual(len(res["peer_distances"]), test_k)

    def test_insufficient_peers_rule(self):
        """Rule 10: If valid candidate reference count < min_k, status must be INSUFFICIENT_PEERS."""
        # Create a tiny artificial reference pool with only 2 candidates
        df_cut100 = self.df_model[self.df_model["cutoff"] == 100].copy()
        tiny_ref = df_cut100[df_cut100["benchmark_partition"] == "train"].head(2).copy()

        matcher = PeerMatcher(k=5, min_k=3).fit(tiny_ref)
        sample_cell = df_cut100[df_cut100["cell_id"] == "b1c0"].iloc[0]
        res = matcher.find_peers_for_cell(sample_cell)

        self.assertEqual(res["peer_match_status"], "INSUFFICIENT_PEERS")
        self.assertLess(res["peer_count"], 3)

    def test_excluded_cells_status(self):
        """Excluded benchmark cells must have EXCLUDED_CELL status and zero matched peers."""
        ex_rows = self.df_peers[self.df_peers["benchmark_partition"] == "excluded"]
        self.assertEqual(len(ex_rows), 48)  # 16 cells * 3 cutoffs
        for _, row in ex_rows.iterrows():
            self.assertEqual(row["peer_match_status"], "EXCLUDED_CELL")
            self.assertEqual(row["peer_count"], 0)
            self.assertTrue(pd.isna(row["peer_ids"]) or row["peer_ids"] == "")


if __name__ == "__main__":
    unittest.main()
