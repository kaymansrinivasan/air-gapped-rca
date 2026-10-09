import unittest

from src.rca_local.workers import keyword_scores, rank_candidates


class RetrievalTests(unittest.TestCase):
    def test_exact_failure_terms_can_change_candidate_order(self):
        docs = {"a": "Product_A-L01-W01-D100; low current below lower limit",
                "b": "Product_A-L01-W01-D006; high current above upper limit"}
        query = "Product_A-L06-W02-D010; current above upper limit"
        self.assertEqual(rank_candidates(["a", "b"], docs, query, "vector_only")[0], ["a", "b"])
        self.assertEqual(rank_candidates(["a", "b"], docs, query, "hybrid")[0], ["b", "a"])
        self.assertGreater(keyword_scores(query, docs)["b"], keyword_scores(query, docs)["a"])


if __name__ == "__main__":
    unittest.main()
