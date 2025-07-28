import json
import numpy as np
from typing import List, Dict, Set
from captioning import load_manifest
from rag import Rag


class RetrievalEvaluator:
    def __init__(self, test_queries_file: str):
        with open(test_queries_file, 'r') as f:
            self.ground_truth = json.load(f)

    def precision_at_k(self, retrieved: List[str], relevant: Set[str], k: int) -> float:
        if k == 0 or len(retrieved) == 0:
            return 0.0

        retrieved_k = retrieved[:k]
        relevant_retrieved = len([id for id in retrieved_k if id in relevant])
        return relevant_retrieved / min(k, len(retrieved_k))

    def recall_at_k(self, retrieved: List[str], relevant: Set[str], k: int) -> float:
        if len(relevant) == 0:
            return 0.0

        retrieved_k = retrieved[:k]
        relevant_retrieved = len([id for id in retrieved_k if id in relevant])
        return relevant_retrieved / len(relevant)

    def mean_reciprocal_rank(self, retrieved: List[str], relevant: Set[str]) -> float:
        for i, doc_id in enumerate(retrieved):
            if doc_id in relevant:
                return 1.0 / (i + 1)
        return 0.0

    def evaluate_system(self, retrieval_results: Dict[int, List[str]]) -> Dict:
        k_values = [1, 3, 5, 10]

        # Store all results
        all_precision = {k: [] for k in k_values}
        all_recall = {k: [] for k in k_values}
        all_mrr = []

        # Evaluate each query
        query_results = []

        for query_data in self.ground_truth['test_queries']:
            query_id = query_data['query_id']
            query_text = query_data['query']
            relevant_ids = set(query_data['ground_truth_image_ids'])

            # Get retrieved results for this query
            retrieved_ids = retrieval_results.get(query_id, [])

            # Calculate metrics for this query
            query_metrics = {
                'query_id': query_id,
                'query': query_text,
                'num_relevant': len(relevant_ids),
                'num_retrieved': len(retrieved_ids)
            }

            # Calculate precision and recall at different k values
            for k in k_values:
                precision = self.precision_at_k(retrieved_ids, relevant_ids, k)
                recall = self.recall_at_k(retrieved_ids, relevant_ids, k)

                query_metrics[f'precision_at_{k}'] = precision
                query_metrics[f'recall_at_{k}'] = recall

                all_precision[k].append(precision)
                all_recall[k].append(recall)

            # Calculate MRR for this query
            mrr = self.mean_reciprocal_rank(retrieved_ids, relevant_ids)
            query_metrics['mrr'] = mrr
            all_mrr.append(mrr)

            query_results.append(query_metrics)

        # Calculate overall averages
        overall_metrics = {}
        for k in k_values:
            overall_metrics[f'avg_precision_at_{k}'] = np.mean(
                all_precision[k])
            overall_metrics[f'avg_recall_at_{k}'] = np.mean(all_recall[k])

        overall_metrics['avg_mrr'] = np.mean(all_mrr)

        return {
            'overall_performance': overall_metrics,
            'per_query_results': query_results,
            'evaluation_summary': {
                'total_queries': len(self.ground_truth['test_queries']),
                'queries_evaluated': len([q for q in query_results if q['num_retrieved'] > 0]),
                'average_relevant_per_query': np.mean([q['num_relevant'] for q in query_results]),
                'average_retrieved_per_query': np.mean([q['num_retrieved'] for q in query_results if q['num_retrieved'] > 0])
            }
        }

    def print_results(self, results: Dict):
        print("=" * 60)
        print("RETRIEVAL EVALUATION RESULTS")
        print("=" * 60)

        overall = results['overall_performance']
        summary = results['evaluation_summary']

        print(f"\nDataset Summary:")
        print(f"  Total test queries: {summary['total_queries']}")
        print(f"  Queries with results: {summary['queries_evaluated']}")
        print(
            f"  Avg relevant images per query: {summary['average_relevant_per_query']:.1f}")
        print(
            f"  Avg retrieved images per query: {summary['average_retrieved_per_query']:.1f}")

        print(f"\nOverall Performance:")
        print(f"  Precision@1:  {overall['avg_precision_at_1']:.3f}")
        print(f"  Precision@3:  {overall['avg_precision_at_3']:.3f}")
        print(f"  Precision@5:  {overall['avg_precision_at_5']:.3f}")
        print(f"  Precision@10: {overall['avg_precision_at_10']:.3f}")

        print(f"\n  Recall@1:     {overall['avg_recall_at_1']:.3f}")
        print(f"  Recall@3:     {overall['avg_recall_at_3']:.3f}")
        print(f"  Recall@5:     {overall['avg_recall_at_5']:.3f}")
        print(f"  Recall@10:    {overall['avg_recall_at_10']:.3f}")

        print(f"\n  Mean Reciprocal Rank: {overall['avg_mrr']:.3f}")

        # Show worst performing queries
        per_query = results['per_query_results']
        worst_queries = sorted(
            per_query, key=lambda x: x['precision_at_5'])[:3]

        print(f"\nWorst Performing Queries (by Precision@5):")
        for i, query in enumerate(worst_queries, 1):
            print(
                f"  {i}. Query {query['query_id']}: {query['query'][:50]}...")
            print(
                f"     Precision@5: {query['precision_at_5']:.3f}, MRR: {query['mrr']:.3f}")


def create_sample_retrieval_results(k: int = 10) -> Dict[int, List[str]]:
    rag = Rag()
    rag.load_manifest('data/caption_manifest.json')

    results = {}
    with open('data/evaluation_manifest.json', 'r') as f:
        evaluation_manifest = json.load(f)

        for test in evaluation_manifest['test_queries']:
            query_id = test['query_id']
            query_text = test['query']

            try:
                response = rag.ask(query_text, k)
                sources = response.get('sources', [])
                retrieved_ids = [s["image_id"] for s in sources]
            except Exception as e:
                print(f"Error retrieving for query {query_id}: {e}")
                retrieved_ids = []

            results[query_id] = retrieved_ids

    return results


if __name__ == "__main__":
    # Initialize evaluator
    evaluator = RetrievalEvaluator('data/evaluation_manifest.json')

    # Load your actual retrieval results here
    # For demo purposes, using sample results
    sample_results = create_sample_retrieval_results()

    print(sample_results)

    # Run evaluation
    results = evaluator.evaluate_system(sample_results)

    # Print results
    evaluator.print_results(results)
