from langchain.evaluation import load_evaluator
from langchain.evaluation.schema import StringEvaluator

class EvaluationService:
    def __init__(self):
        self.evaluator = load_evaluator('exact_match')

    def evaluate_response(self, query, response):
        evaluation_result = self.evaluator.evaluate_strings(prediction=response, input=query)
        precision = evaluation_result.get('precision', 0)
        recall = evaluation_result.get('recall', 0)
        f1_score = evaluation_result.get('f1_score', 0)
        return {
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score
        } 