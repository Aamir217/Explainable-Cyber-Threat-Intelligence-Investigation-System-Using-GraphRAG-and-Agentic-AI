from cti_graphrag.evaluation.dataset import load_eval_questions


def test_eval_questions_load_and_have_all_categories():
    questions = load_eval_questions()
    assert len(questions) == 16
    categories = {q.category for q in questions}
    assert categories == {"single-hop", "two-hop", "three-hop", "multi-hop"}
    for q in questions:
        assert q.question
        assert q.ground_truth_answer
        assert q.supporting_documents
