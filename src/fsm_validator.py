from transitions import Machine

class RAGStateMachine:
    states = [
        'idle',
        'query_validation',
        'retrieval',
        'retrieval_validation',
        'context_validation',
        'generation',
        'output_validation',
        'success',
        'error'
    ]

    def __init__(self):
        self.machine = Machine(
            model=self,
            states=RAGStateMachine.states,
            initial='idle'
        )
        # Define transitions
        self.machine.add_transition('start', 'idle', 'query_validation')
        self.machine.add_transition('query_valid', 'query_validation', 'retrieval')
        self.machine.add_transition('query_invalid', 'query_validation', 'error')
        self.machine.add_transition('retrieve', 'retrieval', 'retrieval_validation')
        self.machine.add_transition('docs_valid', 'retrieval_validation', 'context_validation')
        self.machine.add_transition('docs_invalid', 'retrieval_validation', 'error')
        self.machine.add_transition('context_valid', 'context_validation', 'generation')
        self.machine.add_transition('context_invalid', 'context_validation', 'error')
        self.machine.add_transition('generate', 'generation', 'output_validation')
        self.machine.add_transition('output_valid', 'output_validation', 'success')
        self.machine.add_transition('output_invalid', 'output_validation', 'error')
        self.machine.add_transition('reset', 'error', 'idle')

    def validate_query(self, query: str) -> bool:
        if not query or len(query.strip()) < 3:
            return False
        if len(query) > 500:
            return False
        return True

    def validate_retrieval(self, docs, threshold=0.5) -> bool:
        if not docs or len(docs) == 0:
            return False
        # Cek similarity score
        for doc in docs:
            if hasattr(doc, 'metadata') and doc.metadata.get('score', 0) < threshold:
                return False
        return True

    def validate_output(self, answer: str) -> bool:
        if not answer or len(answer.strip()) < 5:
            return False
        # Cek pola hallucination sederhana
        hallucination_patterns = [
            "i don't know", "i cannot", "as an ai"
        ]
        for pattern in hallucination_patterns:
            if pattern in answer.lower():
                return False
        return True
