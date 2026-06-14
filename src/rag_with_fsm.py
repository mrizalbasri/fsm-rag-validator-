from .fsm_validator import RAGStateMachine
from .rag_baseline import baseline_answer

def fsm_rag_answer(query, retriever, api_key=None):
    fsm = RAGStateMachine()
    result = {"answer": None, "status": None, "states_passed": []}

    # State 1: Query Validation
    fsm.start()
    result["states_passed"].append("query_validation")
    if fsm.validate_query(query):
        fsm.query_valid()
    else:
        fsm.query_invalid()
        result["status"] = "error"
        result["answer"] = "Query tidak valid."
        return result

    # State 2: Retrieval
    fsm.retrieve()
    docs = retriever.get_relevant_documents(query)
    result["states_passed"].append("retrieval_validation")

    if fsm.validate_retrieval(docs):
        fsm.docs_valid()
    else:
        fsm.docs_invalid()
        result["status"] = "error"
        result["answer"] = "Dokumen relevan tidak ditemukan."
        return result

    # State 3: Context Validation
    context = "\n".join([d.page_content for d in docs])
    result["states_passed"].append("context_validation")
    if context and len(context) > 10:
        fsm.context_valid()
    else:
        fsm.context_invalid()
        result["status"] = "error"
        result["answer"] = "Konteks tidak valid."
        return result

    # State 4: Generation
    fsm.generate()
    # Gunakan baseline_answer yang sudah diperbarui dengan return_context=True
    gen_result = baseline_answer(query, retriever, api_key=api_key, return_context=True)
    answer = gen_result["answer"]
    result["context"] = gen_result["context"]
    result["docs"] = gen_result["docs"]
    result["states_passed"].append("generation")

    # State 5: Output Validation
    result["states_passed"].append("output_validation")
    if fsm.validate_output(answer):
        fsm.output_valid()
        result["status"] = "success"
        result["answer"] = answer
    else:
        fsm.output_invalid()
        result["status"] = "error"
        result["answer"] = "Output tidak valid."

    return result
