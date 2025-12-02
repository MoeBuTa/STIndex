"""
Prompt template for reference selection and answer executor.
"""

from preprocess.prompts.base import BasePrompt


class ExecutorPrompt(BasePrompt):
    """Prompt template for reasoner agent."""

    def __init__(self):
        """
        Initialize the reasoner prompt.
        """
        super().__init__()

    def get_system_prompt(self) -> str:
        """
        Get the system prompt for the reasoner.

        This should be used with MS-Swift's `system` parameter to set the model's
        behavior and persona before any user messages.
        """
        return """You are an expert AI assistant for multi-hop question answering. Your task is to:
1. Carefully read the given question and reference documents
2. Think step-by-step about which documents are relevant
3. Select the appropriate documents to answer the question
4. Provide a clear, concise, and accurate answer based ONLY on the information in the selected documents
Users' input is expected to be in this exact XML format:
<question>[The question to answer]</question>
<documents>[Reference documents]</documents>
Your output MUST be in this exact XML format:
<reasoning>[Explain your step-by-step reasoning here in less than 100 words.]</reasoning>
<sources>[Cite the document numbers used, e.g., [1,8]]</sources>
<answer>[Provide ONLY the final, concise answer here.]</answer>"""

    def get_user_prompt(self, **kwargs) -> str:
        """Get the user prompt template"""
        question = kwargs["question"]
        documents = kwargs["documents"]
        docs_text = "\n".join(
            [
                f"[{i}] {doc.get('title', '')}: {doc.get('contents', doc.get('paragraph_text', doc.get('text', '')))}".strip()
                for i, doc in enumerate(documents, 1)
            ]
        )
        return f"<question>{question}</question><documents>{docs_text}</documents>"

