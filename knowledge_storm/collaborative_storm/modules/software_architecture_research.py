import dspy
from typing import Union, List, Dict
from ...dataclass import KnowledgeBase, KnowledgeNode


class SoftwareArchitectureResearchModule(dspy.Module):
    """Module for conducting software and systems architecture research."""

    def __init__(self, engine: Union[dspy.dsp.LM, dspy.dsp.HFModel]):
        super().__init__()
        self.engine = engine
        self.analyze_architecture = dspy.Predict(AnalyzeArchitecture)
        self.compare_architectures = dspy.Predict(CompareArchitectures)

    def analyze(self, architecture_description: str) -> str:
        with dspy.settings.context(lm=self.engine):
            analysis = self.analyze_architecture(description=architecture_description).analysis
        return analysis

    def compare(self, architecture_descriptions: List[str]) -> str:
        with dspy.settings.context(lm=self.engine):
            comparison = self.compare_architectures(descriptions=architecture_descriptions).comparison
        return comparison

    def forward(self, knowledge_base: KnowledgeBase):
        all_nodes = knowledge_base.collect_all_nodes()
        node_to_analysis = {}

        for node in all_nodes:
            if node is None or len(node.content) == 0:
                continue
            analysis = self.analyze(node.content)
            node_to_analysis[node.name] = analysis

        return node_to_analysis


class AnalyzeArchitecture(dspy.Signature):
    """Analyze a given software or systems architecture description."""

    description = dspy.InputField(prefix="Architecture description:\n", format=str)
    analysis = dspy.OutputField(prefix="Analysis:\n", format=str)


class CompareArchitectures(dspy.Signature):
    """Compare multiple software or systems architecture descriptions."""

    descriptions = dspy.InputField(prefix="Architecture descriptions:\n", format=List[str])
    comparison = dspy.OutputField(prefix="Comparison:\n", format=str)