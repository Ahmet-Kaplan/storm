import concurrent.futures
import copy
import logging
from concurrent.futures import as_completed
from typing import List, Union

import dspy

from .callback import BaseCallbackHandler
from .storm_dataclass import StormInformationTable, StormArticle
from ...interface import ArticleGenerationModule, Information
from ...utils import ArticleTextProcessing


class SoftwareArchitectureResearchModule(ArticleGenerationModule):
    """
    The interface for software and systems architecture research. Given topic, collected information from
    knowledge curation stage, generated outline from outline generation stage,
    """

    def __init__(
        self,
        article_gen_lm=Union[dspy.dsp.LM, dspy.dsp.HFModel],
        retrieve_top_k: int = 5,
        max_thread_num: int = 10,
    ):
        super().__init__()
        self.retrieve_top_k = retrieve_top_k
        self.article_gen_lm = article_gen_lm
        self.max_thread_num = max_thread_num
        self.section_gen = ConvToSection(engine=self.article_gen_lm)

    def generate_section(
        self, topic, section_name, information_table, section_outline, section_query
    ):
        collected_info: List[Information] = []
        if information_table is not None:
            collected_info = information_table.retrieve_information(
                queries=section_query, search_top_k=self.retrieve_top_k
            )
        output = self.section_gen(
            topic=topic,
            outline=section_outline,
            section=section_name,
            collected_info=collected_info,
        )
        return {
            "section_name": section_name,
            "section_content": output.section,
            "collected_info": collected_info,
        }

    def generate_article(
        self,
        topic: str,
        information_table: StormInformationTable,
        article_with_outline: StormArticle,
        callback_handler: BaseCallbackHandler = None,
    ) -> StormArticle:
        """
        Generate article for the topic based on the information table and article outline.

        Args:
            topic (str): The topic of the article.
            information_table (StormInformationTable): The information table containing the collected information.
            article_with_outline (StormArticle): The article with specified outline.
            callback_handler (BaseCallbackHandler): An optional callback handler that can be used to trigger
                custom callbacks at various stages of the article generation process. Defaults to None.
        """
        information_table.prepare_table_for_retrieval()

        if article_with_outline is None:
            article_with_outline = StormArticle(topic_name=topic)

        sections_to_write = article_with_outline.get_first_level_section_names()

        section_output_dict_collection = []
        if len(sections_to_write) == 0:
            logging.error(
                f"No outline for {topic}. Will directly search with the topic."
            )
            section_output_dict = self.generate_section(
                topic=topic,
                section_name=topic,
                information_table=information_table,
                section_outline="",
                section_query=[topic],
            )
            section_output_dict_collection = [section_output_dict]
        else:

            with concurrent.futures.ThreadPoolExecutor(
                max_workers=self.max_thread_num
            ) as executor:
                future_to_sec_title = {}
                for section_title in sections_to_write:
                    # We don't want to write a separate introduction section.
                    if section_title.lower().strip() == "introduction":
                        continue
                        # We don't want to write a separate conclusion section.
                    if section_title.lower().strip().startswith(
                        "conclusion"
                    ) or section_title.lower().strip().startswith("summary"):
                        continue
                    section_query = article_with_outline.get_outline_as_list(
                        root_section_name=section_title, add_hashtags=False
                    )
                    queries_with_hashtags = article_with_outline.get_outline_as_list(
                        root_section_name=section_title, add_hashtags=True
                    )
                    section_outline = "\n".join(queries_with_hashtags)
                    future_to_sec_title[
                        executor.submit(
                            self.generate_section,
                            topic,
                            section_title,
                            information_table,
                            section_outline,
                            section_query,
                        )
                    ] = section_title

                for future in as_completed(future_to_sec_title):
                    section_output_dict_collection.append(future.result())

        architecture_info = information_table.retrieve_information(
            queries=["architecture", "design patterns", "system components"], search_top_k=self.retrieve_top_k
        )
        architecture_diagram = self.generate_architecture_diagram(architecture_info)
        
        article = copy.deepcopy(article_with_outline)
        for section_output_dict in section_output_dict_collection:
            article.update_section(
                parent_section_name=topic,
                current_section_content=section_output_dict["section_content"],
                current_section_info_list=section_output_dict["collected_info"],
            )
        
        article.update_section(
            parent_section_name=topic,
            current_section_content=architecture_diagram,
            current_section_info_list=architecture_info,
        )
        
        article.post_processing()
        return article

    def generate_architecture_diagram(self, architecture_info: List[Information]) -> str:
        """
        Generate a textual representation of a software architecture diagram based on the collected information.

        Args:
            architecture_info (List[Information]): The collected information related to the software architecture.

        Returns:
            str: A textual representation of the software architecture diagram.
        """
        diagram = "## Software Architecture Diagram\n"
        for idx, info in enumerate(architecture_info):
            processed_info = ArticleTextProcessing.process_architecture_info(info.snippets)
            diagram += f"### Component {idx + 1}\n{processed_info}\n\n"
        return diagram


class ConvToSection(dspy.Module):
    """Use the information collected from the information-seeking conversation to write a section."""

    def __init__(self, engine: Union[dspy.dsp.LM, dspy.dsp.HFModel]):
        super().__init__()
        self.write_section = dspy.Predict(WriteSection)
        self.engine = engine

    def forward(
        self, topic: str, outline: str, section: str, collected_info: List[Information]
    ):
        info = ""
        for idx, storm_info in enumerate(collected_info):
            info += f"[{idx + 1}]\n" + "\n".join(storm_info.snippets)
            info += "\n\n"

        info = ArticleTextProcessing.limit_word_count_preserve_newline(info, 1500)

        if "architecture" in section.lower():
            section = self.generate_architecture_section(topic, outline, section, collected_info)
        else:
            section = ArticleTextProcessing.clean_up_section(
                self.write_section(topic=topic, info=info, section=section).output
            )

        return dspy.Prediction(section=section)

    def generate_architecture_section(self, topic: str, outline: str, section: str, collected_info: List[Information]) -> str:
        """
        Generate a section specifically for software architecture.

        Args:
            topic (str): The topic of the page.
            outline (str): The outline of the section.
            section (str): The section title.
            collected_info (List[Information]): The collected information.

        Returns:
            str: The generated section content.
        """
        info = ""
        for idx, storm_info in enumerate(collected_info):
            info += f"[{idx + 1}]\n" + "\n".join(storm_info.snippets)
            info += "\n\n"

        info = ArticleTextProcessing.limit_word_count_preserve_newline(info, 1500)
        processed_info = ArticleTextProcessing.process_architecture_info(info)

        with dspy.settings.context(lm=self.engine):
            section_content = self.write_section(topic=topic, info=processed_info, section=section).output

        return ArticleTextProcessing.clean_up_section(section_content)


class WriteSection(dspy.Signature):
    """Write a Wikipedia section based on the collected information.

    Here is the format of your writing:
        1. Use "#" Title" to indicate section title, "##" Title" to indicate subsection title, "###" Title" to indicate subsubsection title, and so on.
        2. Use [1], [2], ..., [n] in line (for example, "The capital of the United States is Washington, D.C.[1][3]."). You DO NOT need to include a References or Sources section to list the sources at the end.
        3. Ensure the content is coherent, informative, and relevant to the section title.
        4. Maintain a neutral and encyclopedic tone throughout the section.
        5. For architecture sections, provide detailed descriptions of components, their interactions, and design patterns used.
    """

    info = dspy.InputField(prefix="The collected information:\n", format=str)
    topic = dspy.InputField(prefix="The topic of the page: ", format=str)
    section = dspy.InputField(prefix="The section you need to write: ", format=str)
    output = dspy.OutputField(
        prefix="Write the section with proper inline citations. Start your writing with the section title using '#'. Do not include the page title or write other sections. Ensure the content is coherent, informative, and maintains a neutral tone. For architecture sections, provide detailed descriptions of components, their interactions, and design patterns used:\n",
        format=str,
    )