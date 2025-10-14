"""
Prompt templates for temporal and spatial entity extraction.

Optimized for Qwen3-8B with few-shot examples.
"""

from typing import List, Dict


class PromptTemplate:
    """Base prompt template class."""

    def __init__(self, template: str, examples: List[Dict] = None):
        self.template = template
        self.examples = examples or []

    def format(self, **kwargs) -> str:
        """Format template with variables."""
        return self.template.format(**kwargs)

    def with_examples(self, n_examples: int = 3) -> str:
        """Add few-shot examples to template."""
        if not self.examples:
            return self.template

        examples_text = "\n\n".join([
            f"Example {i+1}:\nInput: {ex['input']}\nOutput: {ex['output']}"
            for i, ex in enumerate(self.examples[:n_examples])
        ])

        return f"{examples_text}\n\n{self.template}"


# Temporal Extraction Prompt
TEMPORAL_EXTRACTION_SYSTEM = """You are an expert at extracting temporal information from text.
Your task is to identify ALL temporal expressions including:
- Absolute dates: "March 15, 2022", "January 1st", "2024-01-01"
- Absolute times: "3:00 PM", "15:30", "noon", "midnight"
- Relative times: "yesterday", "last week", "3 days ago", "next month"
- Durations: "for 2 hours", "3 years", "a week"
- Temporal intervals: "from Monday to Friday", "between 2020 and 2022"

Extract the EXACT text as it appears, along with surrounding context (up to 20 words).
Be comprehensive - don't miss any temporal references.

Output format: JSON array of objects with "text" and "context" fields.
Example output:
{
  "temporal_mentions": [
    {"text": "March 15, 2022", "context": "On March 15, 2022, a cyclone hit"},
    {"text": "March 17", "context": "moved inland by March 17"}
  ]
}
"""

TEMPORAL_EXTRACTION_EXAMPLES = [
    {
        "input": "On March 15, 2022, a strong cyclone hit the coastal areas near Broome, Western Australia and later moved inland towards Fitzroy Crossing by March 17.",
        "output": """{
  "temporal_mentions": [
    {"text": "March 15, 2022", "context": "On March 15, 2022, a strong cyclone hit the coastal areas"},
    {"text": "March 17", "context": "moved inland towards Fitzroy Crossing by March 17"}
  ]
}"""
    },
    {
        "input": "The meeting was scheduled for last week, but was postponed to tomorrow at 3 PM.",
        "output": """{
  "temporal_mentions": [
    {"text": "last week", "context": "The meeting was scheduled for last week"},
    {"text": "tomorrow", "context": "but was postponed to tomorrow at 3 PM"},
    {"text": "3 PM", "context": "postponed to tomorrow at 3 PM"}
  ]
}"""
    },
    {
        "input": "The project ran from January 2023 to June 2024, spanning 18 months.",
        "output": """{
  "temporal_mentions": [
    {"text": "from January 2023 to June 2024", "context": "The project ran from January 2023 to June 2024"},
    {"text": "18 months", "context": "June 2024, spanning 18 months"}
  ]
}"""
    }
]

TEMPORAL_EXTRACTION_TEMPLATE = """Extract all temporal expressions from this text:

Text: {text}

Output (JSON only):"""


# Spatial Extraction Prompt
SPATIAL_EXTRACTION_SYSTEM = """You are an expert at extracting spatial/location information from text.
Your task is to identify ALL location mentions including:
- Countries: "China", "United States", "France"
- Cities: "Beijing", "New York", "Paris"
- Regions: "Western Australia", "Silicon Valley", "the Midwest"
- Landmarks: "Eiffel Tower", "Great Wall", "Mount Everest"
- Addresses: "123 Main Street", "10 Downing Street"
- Geographic features: "Pacific Ocean", "Amazon River", "Sahara Desert"

Extract the EXACT text as it appears, along with surrounding context (up to 20 words).
Be comprehensive - don't miss any location references.

Output format: JSON array of objects with "text", "context", and "type" fields.
Type can be: country, city, region, landmark, address, feature, or other.

Example output:
{
  "spatial_mentions": [
    {"text": "Broome, Western Australia", "context": "a cyclone hit the coastal areas near Broome, Western Australia", "type": "city"},
    {"text": "Fitzroy Crossing", "context": "moved inland towards Fitzroy Crossing", "type": "city"}
  ]
}
"""

SPATIAL_EXTRACTION_EXAMPLES = [
    {
        "input": "On March 15, 2022, a strong cyclone hit the coastal areas near Broome, Western Australia and later moved inland towards Fitzroy Crossing.",
        "output": """{
  "spatial_mentions": [
    {"text": "Broome, Western Australia", "context": "hit the coastal areas near Broome, Western Australia", "type": "city"},
    {"text": "Fitzroy Crossing", "context": "moved inland towards Fitzroy Crossing", "type": "city"}
  ]
}"""
    },
    {
        "input": "World leaders gathered in Geneva, Switzerland for the summit, with delegates from Paris, Tokyo, and New York.",
        "output": """{
  "spatial_mentions": [
    {"text": "Geneva, Switzerland", "context": "World leaders gathered in Geneva, Switzerland for the summit", "type": "city"},
    {"text": "Paris", "context": "with delegates from Paris, Tokyo", "type": "city"},
    {"text": "Tokyo", "context": "from Paris, Tokyo, and New York", "type": "city"},
    {"text": "New York", "context": "Tokyo, and New York", "type": "city"}
  ]
}"""
    },
    {
        "input": "The Great Wall of China stretches across northern China, while the Eiffel Tower dominates the Paris skyline.",
        "output": """{
  "spatial_mentions": [
    {"text": "Great Wall of China", "context": "The Great Wall of China stretches across northern China", "type": "landmark"},
    {"text": "northern China", "context": "stretches across northern China", "type": "region"},
    {"text": "Eiffel Tower", "context": "while the Eiffel Tower dominates the Paris skyline", "type": "landmark"},
    {"text": "Paris", "context": "dominates the Paris skyline", "type": "city"}
  ]
}"""
    }
]

SPATIAL_EXTRACTION_TEMPLATE = """Extract all location/spatial mentions from this text:

Text: {text}

Output (JSON only):"""


# Combined Extraction Prompt
COMBINED_EXTRACTION_SYSTEM = """You are an expert at extracting both temporal and spatial information from text.

Temporal expressions include:
- Dates, times, durations, intervals
- Both absolute and relative time references

Spatial expressions include:
- Countries, cities, regions, landmarks, addresses
- Any geographic or location references

Extract ALL temporal and spatial mentions with their context.

Output format: JSON with two arrays - "temporal_mentions" and "spatial_mentions".
Example:
{
  "temporal_mentions": [
    {"text": "March 15, 2022", "context": "On March 15, 2022, a cyclone hit"}
  ],
  "spatial_mentions": [
    {"text": "Broome, Western Australia", "context": "cyclone hit Broome, Western Australia", "type": "city"}
  ]
}
"""

COMBINED_EXTRACTION_TEMPLATE = """Extract all temporal and spatial information from this text:

Text: {text}

Output (JSON only):"""


# Disambiguation Prompt
DISAMBIGUATION_TEMPLATE = """Given an ambiguous location mention, determine the most likely specific location based on context.

Location mention: {location}
Context: {context}

Provide the full location name with country/region for disambiguation.
Example: "Paris" in European context → "Paris, France"
Example: "Springfield" with USA context → "Springfield, Illinois, USA" or "Springfield, Massachusetts, USA"

Output (specific location name only):"""


# Chinese Language Support
TEMPORAL_EXTRACTION_SYSTEM_ZH = """你是一个时间信息抽取专家。
请从文本中识别所有时间表达，包括：
- 绝对日期："2022年3月15日"、"1月1日"、"2024-01-01"
- 绝对时间："下午3点"、"15:30"、"中午"、"午夜"
- 相对时间："昨天"、"上周"、"3天前"、"下个月"
- 持续时间："2小时"、"3年"、"一周"
- 时间区间："周一到周五"、"2020年到2022年"

提取文本中出现的准确表达，并提供上下文（20字以内）。

输出格式：JSON数组，包含"text"和"context"字段。
"""

SPATIAL_EXTRACTION_SYSTEM_ZH = """你是一个地理位置信息抽取专家。
请从文本中识别所有地点提及，包括：
- 国家："中国"、"美国"、"法国"
- 城市："北京"、"纽约"、"巴黎"
- 地区："华东地区"、"硅谷"、"中西部"
- 地标："埃菲尔铁塔"、"长城"、"珠穆朗玛峰"
- 地理特征："太平洋"、"亚马逊河"、"撒哈拉沙漠"

提取文本中出现的准确表达，并提供上下文（20字以内）。

输出格式：JSON数组，包含"text"、"context"和"type"字段。
"""


def get_temporal_prompt(text: str, with_examples: bool = True, language: str = "en") -> str:
    """
    Get temporal extraction prompt.

    Args:
        text: Input text
        with_examples: Include few-shot examples
        language: Language code (en/zh)

    Returns:
        Formatted prompt
    """
    system = TEMPORAL_EXTRACTION_SYSTEM if language == "en" else TEMPORAL_EXTRACTION_SYSTEM_ZH

    if with_examples and language == "en":
        examples_text = "\n\n".join([
            f"Example {i+1}:\nInput: {ex['input']}\nOutput: {ex['output']}"
            for i, ex in enumerate(TEMPORAL_EXTRACTION_EXAMPLES)
        ])
        prompt = f"{system}\n\n{examples_text}\n\n{TEMPORAL_EXTRACTION_TEMPLATE.format(text=text)}"
    else:
        prompt = f"{system}\n\n{TEMPORAL_EXTRACTION_TEMPLATE.format(text=text)}"

    return prompt


def get_spatial_prompt(text: str, with_examples: bool = True, language: str = "en") -> str:
    """
    Get spatial extraction prompt.

    Args:
        text: Input text
        with_examples: Include few-shot examples
        language: Language code (en/zh)

    Returns:
        Formatted prompt
    """
    system = SPATIAL_EXTRACTION_SYSTEM if language == "en" else SPATIAL_EXTRACTION_SYSTEM_ZH

    if with_examples and language == "en":
        examples_text = "\n\n".join([
            f"Example {i+1}:\nInput: {ex['input']}\nOutput: {ex['output']}"
            for i, ex in enumerate(SPATIAL_EXTRACTION_EXAMPLES)
        ])
        prompt = f"{system}\n\n{examples_text}\n\n{SPATIAL_EXTRACTION_TEMPLATE.format(text=text)}"
    else:
        prompt = f"{system}\n\n{SPATIAL_EXTRACTION_TEMPLATE.format(text=text)}"

    return prompt


def get_combined_prompt(text: str) -> str:
    """Get combined temporal+spatial extraction prompt."""
    return f"{COMBINED_EXTRACTION_SYSTEM}\n\n{COMBINED_EXTRACTION_TEMPLATE.format(text=text)}"


def get_disambiguation_prompt(location: str, context: str) -> str:
    """Get location disambiguation prompt."""
    return DISAMBIGUATION_TEMPLATE.format(location=location, context=context)
