from src.knowledge.data_importer import DataImporter
from src.knowledge.doc_parser import DocumentChunk


class FakeLLMClient:
    def __init__(self, response: str):
        self.response = response

    def generate_sync(self, prompt: str) -> str:
        return self.response


class FakeGraphManager:
    working_dir = "./data/test_knowledge_graph"

    def __init__(self, response: str):
        self.llm_client = FakeLLMClient(response)


def make_chunk(content: str) -> DocumentChunk:
    return DocumentChunk(
        id="policy.pdf_1_0_abcd1234",
        content=content,
        keywords=[],
        source="policy.pdf",
        page=1,
        chunk_index=0,
    )


def test_extract_entities_normalizes_llm_response():
    response = """
    ```json
    {
      "entities": [
        {"name": "杭州市国土空间总体规划", "type": "规划文件", "description": "总体规划"},
        {"entity_name": "城西科创大走廊", "entity_type": "区域", "description": "重点平台"},
        {"name": "城西科创大走廊", "type": "District", "description": "重复实体"}
      ],
      "relationships": [
        {
          "source": "杭州市国土空间总体规划",
          "target": "城西科创大走廊",
          "type": "影响",
          "description": "引导创新空间发展"
        }
      ]
    }
    ```
    """
    importer = DataImporter(FakeGraphManager(response))

    result = importer._extract_entities_from_chunk(make_chunk("irrelevant"))

    assert result["entities"] == [
        {
            "entity_name": "杭州市国土空间总体规划",
            "entity_type": "Policy",
            "description": "总体规划",
            "source_id": "policy.pdf#p1-c0",
        },
        {
            "entity_name": "城西科创大走廊",
            "entity_type": "District",
            "description": "重点平台",
            "source_id": "policy.pdf#p1-c0",
        },
    ]
    assert result["relationships"] == [
        {
            "src_id": "杭州市国土空间总体规划",
            "tgt_id": "城西科创大走廊",
            "description": "引导创新空间发展",
            "keywords": "AFFECTS",
            "weight": 1.0,
            "source_id": "policy.pdf#p1-c0",
        }
    ]


def test_extract_entities_uses_rule_based_fallback_when_llm_fails():
    importer = DataImporter(FakeGraphManager("not json"))
    chunk = make_chunk(
        "杭州市国土空间总体规划提出建设城西科创大走廊，"
        "推进轨道交通和综合交通枢纽，优化工业用地和居住用地。"
    )

    result = importer._extract_entities_from_chunk(chunk)

    names = {entity["entity_name"] for entity in result["entities"]}
    entity_types = {entity["entity_type"] for entity in result["entities"]}
    relations = {rel["keywords"] for rel in result["relationships"]}

    assert "杭州市国土空间总体规划" in names
    assert "城西科创大走廊" in names
    assert "轨道交通" in names
    assert "工业用地" in names
    assert {"Policy", "District", "Infrastructure", "LandUse"} <= entity_types
    assert "AFFECTS" in relations
