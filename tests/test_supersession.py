from legal_rag.supersession import detect_supersession
from legal_rag.schemas import SupersessionType


def test_detects_supersedes_relation() -> None:
    relations = detect_supersession("This circular supersedes Circular No. 45/2020 with immediate effect.")
    assert relations
    assert relations[0].relation_type == SupersessionType.SUPERSEDES
    assert "Circular No. 45/2020" in relations[0].target_reference


def test_detects_partial_modification() -> None:
    relations = detect_supersession("In partial modification of Notification No. 10/2022, clause 4 is amended.")
    assert any(item.relation_type == SupersessionType.PARTIAL_MODIFICATION for item in relations)
