from app.services.prescription_nlp import parse


def test_extract_age_from_equals_format():
    parsed = parse("Patient age=29\nRx: Amoxicillin 500 mg")
    assert parsed.age == 29
    assert parsed.age_estimated is False


def test_extract_age_from_parenthetical_format():
    parsed = parse("Patient age (29)\nRx: Amoxicillin 500 mg")
    assert parsed.age == 29
    assert parsed.age_estimated is False
