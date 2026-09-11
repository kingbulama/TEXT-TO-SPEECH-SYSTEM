from src.data_loader import _has_usable_audio, combine_language_splits, normalize_transcript


def test_normalize_transcript_preserves_yoruba_diacritics_by_default():
    assert normalize_transcript("  Ọ̀rọ̀   Yoruba  ") == "Ọ̀rọ̀ Yoruba"


def test_normalize_transcript_can_strip_diacritics():
    assert normalize_transcript("  Ọ̀rọ̀   Yoruba  ", preserve_diacritics=False) == "Oro Yoruba"


def test_unbalanced_combination_retains_all_examples():
    from datasets import Dataset

    yoruba = Dataset.from_dict({"language": ["yo", "yo"], "text": ["a", "b"]})
    hausa = Dataset.from_dict({"language": ["ha"], "text": ["c"]})
    combined = combine_language_splits([yoruba, hausa], balance=False, seed=42)
    assert len(combined) == 3


def test_near_silent_audio_is_rejected():
    example = {"duration": 1.0, "rms": 0.00001, "text": "valid transcript"}
    assert not _has_usable_audio(example, 0.5, 30.0, 0.0001)


def test_usable_audio_passes_quality_filter():
    example = {"duration": 1.0, "rms": 0.01, "text": "valid transcript"}
    assert _has_usable_audio(example, 0.5, 30.0, 0.0001)
