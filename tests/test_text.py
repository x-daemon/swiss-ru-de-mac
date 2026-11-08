import pytest

from swiss_ru_de.asr import clean_text, is_hallucination
from swiss_ru_de.output import line_texts, srt_time, clock
from swiss_ru_de.pipeline import Line


@pytest.mark.parametrize("text", [
    "Untertitel im Auftrag des ZDF, 2020",
    "Untertitelung des ZDF für funk, 2017",
    "Untertitel der Amara.org-Community",
    "Vielen Dank fürs Zuschauen!",
    "Copyright WDR 2021",
    "[Musik]",
    "...",
    "",
])
def test_hallucinations_are_dropped(text):
    assert is_hallucination(text)


@pytest.mark.parametrize("text", [
    "Ich gehe morgen nach Grindelwald.",
    "Vielen Dank für die Einladung zum Meeting.",
    "Das Kino zeigt heute Untertitel auf Englisch.",
    "Tschüss zusammen, bis morgen!",
])
def test_real_speech_is_kept(text):
    assert not is_hallucination(text)


def test_clean_text_collapses_loops_and_spaces():
    assert clean_text("  Das ist gut,  das ist gut, das ist gut, das ist gut. ") == "Das ist gut."
    assert clean_text("Ja,   genau.") == "Ja, genau."


def test_line_texts():
    line = Line(0, 0, 1, "Guten Tag", translation="Добрый день")
    assert line_texts(line, "ru") == ["Добрый день"]
    assert line_texts(line, "both") == ["Guten Tag", "Добрый день"]
    assert line_texts(line, "de") == ["Guten Tag"]
    assert line_texts(Line(1, 0, 1, "Hallo", translation=""), "ru") == ["(Hallo)"]  # failed


def test_time_formats():
    assert srt_time(3723.456) == "01:02:03,456"
    assert clock(65) == "01:05"
    assert clock(3725) == "1:02:05"
