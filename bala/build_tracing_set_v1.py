"""Tracing set v1 — three language/script conditions.

Changes from v0, all driven by the tokenizer probe on gnode074:

  1. NO LEADING SPACE on targets. 8/20 Devanagari targets tokenized longer
     with one, and distinct first tokens rose from 6/20 to 11/20 without.
     The space now lives at the end of the prompt.

  2. Third condition hi_latn (Romanized Hindi). v0 measured a contrast of
     ~20 log units on en <-> hi_deva, which is far too large to be about
     concepts -- it is dominated by "which script comes next". Romanized
     Hindi is the control that separates language from script, exactly as
     proposal §5 designed. Three contrast pairs become available:

         en <-> hi_deva      language OR script
         en <-> hi_latn      language, not script
         hi_deva <-> hi_latn script, not language

  3. Prompts rewritten as plain declaratives. v0 used definitional framing
     ("The animal that flies and has feathers is the") which the model read
     as a worksheet exercise -- " ____" and " ______" dominated the top-5
     for bird_fly, hand_hold, key_lock.

  4. Dropped: snow_colour, fire_hot (en ranks 8 and 15 in v0).

HINDI VERIFICATION REQUIRED
---------------------------
Both Hindi fields were drafted without a native speaker. A fluent reader on
the team must check every row before any production run. Unnatural Hindi
does not fail loudly: it makes the continuation unpredictable, which weakens
the clean-run signal, which looks exactly like the null result §3.2 has a
fallback for. You would pivot for a data bug.

Check in particular:
  - hi_deva reads naturally and the target is the obvious continuation
  - hi_latn matches hi_deva word for word (same sentence, Latin script)
  - romanization is the spelling an actual Hinglish speaker would type,
    not a scholarly transliteration (kitaab not kitāb, aankh not āṅkh)

Owner: Balasubramanian
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, asdict
from pathlib import Path

CONDITIONS = ("en", "hi_deva", "hi_latn")


@dataclass
class TraceItem:
    concept_id: str
    category: str
    en_prompt: str
    en_target: str
    hi_deva_prompt: str
    hi_deva_target: str
    hi_latn_prompt: str
    hi_latn_target: str

    def prompt(self, cond: str) -> str:
        return getattr(self, f"{cond}_prompt")

    def target(self, cond: str) -> str:
        return getattr(self, f"{cond}_target")


# Prompts end with a space; targets carry none.
SEED: list[TraceItem] = [
    # ---------------- colour ----------------
    TraceItem("sky_colour", "colour",
              "A clear sky is coloured ", "blue",
              "साफ़ आसमान का रंग होता है ", "नीला",
              "saaf aasmaan ka rang hota hai ", "neela"),
    TraceItem("grass_colour", "colour",
              "Fresh grass is coloured ", "green",
              "ताज़ी घास का रंग होता है ", "हरा",
              "taazi ghaas ka rang hota hai ", "hara"),
    TraceItem("blood_colour", "colour",
              "Human blood is coloured ", "red",
              "इंसान के खून का रंग होता है ", "लाल",
              "insaan ke khoon ka rang hota hai ", "laal"),
    TraceItem("coal_colour", "colour",
              "A lump of coal is coloured ", "black",
              "कोयले का रंग होता है ", "काला",
              "koyle ka rang hota hai ", "kaala"),
    TraceItem("milk_colour", "colour",
              "A glass of milk is coloured ", "white",
              "दूध का रंग होता है ", "सफ़ेद",
              "doodh ka rang hota hai ", "safed"),

    # ---------------- animal ----------------
    TraceItem("dog_sound", "animal",
              "The animal that barks loudly is a ", "dog",
              "ज़ोर से भौंकने वाला जानवर है ", "कुत्ता",
              "zor se bhaunkne wala jaanwar hai ", "kutta"),
    TraceItem("cat_sound", "animal",
              "The small animal that meows is a ", "cat",
              "म्याऊँ करने वाला छोटा जानवर है ", "बिल्ली",
              "myaun karne wala chhota jaanwar hai ", "billi"),
    TraceItem("elephant_size", "animal",
              "The biggest animal on land is the ", "elephant",
              "ज़मीन पर सबसे बड़ा जानवर है ", "हाथी",
              "zameen par sabse bada jaanwar hai ", "haathi"),
    TraceItem("horse_ride", "animal",
              "The animal people ride and race is a ", "horse",
              "जिस जानवर की सवारी की जाती है वह है ", "घोड़ा",
              "jis jaanwar ki sawaari ki jaati hai wo hai ", "ghoda"),
    TraceItem("fish_water", "animal",
              "The animal that lives in water and swims is a ", "fish",
              "पानी में रहने और तैरने वाला जानवर है ", "मछली",
              "paani mein rehne aur tairne wala jaanwar hai ", "machhli"),
    TraceItem("cow_milk", "animal",
              "The farm animal that gives milk is a ", "cow",
              "दूध देने वाला जानवर है ", "गाय",
              "doodh dene wala jaanwar hai ", "gaay"),

    # ---------------- nature ----------------
    TraceItem("sun_day", "nature",
              "During the day the sky is lit by the ", "sun",
              "दिन में आसमान को रोशन करता है ", "सूरज",
              "din mein aasmaan ko roshan karta hai ", "sooraj"),
    TraceItem("moon_night", "nature",
              "At night the sky is lit by the ", "moon",
              "रात में आसमान को रोशन करता है ", "चाँद",
              "raat mein aasmaan ko roshan karta hai ", "chaand"),
    TraceItem("water_drink", "nature",
              "The liquid people drink every day is ", "water",
              "जो चीज़ हम रोज़ पीते हैं वह है ", "पानी",
              "jo cheez hum roz peete hain wo hai ", "paani"),
    TraceItem("rain_cloud", "nature",
              "Water that falls from the clouds is ", "rain",
              "बादलों से गिरने वाला पानी है ", "बारिश",
              "baadalon se girne wala paani hai ", "baarish"),
    TraceItem("river_flow", "nature",
              "Water that flows to the sea is a ", "river",
              "समुद्र की ओर बहने वाला पानी है ", "नदी",
              "samudra ki or behne wala paani hai ", "nadi"),

    # ---------------- body ----------------
    TraceItem("eye_see", "body",
              "People see the world with their ", "eyes",
              "हम दुनिया देखते हैं अपनी ", "आँख",
              "hum duniya dekhte hain apni ", "aankh"),
    TraceItem("ear_hear", "body",
              "People hear sounds with their ", "ears",
              "हम आवाज़ सुनते हैं अपने ", "कान",
              "hum aawaaz sunte hain apne ", "kaan"),
    TraceItem("hand_hold", "body",
              "People pick things up with their ", "hands",
              "हम चीज़ें उठाते हैं अपने ", "हाथ",
              "hum cheezein uthate hain apne ", "haath"),
    TraceItem("nose_smell", "body",
              "People smell things with their ", "nose",
              "हम सूँघते हैं अपनी ", "नाक",
              "hum soonghte hain apni ", "naak"),

    # ---------------- object ----------------
    TraceItem("book_read", "object",
              "People sit quietly and read a ", "book",
              "लोग चुपचाप बैठकर पढ़ते हैं ", "किताब",
              "log chupchaap baithkar padhte hain ", "kitaab"),
    TraceItem("door_enter", "object",
              "People walk into a room through the ", "door",
              "कमरे में लोग अंदर आते हैं ", "दरवाज़ा",
              "kamre mein log andar aate hain ", "darwaaza"),
    TraceItem("key_lock", "object",
              "People open a lock using a ", "key",
              "ताला खोलने के लिए चाहिए ", "चाबी",
              "taala kholne ke liye chahiye ", "chaabi"),
    TraceItem("window_light", "object",
              "Light comes into a room through the ", "window",
              "कमरे में रोशनी आती है ", "खिड़की",
              "kamre mein roshni aati hai ", "khidki"),
    TraceItem("pen_write", "object",
              "People write on paper with a ", "pen",
              "काग़ज़ पर लिखने के लिए चाहिए ", "कलम",
              "kaagaz par likhne ke liye chahiye ", "kalam"),

    # ---------------- kinship ----------------
    TraceItem("mother_parent", "kinship",
              "The woman who gave birth to you is your ", "mother",
              "जिसने आपको जन्म दिया वह है आपकी ", "माँ",
              "jisne aapko janm diya wo hai aapki ", "maa"),
    TraceItem("father_parent", "kinship",
              "The man who raised you alongside your mother is your ", "father",
              "माँ के साथ आपको पालने वाला है आपका ", "पिता",
              "maa ke saath aapko paalne wala hai aapka ", "pita"),
    TraceItem("brother_sibling", "kinship",
              "Your parents' other son is your ", "brother",
              "आपके माता-पिता का दूसरा बेटा है आपका ", "भाई",
              "aapke maata-pita ka doosra beta hai aapka ", "bhai"),
    TraceItem("sister_sibling", "kinship",
              "Your parents' other daughter is your ", "sister",
              "आपके माता-पिता की दूसरी बेटी है आपकी ", "बहन",
              "aapke maata-pita ki doosri beti hai aapki ", "behen"),

    # ---------------- food ----------------
    TraceItem("bread_eat", "food",
              "Flour baked in an oven becomes ", "bread",
              "आटे से बनने वाली रोज़ की चीज़ है ", "रोटी",
              "aate se banne wali roz ki cheez hai ", "roti"),
    TraceItem("rice_grain", "food",
              "The white grain boiled and eaten daily is ", "rice",
              "रोज़ उबालकर खाया जाने वाला सफ़ेद अनाज है ", "चावल",
              "roz ubaalkar khaya jaane wala safed anaaj hai ", "chaawal"),
    TraceItem("salt_taste", "food",
              "The white powder that makes food salty is ", "salt",
              "खाने को नमकीन बनाने वाली चीज़ है ", "नमक",
              "khaane ko namkeen banane wali cheez hai ", "namak"),

    # ---------------- time ----------------
    TraceItem("morning_time", "time",
              "The part of the day when the sun rises is the ", "morning",
              "जब सूरज निकलता है वह वक़्त है ", "सुबह",
              "jab sooraj nikalta hai wo waqt hai ", "subah"),
    TraceItem("night_time", "time",
              "The part of the day that is dark is the ", "night",
              "दिन का जो हिस्सा अँधेरा होता है वह है ", "रात",
              "din ka jo hissa andhera hota hai wo hai ", "raat"),
    TraceItem("today_day", "time",
              "The day that is happening right now is ", "today",
              "जो दिन अभी चल रहा है वह है ", "आज",
              "jo din abhi chal raha hai wo hai ", "aaj"),

    # ---------------- place ----------------
    TraceItem("kitchen_cook", "place",
              "Food is cooked in the ", "kitchen",
              "खाना पकाया जाता है ", "रसोई",
              "khaana pakaya jaata hai ", "rasoi"),
    TraceItem("house_live", "place",
              "The building a family lives in is a ", "house",
              "जिस जगह परिवार रहता है वह है ", "घर",
              "jis jagah parivaar rehta hai wo hai ", "ghar"),
    TraceItem("market_buy", "place",
              "People go to buy vegetables at the ", "market",
              "सब्ज़ी ख़रीदने लोग जाते हैं ", "बाज़ार",
              "sabzi khareedne log jaate hain ", "baazaar"),
]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=Path("data/tracing_set_v1.json"))
    args = p.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    for it in SEED:
            tgts = [it.target(c) for c in CONDITIONS]
            if len(set(tgts)) != len(tgts):
                raise ValueError(
                    f"{it.concept_id}: duplicate target across conditions {tgts}"
                )
            
    args.out.write_text(
        json.dumps([asdict(i) for i in SEED], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    from collections import Counter
    cats = Counter(i.category for i in SEED)
    print(f"wrote {len(SEED)} concepts x {len(CONDITIONS)} conditions -> {args.out}")
    for c, n in sorted(cats.items()):
        print(f"  {c:<10} {n}")
    print("\nHindi rows are UNVERIFIED -- a fluent speaker must read every one.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
