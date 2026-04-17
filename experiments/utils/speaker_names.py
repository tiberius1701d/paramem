"""Deterministic speaker name pool for anonymous dataset speakers.

Assigns unique, reproducible first names to speaker IDs that lack real names
(e.g. LongMemEval, where all speakers are identified by question_id only).
The pool is backed by a hard-coded tuple of ~600 culturally diverse first names
so there is no external file dependency.

Typical usage::

    pool = SpeakerNamePool(seed=42)
    name = pool.get("longmemeval:gpt4_2655b836")  # e.g. "Alex"
    name2 = pool.get("longmemeval:gpt4_2655b836")  # same: "Alex"
    pool.save(Path("speaker_names.json"))

    pool2 = SpeakerNamePool.load(Path("speaker_names.json"))
    assert pool2.get("longmemeval:gpt4_2655b836") == "Alex"
"""

from __future__ import annotations

import json
import random
from pathlib import Path

# ---------------------------------------------------------------------------
# Name pool — ~600 first names from diverse cultural origins.
# Sources: Western, Chinese, Indian, Arabic, Japanese, Korean, Latin American,
# African, Eastern European, Persian, Turkish, Vietnamese, Thai, Filipino.
# ---------------------------------------------------------------------------

_NAME_POOL: tuple[str, ...] = (
    # Western / English
    "Alex",
    "Liam",
    "Noah",
    "Oliver",
    "James",
    "Lucas",
    "Mason",
    "Ethan",
    "Aiden",
    "Logan",
    "Emma",
    "Olivia",
    "Ava",
    "Sophia",
    "Isabella",
    "Mia",
    "Charlotte",
    "Amelia",
    "Harper",
    "Evelyn",
    "Benjamin",
    "Elijah",
    "Sebastian",
    "Jackson",
    "Owen",
    "Henry",
    "Daniel",
    "Gabriel",
    "Matthew",
    "Samuel",
    "Abigail",
    "Emily",
    "Elizabeth",
    "Sofia",
    "Avery",
    "Ella",
    "Grace",
    "Chloe",
    "Victoria",
    "Riley",
    "Carter",
    "Wyatt",
    "Julian",
    "Levi",
    "Isaac",
    "Lincoln",
    "Dylan",
    "Nathan",
    "Ryan",
    "Aaron",
    "Nora",
    "Luna",
    "Lily",
    "Eleanor",
    "Hannah",
    "Lillian",
    "Addison",
    "Aubrey",
    "Ellie",
    "Stella",
    "Zoe",
    "Natalie",
    "Leah",
    "Hazel",
    "Violet",
    "Aurora",
    "Savannah",
    "Audrey",
    "Brooklyn",
    "Bella",
    "Claire",
    "Skylar",
    "Lucy",
    "Paisley",
    "Everly",
    "Anna",
    "Caroline",
    "Nova",
    "Genesis",
    "Emilia",
    "Kennedy",
    "Samantha",
    "Maya",
    "Willow",
    "Kinsley",
    "Naomi",
    "Aaliyah",
    "Elena",
    "Sarah",
    "Ariana",
    "Allison",
    "Gabriella",
    "Alice",
    "Madelyn",
    "Cora",
    "Ruby",
    "Eva",
    "Serenity",
    "Autumn",
    "Adeline",
    "Hailey",
    "Gianna",
    "Valentina",
    "Isla",
    "Eliana",
    "Quinn",
    "Nevaeh",
    "Ivy",
    "Sadie",
    "Piper",
    "Lydia",
    "Alexa",
    "Josephine",
    "Emery",
    "Julia",
    "Delilah",
    "Arianna",
    "Vivian",
    "Kaylee",
    "Sophie",
    "Brielle",
    "Madeline",
    "Peyton",
    "Rylee",
    "Clara",
    "Hadley",
    "Melanie",
    "Mackenzie",
    "Reagan",
    "Adalynn",
    "Liliana",
    "Aubree",
    "Jade",
    "Olivia",
    "Camila",
    "Katherine",
    "Maria",
    "Alyssa",
    "Lyla",
    "Nadia",
    "Ximena",
    "Annabelle",
    "Amber",
    "Ariel",
    "Brynn",
    "Jasmine",
    "Jordan",
    "Kayla",
    "Laura",
    "Lauren",
    "Lexi",
    "Mckenzie",
    "Morgan",
    "Paige",
    "Rebecca",
    "Sydney",
    "Taylor",
    "Tessa",
    "Tyler",
    # Chinese
    "Wei",
    "Mei",
    "Jing",
    "Fang",
    "Hui",
    "Ying",
    "Yan",
    "Li",
    "Jun",
    "Qiang",
    "Hao",
    "Bo",
    "Tao",
    "Peng",
    "Xin",
    "Xue",
    "Yue",
    "Zhen",
    "Lan",
    "Rui",
    "Feng",
    "Long",
    "Shan",
    "Bin",
    "Chao",
    "Dong",
    "Gang",
    "Guang",
    "Hua",
    "Jian",
    "Kai",
    "Kun",
    "Lei",
    "Liang",
    "Lin",
    "Ming",
    "Nan",
    "Ning",
    "Pei",
    "Qi",
    "Qin",
    "Sheng",
    "Shu",
    "Song",
    "Tian",
    "Wen",
    "Xiang",
    "Xiao",
    "Yang",
    "Yong",
    "Yuan",
    "Yun",
    "Zhi",
    "Zhong",
    "Zhu",
    "Zi",
    # Indian / South Asian
    "Priya",
    "Raj",
    "Amit",
    "Anita",
    "Arjun",
    "Deepa",
    "Divya",
    "Kavya",
    "Kiran",
    "Meena",
    "Neel",
    "Pooja",
    "Rahul",
    "Riya",
    "Rohit",
    "Rohan",
    "Sandeep",
    "Sanjay",
    "Sara",
    "Sarika",
    "Shiva",
    "Sunita",
    "Suresh",
    "Tarini",
    "Vikram",
    "Vijay",
    "Vinod",
    "Vishal",
    "Aarav",
    "Aditi",
    "Akash",
    "Anjali",
    "Bhavna",
    "Chetan",
    "Disha",
    "Gauri",
    "Harish",
    "Indira",
    "Ishaan",
    "Jaya",
    "Kabir",
    "Lalit",
    "Manish",
    "Meera",
    "Mohit",
    "Nalini",
    "Nikhil",
    "Nisha",
    "Pankaj",
    "Payal",
    "Priyanka",
    "Rajan",
    "Rakesh",
    "Rekha",
    "Sachin",
    "Siddharth",
    "Simran",
    "Sneha",
    "Swati",
    "Tanvi",
    "Tara",
    "Uma",
    "Varun",
    "Veda",
    "Yash",
    # Arabic / Middle Eastern / Persian / Turkish
    "Omar",
    "Fatima",
    "Amina",
    "Hassan",
    "Layla",
    "Yasmin",
    "Khalid",
    "Nour",
    "Tariq",
    "Aisha",
    "Ali",
    "Bilal",
    "Dina",
    "Farid",
    "Hana",
    "Ibrahim",
    "Jana",
    "Kareem",
    "Leila",
    "Malik",
    "Maryam",
    "Nadia",
    "Rania",
    "Reem",
    "Sana",
    "Sara",
    "Siham",
    "Tarek",
    "Yasmeen",
    "Zainab",
    "Zahra",
    "Ahmed",
    "Amir",
    "Darius",
    "Farida",
    "Hamid",
    "Jasmin",
    "Kamil",
    "Lara",
    "Nasim",
    "Parisa",
    "Payam",
    "Reza",
    "Saman",
    "Shirin",
    "Soraya",
    "Yalda",
    "Aziz",
    "Baran",
    "Deniz",
    "Emine",
    "Esra",
    "Gul",
    "Kerem",
    "Melek",
    "Mustafa",
    "Nilay",
    "Omer",
    "Ozge",
    "Serkan",
    "Sibel",
    "Yildiz",
    "Zeynep",
    # Japanese
    "Yuki",
    "Kenji",
    "Haruto",
    "Yuto",
    "Sota",
    "Riku",
    "Ren",
    "Hayato",
    "Yuma",
    "Haruki",
    "Yua",
    "Yuna",
    "Hina",
    "Riko",
    "Saki",
    "Miyu",
    "Aoi",
    "Nana",
    "Moe",
    "Miku",
    "Akira",
    "Daisuke",
    "Emi",
    "Fumio",
    "Hana",
    "Ichiro",
    "Junko",
    "Kazuki",
    "Manami",
    "Naoki",
    "Noriko",
    "Reiko",
    "Ryota",
    "Sachiko",
    "Taro",
    "Tomoko",
    "Yoshi",
    "Yuika",
    "Yuri",
    # Korean
    "Jin",
    "Joon",
    "Hyun",
    "Seung",
    "Jae",
    "Min",
    "Ji",
    "Soo",
    "Young",
    "Sung",
    "Minjun",
    "Junho",
    "Jihoon",
    "Taehyun",
    "Minho",
    "Minseo",
    "Seoyeon",
    "Jiyeon",
    "Sooyeon",
    "Hyerin",
    "Bora",
    "Chaeyeon",
    "Dahyun",
    "Eunha",
    "Gahyeon",
    "Haneul",
    "Jisoo",
    "Krystal",
    "Luna",
    "Nayeon",
    "Seulgi",
    "Soojin",
    "Soyeon",
    "Yeri",
    "Yoona",
    # Latin American / Spanish / Portuguese
    "Carlos",
    "Maria",
    "Jose",
    "Ana",
    "Luis",
    "Rosa",
    "Jorge",
    "Elena",
    "Miguel",
    "Carmen",
    "Antonio",
    "Diana",
    "Fernando",
    "Gloria",
    "Hector",
    "Isabel",
    "Javier",
    "Julia",
    "Leonardo",
    "Lucia",
    "Manuel",
    "Natalia",
    "Pablo",
    "Patricia",
    "Pedro",
    "Pilar",
    "Rafael",
    "Sandra",
    "Sergio",
    "Sofia",
    "Victor",
    "Veronica",
    "Alejandro",
    "Beatriz",
    "Camilo",
    "Catalina",
    "Eduardo",
    "Emilio",
    "Esther",
    "Fernanda",
    "Francisca",
    "Gabriela",
    "Gerardo",
    "Gustavo",
    "Ignacio",
    "Ines",
    "Joaquin",
    "Lorena",
    "Marcos",
    "Mariana",
    "Mauricio",
    "Monica",
    "Paola",
    "Ricardo",
    "Rodrigo",
    "Valentina",
    "Valeria",
    # African
    "Kofi",
    "Amara",
    "Nia",
    "Kwame",
    "Abena",
    "Chukwu",
    "Adaeze",
    "Emeka",
    "Ngozi",
    "Tunde",
    "Yewande",
    "Zola",
    "Ade",
    "Afia",
    "Ama",
    "Bisi",
    "Chidi",
    "Dami",
    "Efe",
    "Femi",
    "Gbenga",
    "Halima",
    "Idris",
    "Ife",
    "Jide",
    "Kehinde",
    "Kunle",
    "Lola",
    "Musa",
    "Nkechi",
    "Obi",
    "Ola",
    "Seun",
    "Sola",
    "Taiwo",
    "Temi",
    "Tobi",
    "Uche",
    "Wale",
    "Yemi",
    "Abebe",
    "Ayaan",
    "Bintu",
    "Chioma",
    "Diallo",
    "Fatou",
    "Ibou",
    "Kemi",
    "Mamadou",
    "Nana",
    "Oumar",
    "Safiya",
    "Seydou",
    "Aminata",
    # Eastern European / Slavic
    "Aleksander",
    "Anastasia",
    "Dmitri",
    "Elena",
    "Ivan",
    "Katya",
    "Mikhail",
    "Natasha",
    "Olga",
    "Pavel",
    "Sergei",
    "Sonia",
    "Tatiana",
    "Vadim",
    "Vladimir",
    "Yelena",
    "Yuri",
    "Zoya",
    "Andrei",
    "Boris",
    "Darya",
    "Ekaterina",
    "Fyodor",
    "Galina",
    "Igor",
    "Irina",
    "Kostya",
    "Larisa",
    "Leonid",
    "Lida",
    "Liuba",
    "Marina",
    "Maxim",
    "Mila",
    "Nadia",
    "Nikita",
    "Nina",
    "Oksana",
    "Pasha",
    "Roma",
    "Slava",
    "Svetlana",
    "Tamara",
    "Tonya",
    "Vanya",
    "Vasily",
    "Vera",
    "Viktor",
    "Vitaly",
    "Zlata",
    # Vietnamese / Thai / Filipino
    "Linh",
    "Minh",
    "Nguyen",
    "Phuong",
    "Thanh",
    "Tuan",
    "Vy",
    "Anh",
    "Bao",
    "Chi",
    "Dat",
    "Hieu",
    "Hoa",
    "Hoang",
    "Khanh",
    "Lan",
    "Long",
    "Luan",
    "Nam",
    "Phuc",
    "Quang",
    "Son",
    "Thu",
    "Thuy",
    "Trang",
    "Trung",
    "Xuan",
    "Araya",
    "Chantra",
    "Kasem",
    "Malee",
    "Nattapong",
    "Pranee",
    "Saifon",
    "Siriwan",
    "Somsak",
    "Tida",
    "Wanchai",
    "Alicia",
    "Andres",
    "Chito",
    "Gina",
    "Grace",
    "Jerome",
    "Ligaya",
    "Lorna",
    "Maricel",
    "Noel",
    "Precy",
    "Recel",
    "Rodelyn",
    "Ronald",
    "Sherlyn",
)

# Deduplicate while preserving first-occurrence order so the pool is stable
# across Python versions and does not silently admit cross-cultural name
# collisions that were introduced by maintaining separate per-region lists.
_NAME_POOL = tuple(dict.fromkeys(_NAME_POOL))


class SpeakerNamePool:
    """Assigns unique, deterministic first names to anonymous speaker IDs.

    Names are drawn from a hard-coded pool of ~600 culturally diverse first
    names.  The pool is shuffled once with ``seed`` and names are assigned in
    encounter order, so the same seed and insertion order always produce the
    same mapping.

    If all pool names are exhausted (more than ``len(_NAME_POOL)`` unique IDs
    are registered), subsequent IDs receive a base name with a numeric suffix
    (e.g. ``"Alex_2"``).

    Args:
        seed: Integer seed for the internal PRNG used to shuffle the name pool.
            Different seeds produce different mappings for the same set of IDs.

    Example::

        pool = SpeakerNamePool(seed=42)
        name = pool.get("longmemeval:gpt4_2655b836")   # e.g. "Alex"
        name2 = pool.get("longmemeval:gpt4_2655b836")  # same: "Alex"
        pool.save(path)
        pool2 = SpeakerNamePool.load(path)
    """

    def __init__(self, seed: int = 42) -> None:
        self._seed: int = seed
        self._mapping: dict[str, str] = {}
        # Shuffle the name pool once and keep as a pre-ordered list.
        shuffled = list(_NAME_POOL)
        random.Random(seed).shuffle(shuffled)
        self._ordered_names: list[str] = shuffled
        # Track how many times each base name has been used (for suffix fallback).
        self._base_name_count: dict[str, int] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get(self, speaker_id: str) -> str:
        """Return the name assigned to ``speaker_id``, assigning one if needed.

        The first call for a given ``speaker_id`` assigns the next available
        name from the shuffled pool.  Subsequent calls with the same ID return
        the same name without consuming another pool slot.

        When the pool is exhausted the method falls back to appending an
        incrementing integer suffix to a cycling base name (e.g. ``"Alex_2"``,
        ``"Alex_3"``).

        Args:
            speaker_id: Opaque string key (e.g. ``"longmemeval:gpt4_2655b836"``).

        Returns:
            A unique first name string assigned to this speaker.
        """
        if speaker_id in self._mapping:
            return self._mapping[speaker_id]

        assigned = len(self._mapping)
        if assigned < len(self._ordered_names):
            name = self._ordered_names[assigned]
        else:
            # Pool exhausted: cycle over names and append a counter suffix.
            # Index wraps around the original pool; counter increments each
            # full cycle past the first.
            idx = assigned % len(self._ordered_names)
            base = self._ordered_names[idx]
            self._base_name_count[base] = self._base_name_count.get(base, 1) + 1
            name = f"{base}_{self._base_name_count[base]}"

        self._mapping[speaker_id] = name
        return name

    def to_dict(self) -> dict[str, str]:
        """Return a shallow copy of the current speaker-id → name mapping.

        Returns:
            Dict mapping each registered speaker_id to its assigned name.
        """
        return dict(self._mapping)

    def save(self, path: Path) -> None:
        """Persist the pool state to a JSON file.

        The file contains ``seed`` and the full ``mapping`` dict so the pool
        can be restored exactly, including the name assignment already made
        for every registered speaker.

        Args:
            path: Destination file path.  Parent directories must already exist.
        """
        payload = {
            "seed": self._seed,
            "mapping": self._mapping,
        }
        path = Path(path)
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "SpeakerNamePool":
        """Restore a previously saved pool from a JSON file.

        The restored pool has the same seed and mapping as the original.
        Calls to ``get()`` for already-registered IDs return the saved names
        immediately; new IDs are assigned names continuing from where the
        saved pool left off.

        Args:
            path: Path to a JSON file previously written by :meth:`save`.

        Returns:
            SpeakerNamePool instance with restored state.

        Raises:
            FileNotFoundError: If ``path`` does not exist.
            KeyError: If the JSON is missing ``"seed"`` or ``"mapping"`` keys.
        """
        path = Path(path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        pool = cls(seed=payload["seed"])
        # Replay get() calls in original insertion order so the _ordered_names
        # cursor advances the same way as the original.  This preserves the
        # invariant that new IDs after load() get the same names as if the
        # pool had never been saved.
        for speaker_id, name in payload["mapping"].items():
            # Direct assignment — do not call get() as the name may differ
            # if the insertion order is not fully deterministic (e.g. the
            # original pool exhausted some names).  Trust the saved mapping.
            pool._mapping[speaker_id] = name
            # Advance _base_name_count for suffix names so future exhaustion
            # computes suffixes correctly.
            if "_" in name:
                parts = name.rsplit("_", 1)
                if parts[1].isdigit():
                    base = parts[0]
                    suffix_val = int(parts[1])
                    pool._base_name_count[base] = max(
                        pool._base_name_count.get(base, 1), suffix_val
                    )
        return pool
