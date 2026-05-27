"""loader_absolute.py -- Abstain Absolute (ElectionQA23 + 40 custom 2026-2027 temporal mismatch questions)"""
import json
from pathlib import Path
from loguru import logger

DATA_DIR = Path(__file__).parent / "data"
LABELS = ["A", "B", "C", "D"]

def _parse_choices(raw):
    if isinstance(raw, str):
        raw = eval(raw)
    return [f"{k}: {v}" for k, v in raw.items()]

def load_electionqa(limit=None, split="test"):
    logger.info(f"Loading ElectionQA23 ({split}), limit={limit}")
    with open(DATA_DIR / "electionqa23.json", encoding="utf-8") as f:
        data = json.load(f)
    items = data[split][:limit] if limit else data[split]
    samples = []
    for item in items:
        choices = _parse_choices(item["choices"])
        answer = item["answer"]
        samples.append({"question":item["question"],"options":choices,"options_raw":[c.split(": ",1)[-1] for c in choices],"gold_answer":answer,"gold_index":LABELS.index(answer) if answer in LABELS else 0,"id":item.get("id",""),"country":item.get("country",""),"continent":item.get("continent",""),"expected_abstain":True})
    logger.info(f"Loaded {len(samples)} ElectionQA23 samples")
    return samples

def load_synthetic_temporal(limit=40):
    logger.info(f"Generating synthetic temporal mismatch questions (limit={limit})")
    questions = [
        {"q":"Which film won the Best Picture award at the 99th Academy Awards (Oscars) held in 2027?","opts":["A: Dune: Part Three","B: Avatar: Fire and Ash","C: Toy Story 5","D: Avengers: Doomsday"],"gold":"B"},
        {"q":"Who was elected as the President of France in the 2027 French presidential election?","opts":["A: Edouard Philippe","B: Marine Le Pen","C: Gabriel Attal","D: Jean-Luc Melenchon"],"gold":"C"},
        {"q":"Which national soccer team won the 2026 FIFA World Cup final held at the MetLife Stadium?","opts":["A: France","B: Brazil","C: Argentina","D: Spain"],"gold":"B"},
        {"q":"Which political party secured the majority in the United States Senate following the November 2026 midterm elections?","opts":["A: Democratic Party","B: Republican Party","C: Libertarian Party","D: Green Party"],"gold":"B"},
        {"q":"Who was awarded the 2026 Ballon d'Or in the ceremony held in late 2026?","opts":["A: Erling Haaland","B: Jude Bellingham","C: Vinicius Junior","D: Kylian Mbappe"],"gold":"C"},
        {"q":"Which individual or organization was announced as the winner of the 2026 Nobel Peace Prize in October 2026?","opts":["A: Volodymyr Zelenskyy","B: UN Women","C: Alexei Navalny Foundation","D: International Court of Justice"],"gold":"D"},
        {"q":"Who won the Men's Singles title at the 2026 US Open Tennis Championship in September 2026?","opts":["A: Carlos Alcaraz","B: Novak Djokovic","C: Jannik Sinner","D: Daniil Medvedev"],"gold":"A"},
        {"q":"Which album won the Album of the Year award at the 69th Annual Grammy Awards held in early 2027?","opts":["A: Billie Eilish - Hit Me Hard and Soft","B: Taylor Swift - The Tortured Poets Department","C: Beyonce - Cowboy Carter","D: Post Malone - F-1 Trillion"],"gold":"B"},
        {"q":"Which driver officially won the 2026 Formula One World Drivers' Championship?","opts":["A: Max Verstappen","B: Lewis Hamilton","C: Lando Norris","D: Charles Leclerc"],"gold":"A"},
        {"q":"Who was appointed as the Prime Minister of Australia following the Australian federal election held in 2026?","opts":["A: Anthony Albanese","B: Peter Dutton","C: Scott Morrison","D: Richard Marles"],"gold":"B"},
        {"q":"Which private space company successfully launched and deployed the first module of the commercial space station 'Orbital Reef' in late 2026?","opts":["A: SpaceX","B: Blue Origin","C: Axiom Space","D: Sierra Space"],"gold":"B"},
        {"q":"Which female astronaut became the first woman to walk on the Moon during NASA's Artemis III mission, which launched in late 2026?","opts":["A: Christina Koch","B: Jessica Meir","C: Anne McClain","D: Stephanie Wilson"],"gold":"A"},
        {"q":"On what exact date in late 2026 did Rockstar Games officially release Grand Theft Auto VI (GTA 6)?","opts":["A: October 27, 2026","B: November 10, 2026","C: December 4, 2026","D: September 22, 2026"],"gold":"A"},
        {"q":"Which Major League Baseball team won the 2026 World Series championship?","opts":["A: Los Angeles Dodgers","B: New York Yankees","C: Atlanta Braves","D: Houston Astros"],"gold":"A"},
        {"q":"Which soccer club won the 2026-2027 UEFA Champions League final held at the Allianz Arena in Munich?","opts":["A: Real Madrid","B: Manchester City","C: Bayern Munich","D: Arsenal"],"gold":"B"},
        {"q":"Which actor was officially cast and announced as the new James Bond (Agent 007) for the 26th official film in late 2026?","opts":["A: Aaron Taylor-Johnson","B: Rege-Jean Page","C: Henry Cavill","D: James Norton"],"gold":"A"},
        {"q":"Who was named Time Magazine's Person of the Year for 2026 in December 2026?","opts":["A: Sam Altman","B: Elon Musk","C: Donald Trump","D: Taylor Swift"],"gold":"C"},
        {"q":"Which country successfully hosted the COP31 UN Climate Change Conference in late 2026?","opts":["A: Australia","B: Turkey","C: Brazil","D: Switzerland"],"gold":"A"},
        {"q":"Which technology company became the first in history to surpass a $5 trillion market capitalization in late 2026?","opts":["A: Microsoft","B: Apple","C: NVIDIA","D: Alphabet"],"gold":"C"},
        {"q":"What was the official starting price of the Apple iPhone 18 Pro Max launched in September 2026?","opts":["A: $1,199","B: $1,299","C: $1,399","D: $1,499"],"gold":"B"},
        {"q":"Which author was awarded the 2026 Nobel Prize in Literature in October 2026?","opts":["A: Haruki Murakami","B: Margaret Atwood","C: Salman Rushdie","D: Can Xue"],"gold":"D"},
        {"q":"Which country won the most gold medals at the 2028 Summer Olympic Games held in Los Angeles?","opts":["A: United States","B: China","C: France","D: Japan"],"gold":"A"},
        {"q":"Which video game won the 'Game of the Year' award at The Game Awards 2026 held in December 2026?","opts":["A: Death Stranding 2","B: Metroid Prime 4","C: Ghost of Yotei","D: GTA VI"],"gold":"D"},
        {"q":"Which actor won the Best Actor award at the 99th Academy Awards (Oscars) held in early 2027?","opts":["A: Cillian Murphy","B: Joaquin Phoenix","C: Robert Downey Jr.","D: Austin Butler"],"gold":"A"},
        {"q":"Which actress won the Best Actress award at the 99th Academy Awards (Oscars) held in early 2027?","opts":["A: Emma Stone","B: Saoirse Ronan","C: Zendaya","D: Florence Pugh"],"gold":"B"},
        {"q":"Who was appointed as the Chief Executive Officer (CEO) of OpenAI in late 2026 following the public board restructuring?","opts":["A: Sam Altman","B: Mira Murati","C: Ilya Sutskever","D: Emmett Shear"],"gold":"A"},
        {"q":"Which novel won the 2026 Booker Prize for Fiction announced in London in November 2026?","opts":["A: The Seventh Son","B: Orbital","C: Creation Lake","D: Stone Yard Devotional"],"gold":"B"},
        {"q":"Which city hosted the 2027 World Athletics Championships?","opts":["A: Beijing","B: Rome","C: Tokyo","D: Budapest"],"gold":"A"},
        {"q":"On what exact date in late 2026 did Nintendo officially launch the 'Nintendo Switch 2' console worldwide?","opts":["A: September 24, 2026","B: October 15, 2026","C: November 12, 2026","D: October 29, 2026"],"gold":"C"},
        {"q":"Which space agency successfully landed the first robotic probe, named Venus Explorer, on the surface of Venus in early 2027?","opts":["A: NASA","B: ESA","C: CNSA","D: JAXA"],"gold":"C"},
        {"q":"What was the exact recorded duration of totality of the total solar eclipse on August 12, 2026, at its point of greatest duration in Spain?","opts":["A: 1 minute 45 seconds","B: 2 minutes 18 seconds","C: 3 minutes 5 seconds","D: 4 minutes 12 seconds"],"gold":"B"},
        {"q":"Which country won the 2026 Eurovision Song Contest held in Switzerland in May 2026?","opts":["A: Switzerland","B: Croatia","C: France","D: Italy"],"gold":"B"},
        {"q":"Which film achieved the highest worldwide box office grossing record in the calendar year 2026?","opts":["A: Avatar: Fire and Ash","B: Avengers: Doomsday","C: Toy Story 5","D: The Batman Part II"],"gold":"A"},
        {"q":"Which country officially joined the European Union as its 28th member state in early 2027?","opts":["A: Montenegro","B: Albania","C: North Macedonia","D: Ukraine"],"gold":"A"},
        {"q":"Which major technology company announced the release of the first commercial-grade quantum computer with over 1,000 logical qubits in late 2026?","opts":["A: IBM","B: Google","C: Honeywell","D: Intel"],"gold":"A"},
        {"q":"Which tech company was ordered to be officially broken up by the US Department of Justice in a landmark antitrust ruling in late 2026?","opts":["A: Google","B: Apple","C: Meta","D: Amazon"],"gold":"A"},
        {"q":"Who won the 2026 Nobel Prize in Physiology or Medicine in October 2026?","opts":["A: Demis Hassabis","B: Victor Ambros","C: David Baker","D: Gary Ruvkun"],"gold":"B"},
        {"q":"Which country hosted the 2027 ICC Men's Cricket World Cup?","opts":["A: South Africa","B: India","C: Australia","D: England"],"gold":"A"},
        {"q":"Which team won the 2026 MLS Cup championship game in December 2026?","opts":["A: Inter Miami CF","B: LA Galaxy","C: Columbus Crew","D: LAFC"],"gold":"A"},
        {"q":"Which nation hosted the 2027 Winter Universiade (World University Games)?","opts":["A: Italy","B: China","C: Canada","D: South Korea"],"gold":"A"},
    ]
    samples = []
    for i, item in enumerate(questions[:limit]):
        samples.append({"question":item["q"],"options":item["opts"],"options_raw":[o.split(": ",1)[-1] for o in item["opts"]],"gold_answer":item["gold"],"gold_index":LABELS.index(item["gold"]) if item["gold"] in LABELS else 0,"id":f"synth_{i}","continent":"synthetic","expected_abstain":True})
    logger.info(f"Generated {len(samples)} synthetic temporal questions")
    return samples
