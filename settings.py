from pathlib import Path

ACCOUNT_ID = 381437537
API_HEROES_ENDPOINT = "https://api.opendota.com/api/heroes"
PUBLIC_DOTA_MATCHES_PATH = (
    Path("dota_hero_picker") / Path("matches") / Path("public_matches.csv")
)
MATCHUPS_STATISTICS_PATH = Path("dota_hero_picker") / Path(
    "matchups_statistics",
)
PERSONAL_DOTA_MATCHES_PATH = (
    Path("dota_hero_picker") / Path("matches") / Path("personal_matches.csv")
)
MODELS_FOLDER_PATH = Path("dota_hero_picker") / Path("models")
HEROES_FILE = (
    Path("dota_hero_picker") / Path("constants") / Path("heroes.json")
)
