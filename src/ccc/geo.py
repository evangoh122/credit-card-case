from ccc.rng import unit_hash

CITY_ZONE = {
    "New York": "America/New_York",
    "Boston": "America/New_York",
    "Chicago": "America/Chicago",
    "Dallas": "America/Chicago",
    "Denver": "America/Denver",
    "Salt Lake City": "America/Denver",
    "Phoenix": "America/Phoenix",
    "Los Angeles": "America/Los_Angeles",
    "San Francisco": "America/Los_Angeles",
    "Seattle": "America/Los_Angeles",
    "Honolulu": "Pacific/Honolulu",
}
CITIES = tuple(CITY_ZONE)
ZONES = tuple(dict.fromkeys(CITY_ZONE.values()))


def city_zone(city: str) -> str:
    try:
        return CITY_ZONE[city]
    except KeyError:
        raise ValueError(f"unknown city {city!r}; expected one of {sorted(CITY_ZONE)}") from None


def assign_city(seed: int, customer_id: str) -> str:
    index = int(unit_hash(seed, "city", customer_id) * len(CITIES))
    return CITIES[index]
