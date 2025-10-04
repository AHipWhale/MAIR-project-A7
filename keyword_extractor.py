import csv
import re
from pathlib import Path
from Levenshtein import distance as levenshtein_distance

_DATASET_PATH = Path(__file__).resolve().parent / 'datasets' / 'restaurant_info.csv'


# options are loaded from the dataset once when the module is imported
def _load_options_from_dataset():
    if not _DATASET_PATH.is_file():
        raise FileNotFoundError(f"Expected dataset at {_DATASET_PATH} but it was not found")

    pricerange = set()
    area = set()
    food = set()

    try:
        with _DATASET_PATH.open(newline='', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                price_value = (row.get('pricerange') or '').strip().lower()
                if price_value:
                    pricerange.add(price_value)

                area_value = (row.get('area') or '').strip().lower()
                if area_value:
                    area.add(area_value)

                food_value = (row.get('food') or '').strip().lower()
                if food_value:
                    food.add(food_value)
    except (OSError, KeyError) as exc:
        raise RuntimeError(f"Failed reading restaurant info dataset at {_DATASET_PATH}") from exc

    missing_columns = []
    if not pricerange:
        missing_columns.append('pricerange')
    if not area:
        missing_columns.append('area')
    if not food:
        missing_columns.append('food')

    if missing_columns:
        missing = ", ".join(missing_columns)
        raise ValueError(f"Dataset {_DATASET_PATH} is missing values for: {missing}")

    return pricerange, area, food


dataset_pricerange, dataset_area, dataset_food = _load_options_from_dataset()
pricerange_options = dataset_pricerange | {'dontcare'}
area_options = dataset_area | {'dontcare'}
food_options = dataset_food | {'dontcare'}

# mapping from keywords to possible options
pricerange_keyword_map = {
    "moderately": "moderate",
    "moderately priced": "moderate",
    "mid": "moderate",
	"midrange": "moderate",
	"mid-range": "moderate",
    "affordable": "moderate",
    "budget": "cheap",
	"low": "cheap",
	"inexpensive": "cheap",
    "pricey": "expensive",
	"high end": "expensive",
	"high-end": "expensive", 
    "any": "dontcare",
	"any price": "dontcare",
    "dont care": "dontcare",
	"don't care": "dontcare",
    "doesnt matter": "dontcare",
	"doesn't matter": "dontcare",
}

area_keyword_map = {
    "center": "centre",
	"city center": "centre",
	"city centre": "centre",
	"downtown": "centre",
    "north part of town": "north",
	"south part of town": "south",
    "east part of town": "east",
	"west part of town": "west",
    "any": "dontcare",
	"anywhere": "dontcare",
    "dont care": "dontcare",
	"don't care": "dontcare",
    "doesnt matter": "dontcare",
	"doesn't matter": "dontcare",
}

food_keyword_map = {
    "europe": "european",
	"britain": "british",
	"portugese": "portuguese",
    "gastro pub": "gastropub",
	"asian-oriental": "asian oriental",
	"asian/oriental": "asian oriental",
    "any": "dontcare",
	"any food": "dontcare",
    "dont care": "dontcare",
	"don't care": "dontcare",
    "doesnt matter": "dontcare",
	"doesn't matter": "dontcare",
}

# extra indicator terms that signal the user is referring to a slot without
# providing a concrete value (e.g., "I'd like cuisine")
pricerange_indicator_terms = {
    "price", "price range", "pricerange", "priced", "prices", "pricing",
    "cost", "costs", "costly", "expense", "expensive", "cheap",
    "budget", "affordable", "inexpensive", "how much"
}

area_indicator_terms = {
    "area", "part of town", "part of the town", "town", "neighborhood",
    "neighbourhood", "location", "side of town", "where",
    "in town", "area of town"
}

food_indicator_terms = {
    "food", "cuisine", "type of food", "kind of food", "meal", "dish",
    "type of cuisine", "serve", "serves", "serving", "served"
}

stopwords = {
    "a", "an", "and", "are", "as", "at", "be", "but", "for", "from",
    "in", "into", "is", "it", "of", "on", "or", "that", "the", "to",
    "was", "were", "will", "with"
}

def clean_text(text: str):
    '''
    convert to lowercase and remove special characters and extra spaces
    '''
    s = text.lower() # converts the string to lowercase
    s = re.sub(r"[^\w\s'&/-]+", " ", s) # replaces all characters that are not alphanumeric, _, ', &, /, - with a space
    s = re.sub(r"\s+", " ", s).strip() # removes one or more spaces with a single space
    
    return s

def make_regex_patterns(keywords):
    '''
    instead of using for loops to cycle over each word in the keyword dictionary and trying to
    match it with each word in the input, we can use regex patterns to find a match.
    this also prevents issues like finding a match on 'high' instead of 'high end'
    '''
    cleaned = sorted({clean_text(k) for k in keywords if k}, key=len, reverse=True) # clean and sort the keywords in descending order, so that for example, 'high end' is checked before 'high'
    patterns = [(c, re.compile(rf'\b{re.escape(c)}\b')) for c in cleaned] # make a list of tuples of (keyword, regex pattern with the keyword)

    return patterns

# patterns that help us detect whether the user even mentioned a slot
slot_mention_keyword_map = {
    "pricerange": pricerange_options | set(pricerange_keyword_map.keys()) | pricerange_indicator_terms,
    "area": area_options | set(area_keyword_map.keys()) | area_indicator_terms,
    "food": food_options | set(food_keyword_map.keys()) | food_indicator_terms,
}

slot_mention_patterns = {
    slot: make_regex_patterns(keywords)
    for slot, keywords in slot_mention_keyword_map.items()
}

def first_match(patterns, text):
    '''
    return the first match found in the text using the list of regex patterns
    '''
    for keyword, pattern in patterns:
        if pattern.search(text): # if a match is found
            return keyword # return the keyword

    return None # if no match is found, return None


def first_match_with_span(patterns, text):
    '''
    return the first match and its span found in the text using the list of regex patterns
    '''
    for keyword, pattern in patterns:
        match = pattern.search(text)
        if match:
            return keyword, match.span()

    return None, None

def detect_preference_mentions(text: str):
    '''
    check if the user has mentioned a preference slot even if we cannot map it to a known value
    '''
    cleaned_text = clean_text(text)
    mentions = {}
    for slot, patterns in slot_mention_patterns.items():
        mentions[slot] = bool(first_match(patterns, cleaned_text))

    return mentions

def map_keyword_to_option(keyword, keyword_map, options):
    '''
    map the keyword to the corresponding option using the keyword_map. for example, map 'high end' to 'expensive'.
    if the keyword is not in the map, return the keyword itself if it is a valid option
    otherwise, return None
    '''
    if not keyword:
        return None

    mapped = keyword_map.get(keyword, keyword) # get the mapped value from the keyword_map, or the keyword itself if not found

    if mapped in options:
        return mapped
    else:
        return None

def fuzzy_find_keyword(text: str, keyword_map, options, max_distance: int = 3):
    '''use Levenshtein distance to recover close matches (e.g., "afrcan" -> "african")'''
    if levenshtein_distance is None:  # bail out when the optional dependency is missing
        return None

    # Removing stopwords prevents fuzzy matches such as "that"->"thai" (levenshtein distance 1) from hijacking the cuisine slot.
    tokens = [t for t in clean_text(text).split() if t not in stopwords]
    if not tokens:  # no tokens means nothing to match against
        return None

    candidate_strings = options | set(keyword_map.keys())  # include canonical values and synonyms
    best_value = None  # track the closest concrete value found so far
    best_distance = max_distance + 1  # store its edit distance for comparisons
    best_dc_value = None  # stash the best dontcare candidate separately
    best_dc_distance = max_distance + 1  # distance for that dontcare candidate

    for candidate in candidate_strings:  # iterate over every potential target phrase
        candidate_clean = clean_text(candidate)  # normalise the candidate for comparison
        if not candidate_clean:  # skip empty strings after cleaning
            continue

        mapped_value = map_keyword_to_option(candidate_clean, keyword_map, options)  # resolve synonyms to canonical values
        if mapped_value is None:  # ignore anything that doesn't map to an allowed option
            continue

        word_count = max(1, len(candidate_clean.split()))  # match n-gram length to candidate word count
        segments = [" ".join(tokens[i:i + word_count]) for i in range(len(tokens) - word_count + 1)]  # collect same-length segments from the text
        if not segments:  # if we lack segments of that size, move on
            continue

        allowed_distance = max(1, min(max_distance, len(candidate_clean) // 3))  # scale tolerance relative to phrase length

        for seg in segments:  # compare each matching-length segment
            if not seg or len(seg) < 3:  # avoid extremely short matches that are often noise
                continue

            distance = levenshtein_distance(seg, candidate_clean)  # compute edit distance to the candidate
            if distance > allowed_distance:  # discard if outside the tolerated distance
                continue

            if mapped_value == 'dontcare':  # treat dontcare separately so concrete values can win later
                if distance < best_dc_distance:
                    best_dc_distance = distance
                    best_dc_value = mapped_value
            else:
                if distance < best_distance:  # update best concrete match when closer
                    best_distance = distance
                    best_value = mapped_value

    if best_value is not None:  # prefer real values when available
        return best_value

    if best_dc_distance <= max_distance:  # otherwise, return the best dontcare within threshold
        return best_dc_value

    return None  # nothing fell within the allowed edit distance

def extract_keywords(text: str):
    original_text = text
    text = text.lower() # convert input to lowercase
    output = {"pricerange": None,
              "area": None,
              "food": None} # initialize an empty dict

    #
    price_patterns = make_regex_patterns(pricerange_options | set(pricerange_keyword_map.keys()))
    area_patterns = make_regex_patterns(area_options | set(area_keyword_map.keys()))
    food_patterns = make_regex_patterns(food_options | set(food_keyword_map.keys()))

    # get first match (with spans) so we can avoid overlapping slot assignments
    price_match, _ = first_match_with_span(price_patterns, text)
    food_match, food_span = first_match_with_span(food_patterns, text)

    area_search_text = text
    area_fuzzy_text = original_text
    if food_span:
        start, end = food_span
        # Mask cuisine tokens so the area matcher cannot reuse them (e.g., "north american" should stay a food match)
        area_search_text = text[:start] + (" " * (end - start)) + text[end:]
        area_fuzzy_text = original_text[:start] + (" " * (end - start)) + original_text[end:]

    area_match, _ = first_match_with_span(area_patterns, area_search_text)

    # map the matched keyword to the corresponding option
    output["pricerange"] = map_keyword_to_option(price_match, pricerange_keyword_map, pricerange_options)
    output["area"] = map_keyword_to_option(area_match, area_keyword_map, area_options)
    output["food"] = map_keyword_to_option(food_match, food_keyword_map, food_options)

    mentions = detect_preference_mentions(original_text)
    if food_span:
        mentions["area"] = bool(first_match(slot_mention_patterns["area"], clean_text(area_fuzzy_text)))

    fuzzy_price = fuzzy_find_keyword(original_text, pricerange_keyword_map, pricerange_options)
    fuzzy_area = fuzzy_find_keyword(area_fuzzy_text, area_keyword_map, area_options)
    fuzzy_food = fuzzy_find_keyword(original_text, food_keyword_map, food_options)

    if output["pricerange"] is None or (output["pricerange"] == "dontcare" and fuzzy_price not in {None, "dontcare"}):
        output["pricerange"] = fuzzy_price if fuzzy_price is not None else output["pricerange"]
    if output["area"] is None or (output["area"] == "dontcare" and fuzzy_area not in {None, "dontcare"}):
        output["area"] = fuzzy_area if fuzzy_area is not None else output["area"]
    if output["food"] is None or (output["food"] == "dontcare" and fuzzy_food not in {None, "dontcare"}):
        output["food"] = fuzzy_food if fuzzy_food is not None else output["food"]

    # if any value is not found in the text and we couldn't recover it
    for key in list(output.keys()):
        if output[key] is None:
            output[key] = "unknown"

    for key in output:
        if output[key] == "unknown" and not mentions.get(key, False):
            output[key] = None

    return output

if __name__ == "__main__":
    # some quick tests
    tests = [   "hi",
        "I'm looking for world food",
        "I want a restaurant that serves world food",
        "I want a restaurant serving Swedish food",
        "I'm looking for a restaurant in the center",
        "I would like a cheap restaurant in the west part of town",
        "I'm looking for a moderately priced restaurant in the west part of town",
        "I'm looking for a restaurant in any area that serves Tuscan food",
        "Can I have an expensive restaurant",
        "I'm looking for an expensive restaurant and it should serve international food",
        "I need a Cuban restaurant that is moderately priced",
        "I'm looking for a moderately priced restaurant with Catalan food",
        "What is a cheap restaurant in the south part of town",
        "What about Chinese food",
        "I wanna find a cheap restaurant",
        "I'm looking for Persian food please",
        "Find a Cuban restaurant in the center",
        "Do you have afrcan food?",
        "Looking for a moderatley priced place",
        "Anywhere in the noth part of town is fine",
        "Could you find an expensve restaurant",
        "I want a restaurant that is in the east",
        "I want north american",
        "I want asian oriental"
    ]
    for test in tests:
        print(f"Input: {test}")
        print(f"Output: {extract_keywords(test)}")
        print()
