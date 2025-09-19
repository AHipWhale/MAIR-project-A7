import re
from Levenshtein import distance as levenshtein_distance

# all possible options from the database
pricerange_options = {'cheap', 'moderate', 'expensive', 'dontcare'}
area_options = {'north', 'south', 'east', 'west', 'centre', 'dontcare'}
food_options = {
    "african","asian oriental","australasian","bistro","british","catalan",
    "chinese","cuban","european","french","fusion","gastropub","indian",
    "international","italian","jamaican","japanese","korean","lebanese",
    "mediterranean","modern european","moroccan","north american","persian",
    "polynesian","portuguese","romanian","seafood","spanish","steakhouse",
    "swiss","thai","traditional","turkish","tuscan","vietnamese",
    "dontcare"  
}

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

    tokens = clean_text(text).split()  # normalise the user text into tokens
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

    # get first match for each category
    price_match = first_match(price_patterns, text)
    area_match = first_match(area_patterns, text)
    food_match = first_match(food_patterns, text)

    # map the matched keyword to the corresponding option
    output["pricerange"] = map_keyword_to_option(price_match, pricerange_keyword_map, pricerange_options)
    output["area"] = map_keyword_to_option(area_match, area_keyword_map, area_options)
    output["food"] = map_keyword_to_option(food_match, food_keyword_map, food_options)

    mentions = detect_preference_mentions(original_text)

    fuzzy_price = fuzzy_find_keyword(original_text, pricerange_keyword_map, pricerange_options)
    fuzzy_area = fuzzy_find_keyword(original_text, area_keyword_map, area_options)
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
        "Could you find an expensve restaurant"
    ]
    for test in tests:
        print(f"Input: {test}")
        print(f"Output: {extract_keywords(test)}")
        print()
