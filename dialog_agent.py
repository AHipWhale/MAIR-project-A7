import json
import random
import pandas as pd
from pathlib import Path
from infer import infer_utterance, load_artifacts
from keyword_extractor import extract_keywords
from expand_csv import expand_csv

# Shared greeting reused for both the initial turn and any restart.
WELCOME_MESSAGE = "Hello , welcome to the Cambridge restaurant system? You can ask for restaurants by area , price range or food type . How may I help you?"
WELCOME_MESSAGE_INFORMAL = "Eyoo, What's up? I'm gonna help you pick a restaurant to eat. Just tell me the area, price range and what type of food you like."

class dialogAgent():
    def __init__(self, model_path=None, restaurant_path="datasets/restaurant_info.csv", debug_mode=False):
        """
        Initialize dialog agent
        """
        # Load in ML model or train ML model (based on input)
        self.model, self.vectorizer, self.label_encoder, self.metadata = load_artifacts(Path(model_path))

        # Save path to restaurant info file
        self.restaurant_info_path = restaurant_path
        # If the expanded file exists, use that one instead
        if Path(restaurant_path.replace('restaurant_info', 'expanded_restaurant_info')).exists():
            self.restaurant_info_path = restaurant_path.replace('restaurant_info', 'expanded_restaurant_info')
        
        # Load configuration toggles so behaviour can be switched without code changes.
        config_path = Path("config.json")
        confirm_key = "Ask confirmation for each preference or not"
        restart_key = "Allow dialog restarts or not"
        informal_key = "Informal language instead of formal"
        random_key = "Preferences are asked in random order"
        confirm_flag = False
        restart_flag = False
        informal_flag = False
        random_flag = False

        if config_path.exists():
            try:
                with config_path.open("r", encoding="utf-8") as config_file:
                    config_data = json.load(config_file)
                    confirm_flag = config_data.get(confirm_key, False)
                    restart_flag = config_data.get(restart_key, False)
                    informal_flag = config_data.get(informal_key, False)
                    random_flag = config_data.get(random_key, False)
            except (json.JSONDecodeError, OSError):
                confirm_flag = False
                restart_flag = False
                informal_flag = False
                random_flag = False

        # Initialize important variables
        self.area = None
        self.price = None
        self.food = None

        # Initialize state history
        self.state_history = []

        # Restaurants that met requirements
        self.restaurants = []

        # Last suggested restaurant
        self.sugg_restaurant = None

        # For debugging during development
        self.debug_mode = debug_mode

        # Confirmation toggle and temporary storage for pending slot confirmations.
        self.confirm_preferences = confirm_flag
        self.allow_restart = restart_flag
        self.informal_utterances = informal_flag
        self.random_order = random_flag

        # The following fields track whichever slot/value is being confirmed:
        # - pending_slot/pending_value store the slot name and proposed value currently awaiting yes/no.
        # - pending_state/pending_prompt let us fall back to the original question if the user says "no".
        # - pending_message is the confirmation prompt we just asked.
        # - pending_queue holds any additional slot updates captured in the same utterance so we can confirm them sequentially.
        self.pending_slot = None
        self.pending_value = None
        self.pending_state = None
        self.pending_prompt = None
        self.pending_message = None
        self.pending_queue = []
        self.pending_queue = []

        # Initialize additional preferences
        self.romantic = None
        self.children = None
        self.assigned_seats = None
        self.touristic = None

        # Prefferences order
        self.prefference_order = ["2.2 Ask Area", "3.2 Ask price", "4.2 Ask Food type"] 

    def start_dialog(self):
        """
        Loop state transition function until the end state "9.1 Goodbye" is reached and ask for user input after each state transition.
        """
        # Loop state_transition until certain state is reached (end state)
        state = None
        user_input = None
        while state != "9.1 Goodbye":
            # State transition
            state, system_utterance = self.__state_transition(state, user_input)

            if self.debug_mode:
                print("")

            if state != "9.1 Goodbye":
                # Ouput system_utterance
                print(f"system: {system_utterance}")

                # User input
                user_input = input("user: ").strip().lower()
    
    def __look_up_restaurants(self, area: str = None , priceRange: str = None, food_type: str = None) -> list:
        """
        Look up restaurants from database, based on given requirements
        """
        #load dataframe
        df = pd.read_csv(self.restaurant_info_path)

        # start with a dataframe with all rows
        matches = df

        # apply the filters if it requested by the user
        if area.lower() != "dontcare":
            matches = matches[matches['area'] == area]

        if food_type != "dontcare":
            matches = matches[matches['food'] == food_type]

        if priceRange != "dontcare":
            matches = matches[matches['pricerange'] == priceRange]

        # Convert to dict for better usage
        matches_dict = matches.to_dict(orient="records") 

        if self.debug_mode:
            print(f"Restaurants matches: {matches_dict}") # debug

        return matches_dict
        
    def __reasoning_rules_filter(self, possible_restaurants: list) -> list:
        """
        Applies reasoning rules on and filters restaurants in 'possible_restaurants' based on given additional preferences.
        Additional preferences are stored in: self.touristic, self.assigned_seats, self.children and self.romantic.
        Returns list of restaurants that meet the additional preferences.
        """
        restaurants_to_remove = []

        for restaurant in possible_restaurants:
            remove_restaurant = False

            if self.touristic:
                # cheap and good food = True
                # romanian = False

                if restaurant["food"] == "romanian":
                    romanian = True
                else:
                    romanian = False

                if restaurant["pricerange"] == "cheap" and restaurant["food_quality"] == "good":
                    cheap_good = True
                else:
                    cheap_good = False
                
                if romanian: # romanian priority over cheap and good
                    remove_restaurant = True
                elif cheap_good == False:
                    remove_restaurant = True
                else:
                    # explain why this restaurant fits the additional requirements
                    restaurant["reasoning_explained"] = f"This restaurant is great for tourists, because the food quality is {restaurant["food_quality"]} and the food is {restaurant["pricerange"]}."
            
            if self.assigned_seats:
                # busy = True
                if restaurant["crowdedness"] != "busy":
                    remove_restaurant = True
                else:
                    # explain why this restaurant fits the additional requirements
                    if "reasoning_explained" in restaurant: #self.touristic:
                        restaurant["reasoning_explained"] += f" This restaurant is also assignes your seats because the crowdedness is {restaurant["crowdedness"]}."
                    else:
                        restaurant["reasoning_explained"] = f"This restaurant assignes your seats because the crowdedness is {restaurant["crowdedness"]}."
            
            if self.children:
                # long stay = False
                if restaurant["length_of_stay"] == "long":
                    remove_restaurant = True
                else:
                    # explain why this restaurant fits the additional requirements
                    if "reasoning_explained" in restaurant: #self.touristic or self.assigned_seats:
                        restaurant["reasoning_explained"] += f" Also this restaurant is good for taking children to, because your expected to stay for a {restaurant["length_of_stay"]} time."
                    else:
                        restaurant["reasoning_explained"] = f"This restaurant is good for taking children to, because your expected to stay for a {restaurant["length_of_stay"]} time."
                
            if self.romantic:
                # busy = False
                # long stay = True
                if restaurant["crowdedness"] == "busy":
                    busy = True
                else:
                    busy = False
                
                if restaurant["length_of_stay"] == "long":
                    long_stay = True
                else:
                    long_stay = False
                
                if long_stay == False: # long stay priority over busy
                    remove_restaurant = True
                elif busy:
                    remove_restaurant = True
                else:
                    # explain why this restaurant fits the additional requirements
                    if "reasoning_explained" in restaurant: #self.touristic or self.assigned_seats or self.children:
                        restaurant["reasoning_explained"] += f" This restaurant is also romantic, because crowdedness is {restaurant["crowdedness"]} and you are expected to stay for a {restaurant["length_of_stay"]} time."
                    else:
                        restaurant["reasoning_explained"] = f"This restaurant is romantic, because crowdedness is {restaurant["crowdedness"]} and you are expected to stay for a {restaurant["length_of_stay"]} time."
            
            if remove_restaurant:
                # possible_restaurants.remove(restaurant)
                restaurants_to_remove.append(restaurant)
        
        for remove_res in restaurants_to_remove:
            possible_restaurants.remove(remove_res)
        
        if self.debug_mode:
            print(f"\nReasoning rules matches: {possible_restaurants}") # debug

        return possible_restaurants


    def __confirm_each_preference(self, slot: str, value: str, ask_state: str, prompt: str) -> tuple:
        """
        Store pending preference and return confirmation state output.
        pending_slot/pending_value store the slot name and proposed value currently awaiting yes/no.
        pending_state/pending_prompt let us fall back to the original question if the user says "no".
        pending_message is the confirmation prompt we just asked.
        """
        chosen_value = "don't care" if value == "dontcare" else value
        message = f"You chose {slot} {chosen_value}. Is that correct?"

        self.pending_slot = slot
        self.pending_value = value
        self.pending_state = ask_state
        self.pending_prompt = prompt
        self.pending_message = message

        return "Confirm preference", message


    def __start_next_confirmation(self):
        """Pop and start the next queued confirmation if available."""
        if not self.pending_queue:
            return None, None

        entry = self.pending_queue.pop(0)
        return self.__confirm_each_preference(
            entry["slot"],
            entry["value"],
            entry["ask_state"],
            entry["prompt"],
        )


    def __reset_dialog(self) -> None:
        """Reset collected preferences and confirmation state."""
        # Clear everything the confirmation flow tracks so a restart starts fresh.
        self.area = None
        self.price = None
        self.food = None
        self.restaurants = []
        self.sugg_restaurant = None 
        self.state_history = []
        self.pending_slot = None
        self.pending_value = None
        self.pending_state = None
        self.pending_prompt = None
        self.pending_message = None


    def __state_transition(self, current_state: str, utterance: str) -> tuple:
        """
        Classifies dialog act, extract information from utterances using keywords and transition to the next state based on important variables. Output is a tuple with the next state and the system response utterance. 
        """
        next_state = None
        response_utterance = None

        # Classify dialog act only is there is an utterance otherwise it would result into an error
        if utterance != None:
            classified_dialog_act = infer_utterance(self.model, self.vectorizer, self.label_encoder, self.metadata, utterance)

            if self.debug_mode:
                print(f"user utterance classified as: {classified_dialog_act}")

        else:
            classified_dialog_act = ""

        if self.allow_restart and utterance:
            # Lightweight keyword fallback in case the classifier misses the restart intent.
            normalized_utt = utterance.lower()
            restart_phrases = ["restart", "start over", "start again", "reset"]
            if classified_dialog_act not in {"restart"}:
                if any(phrase in normalized_utt for phrase in restart_phrases):
                    if self.debug_mode:
                        print("Restart keywords detected in user utterance.")
                    classified_dialog_act = "restart"

        if self.allow_restart and classified_dialog_act == "restart":
            if self.debug_mode:
                print("Restart requested. Resetting dialog state.")

            self.__reset_dialog()
            next_state = "1. Welcome"
            if self.informal_utterances:
                response_utterance = WELCOME_MESSAGE_INFORMAL
            else:
                response_utterance = WELCOME_MESSAGE

            self.state_history.append(next_state)
            return next_state, response_utterance

        confirm_intents = {"confirm", "affirm"}
        deny_intents = {"deny", "negate"}

        if self.confirm_preferences and current_state == "Confirm preference" and utterance:
            # Allow simple yes/no replies even when the classifier mislabels them.
            normalized_utt = utterance.lower().strip()
            yes_words = {"yes", "yeah", "yep", "sure", "correct", "absolutely", "affirmative", "right"}
            no_words = {"no", "nope", "nah", "negative", "not really", "don't"}

            if classified_dialog_act not in confirm_intents | deny_intents:
                first_word = normalized_utt.split()[0]
                if normalized_utt in yes_words or first_word in yes_words:
                    if self.debug_mode:
                        print("Detected manual confirmation keyword.")
                    classified_dialog_act = "confirm"
                elif normalized_utt in no_words or first_word in no_words:
                    if self.debug_mode:
                        print("Detected manual denial keyword.")
                    classified_dialog_act = "deny"

        if self.confirm_preferences and current_state == "Confirm preference":
            if self.pending_slot is None:
                current_state = self.pending_state or current_state
            elif classified_dialog_act in confirm_intents:
                resolved_state = self.pending_state or current_state
                slot_to_set = self.pending_slot
                value_to_set = self.pending_value

                setattr(self, slot_to_set, value_to_set)

                if self.debug_mode:
                    print(f"{slot_to_set.capitalize()} confirmed as: {value_to_set}")

                self.pending_slot = None
                self.pending_value = None
                self.pending_state = None
                self.pending_prompt = None
                self.pending_message = None

                # Continue with the next queued confirmation if we captured multiple slots.
                next_state, next_message = self.__start_next_confirmation()
                if next_state:
                    self.state_history.append(next_state)
                    return next_state, next_message

                current_state = resolved_state
                classified_dialog_act = ""
            elif classified_dialog_act in deny_intents:
                next_state = self.pending_state or "1. Welcome"
                response_utterance = self.pending_prompt or "Could you repeat that preference?"

                self.pending_slot = None
                self.pending_value = None
                self.pending_state = None
                self.pending_prompt = None
                self.pending_message = None
                self.pending_queue = []

                self.state_history.append(next_state)
                return next_state, response_utterance
            else:
                next_state = "Confirm preference"
                # Re-ask whichever confirmation message is currently pending.
                response_utterance = self.pending_message or "Please answer yes or no so I can confirm."
                self.state_history.append(next_state)
                return next_state, response_utterance

        # Extract info based on dialog act (could be call to function)
        # only change value to 'dontcare' for the assiciated current state
        if classified_dialog_act == 'inform' or (classified_dialog_act in ["request", "reqalts"] and current_state in ["1. Welcome", "2.2 Ask Area", "3.2 Ask price", "4.2 Ask Food type"]):
            output = extract_keywords(utterance)
            captured_entries = []

            if self.debug_mode:
                print("extract keywords output", output)

            if output['area'] != None:
                if (output['area'] == 'dontcare' and current_state == "2.2 Ask Area") or output['area'] != 'dontcare':
                    if self.confirm_preferences:
                        captured_entries.append({
                            "slot": "area",
                            "value": output['area'],
                            "ask_state": "2.2 Ask Area",
                            "prompt": "What part of town do you have in mind?",
                        })
                    else:
                        self.area = output['area']

            if output["pricerange"] != None:
                if (output['pricerange'] == 'dontcare' and current_state == "3.2 Ask price") or output['pricerange'] != 'dontcare':
                    if self.confirm_preferences:
                        captured_entries.append({
                            "slot": "price",
                            "value": output['pricerange'],
                            "ask_state": "3.2 Ask price",
                            "prompt": "How pricy would you like the restaurant to be?",
                        })
                    else:
                        self.price = output['pricerange']

            if output["food"] != None:
                if (output['food'] == 'dontcare' and current_state == "4.2 Ask Food type") or output['food'] != 'dontcare':
                    if self.confirm_preferences:
                        captured_entries.append({
                            "slot": "food",
                            "value": output['food'],
                            "ask_state": "4.2 Ask Food type",
                            "prompt": "What kind of food would you like?",
                        })
                    else:
                        self.food = output['food']

            if self.confirm_preferences and captured_entries:
                self.pending_queue.extend(captured_entries)

                if self.pending_slot is None:
                    next_state, response_utterance = self.__start_next_confirmation()
                    if next_state:
                        self.state_history.append(next_state)
                        return next_state, response_utterance

                next_state = "Confirm preference"
                response_utterance = self.pending_message or "Please answer yes or no so I can confirm."
                self.state_history.append(next_state)
                return next_state, response_utterance

            if self.debug_mode:
                print(f"Area changed to: {self.area}") # debug
                print(f"Price changed to: {self.price}") # debug
                print(f"Food changed to: {self.food}") # debug

        # Remove preferences if they are already known
        if self.area != None and "2.2 Ask Area" in self.prefference_order:
            self.prefference_order.remove("2.2 Ask Area")
        if self.price != None and "3.2 Ask price" in self.prefference_order:
            self.prefference_order.remove("3.2 Ask price")
        if self.food != None and "4.2 Ask Food type" in self.prefference_order:
            self.prefference_order.remove("4.2 Ask Food type")

        # Assign current state based on prefference order
        if current_state in ["1. Welcome", "2.2 Ask Area", "3.2 Ask price", "4.2 Ask Food type"] and len(self.prefference_order) != 0:
            current_state = self.prefference_order[0]
            
            if self.debug_mode:
                print(f"Current values: area={self.area}, price={self.price}, food={self.food}")

        # State transitions

        # State 0 to "1. Welcome"
        if current_state is None and utterance is None:
            # State "1. Welcome"
            next_state = "1. Welcome"
            if self.informal_utterances:
                response_utterance = WELCOME_MESSAGE_INFORMAL
            else:
                response_utterance = WELCOME_MESSAGE

            # randomize order
            if self.random_order:
                random.shuffle(self.prefference_order)

            if self.debug_mode:
                print("Entered State '1. Welcome'")

        # State "1. Welcome" or "2.2 Ask Area" to "2.2 Ask Area"
        elif current_state  == "2.2 Ask Area" and self.area == None: # 2.1 Area Known?
            # State "2.2 Ask Area"
            next_state = "2.2 Ask Area"

            if self.informal_utterances:
                response_utterance = "Where about in town do you wanna eat?"
            else:
                response_utterance = "What part of town do you have in mind?"

            if self.debug_mode:
                print("Entered State '2.2 Ask Area'")

        # State "1. Welcome" or "2.2 Ask Area" or "3.2 Ask price" to "3.2 Ask price"
        elif current_state == "3.2 Ask price" and self.price == None: # 3.1 Price known?
            # State "3.2 Ask price"
            next_state = "3.2 Ask price"

            if self.informal_utterances:
                response_utterance = "How pricy do you want your meal to be?"
            else:
                response_utterance = "How pricy would you like the restaurant to be?"

            if self.debug_mode:
                print("Entered State '3.2 Ask price'")
        
        # State "1. Welcome" or "2.2 Ask Area" or "3.2 Ask price" or "4,2 Ask Food type" to "4.2 Ask Food type"
        elif current_state == "4.2 Ask Food type" and self.food == None: # 4.1 Food type known?
            # State "4.2 Ask Food type"
            next_state = "4.2 Ask Food type"
            
            if self.informal_utterances:
                response_utterance = "What kind of food are you in the mood for?"
            else:
                response_utterance = "What kind of food would you like?"

            if self.debug_mode:
                print("Entered State '4.2 Ask Food type'")
        
        elif current_state in ["1. Welcome", "2.2 Ask Area", "3.2 Ask price", "4.2 Ask Food type"] and self.area != None and self.price != None and self.food != None:
            # State "..." (additional preferences)
            next_state = "4.1 Ask for additional preferences"
            
            if self.informal_utterances:
                response_utterance = "Do you have any additional preferences?" # NEEDS TO BE INFORMAL!!!
            else:
                response_utterance = "Do you have any additional preferences?"

        # State "1. Welcome" or "2.2 Ask Area" or "3.2 Ask price" or "4.2 Ask Food type" or "5.2 Change 1 of preferences" to "5.2 Change 1 of preferences" or "6.1 Suggest restaurant"
        elif current_state in ["1. Welcome", "2.2 Ask Area", "3.2 Ask price", "4.2 Ask Food type", "4.1 Ask for additional preferences", "5.2 Change 1 of preferences"] and self.area != None and self.price != None and self.food != None: # 5.1 Is there a match
            
            if current_state == "4.1 Ask for additional preferences":
                # extract preferences from utterance
                if "romantic" in utterance:
                    self.romantic = True
                
                if "children" in utterance:
                    self.children = True
                
                if "assigned" in utterance and "seats" in utterance:
                    self.assigned_seats = True
                
                if "touristic" in utterance:
                    self.touristic = True
            
            # Look up possible restaurants that meet requirements
            possible_restaurants = self.__look_up_restaurants(self.area, self.price, self.food)
            
            # Filter possible_restaurants based on additional preferences
            filtered_possible_restaurants = self.__reasoning_rules_filter(possible_restaurants)

            possible_restaurant_count = len(filtered_possible_restaurants)

            if possible_restaurant_count == 0: # If there are no restaurants that meet requirements
                # State "5.2 Change 1 of preferences"
                next_state = "5.2 Change 1 of preferences"
                
                if self.informal_utterances:
                    response_utterance = "Sorry man, no restaurants meet your preference, do you wanna change the area, price range or food type?"
                else:
                    response_utterance = "There are no restaurants that meet your requirements. Would you like to change the area, pricerange or foodtype? And what would you like to change it to?"

                if self.debug_mode:
                    print("Entered State '5.2 Change 1 of preferences'")

            else:
                if possible_restaurant_count == 1: # If there is one restaurant that meet requirements
                    self.sugg_restaurant = filtered_possible_restaurants[0]

                    # explain reasoning if additional preferences were given
                    if self.informal_utterances:
                        response_utterance = f"{self.sugg_restaurant['restaurantname']} is the only match. Do you want to know something about this restaurant?"
                    else:
                        response_utterance = f"{self.sugg_restaurant['restaurantname']} is the only restaurant that meet your requirements. {self.sugg_restaurant["reasoning_explained"] if "reasoning_explained" in self.sugg_restaurant else ""} Would you like some infromation about this restaurant?"
                    
                else: # If there are multiple restaurants that meet requirements
                    # Choose random restaurant and save the others
                    self.sugg_restaurant = random.choice(filtered_possible_restaurants)
                    self.restaurants = filtered_possible_restaurants
                    self.restaurants.remove(self.sugg_restaurant)

                    # explain reasoning if additional preferences were given
                    if self.informal_utterances:
                        response_utterance = f"{self.sugg_restaurant['restaurantname']} is just what you. where looking for. {self.sugg_restaurant["reasoning_explained"] if "reasoning_explained" in self.sugg_restaurant else ""} Do you want to know something about this restaurant?"
                    else:
                        response_utterance = f"{self.sugg_restaurant['restaurantname']} is a restaurant that meet your requirements. {self.sugg_restaurant["reasoning_explained"] if "reasoning_explained" in self.sugg_restaurant else ""} Would you like some infromation about this restaurant or an alternative restaurant?"
                    
                # State "6.1 Suggest restaurant"
                next_state = "6.1 Suggest restaurant"

                if self.debug_mode:
                    print("Entered State '6.1 Suggest restaurant'")

        # State "6.1 Suggest restaurant" to "6.1 Suggest restaurant"
        elif current_state == "6.1 Suggest restaurant" and classified_dialog_act == "reqalts":
            # Check if there are other resaurants to suggest
            if len(self.restaurants) >= 1:
                self.sugg_restaurant = random.choice(self.restaurants)
                self.restaurants.remove(self.sugg_restaurant)

                next_state = "6.1 Suggest restaurant"   

                # explain reasoning if additional preferences were given
                if self.informal_utterances:
                    response_utterance = f"{self.sugg_restaurant['restaurantname']} is {"the only other" if len(self.restaurants) == 0 else "another"} match. {self.sugg_restaurant["reasoning_explained"] if "reasoning_explained" in self.sugg_restaurant else ""} Do you want to know something about this restaurant?"
                else:
                    response_utterance = f"{self.sugg_restaurant['restaurantname']} is {"the only other" if len(self.restaurants) == 0 else "another"} restaurant that meet your requirements. {self.sugg_restaurant["reasoning_explained"] if "reasoning_explained" in self.sugg_restaurant else ""} Would you like some infromation about this restaurant or a different restaurant?" 
            
            else:
                next_state = "6.1 Suggest restaurant"
                if self.informal_utterances:
                    response_utterance = "" # NEED INFORMAL UTTERANCE!!!!!
                else:
                    response_utterance = "There are no other restaurant left that meet your requirements. Would you like some infromation about the last suggested restaurant?"
                
            if self.debug_mode:
                print("Entered State '6.1 Suggest restaurant'")

        # State "6.1 Suggest restaurant" or "7.1 Provide information questioned" or "8.1 Last restaurant statement" to "7.1 Provide information questioned"
        elif current_state in ["6.1 Suggest restaurant", "7.1 Provide information questioned", "8.1 Last restaurant statement"] and classified_dialog_act == "request":
            # Generate response utterance based on requested info, like phone number, address or postcode
            if any(word in utterance for word in ["phone", "phonenumber", "telephone", "number"]):
                if self.informal_utterances:
                    response_utterance = f"You can call {self.sugg_restaurant['restaurantname']} with {self.sugg_restaurant['phone']}"
                else:
                    response_utterance = f"The phone number of restaurant {self.sugg_restaurant['restaurantname']} is {self.sugg_restaurant['phone']}"
            
            elif any(word in utterance for word in ["address","adress","street"]):
                if self.informal_utterances:
                    response_utterance = f"{self.sugg_restaurant['restaurantname']} is on {self.sugg_restaurant['addr']}"
                else:
                    response_utterance = f"The adres of {self.sugg_restaurant['restaurantname']} is {self.sugg_restaurant['addr']}"
            
            elif any(word in utterance for word in ["postal", "postcode", "zip", "post", "code"]):
                if self.informal_utterances:
                    response_utterance = f"You can find {self.sugg_restaurant['restaurantname']} over here {self.sugg_restaurant['postcode']}"
                else:
                    response_utterance = f"The postcode of {self.sugg_restaurant['restaurantname']} is {self.sugg_restaurant['postcode']}"
            
            else:
                if self.informal_utterances:
                    response_utterance = f"Here you have some information about {self.sugg_restaurant['restaurantname']}, phone number: '{self.sugg_restaurant['phone']}', address: '{self.sugg_restaurant['addr']}' and postcode: '{self.sugg_restaurant['postcode']}'"
                else:
                    response_utterance = f"The detail information of {self.sugg_restaurant['restaurantname']} are phone number: '{self.sugg_restaurant['phone']}', address: '{self.sugg_restaurant['addr']}' and postcode: '{self.sugg_restaurant['postcode']}'"

            # State "7.1 Provide information questioned"
            next_state = "7.1 Provide information questioned"

            if self.debug_mode:
                print("Entered State '7.1 Provide information questioned'")

        # State "6.1 Suggest restaurant" to "8.1 Last restaurant statement"    
        elif current_state in ["6.1 Suggest restaurant", "7.1 Provide information questioned"] and classified_dialog_act in {"confirm", "affirm", "null"}:
            # "8.1 Last restaurant statement"
            next_state = "8.1 Last restaurant statement"
            if self.informal_utterances:
                response_utterance = f"Restaurant {self.sugg_restaurant['restaurantname']} is great! You will love it!"
            else:
                response_utterance = f"Restaurant {self.sugg_restaurant['restaurantname']} is an outstanding restaurant."

            if self.debug_mode:
                print("Entered State '8.1 Last restaurant statement'")
        
        # State "8.1 Last restaurant statement" or "7.1 Provide information questioned" to "9.1 Goodbye"
        elif current_state in ["7.1 Provide information questioned", "8.1 Last restaurant statement"] and classified_dialog_act == "bye":
            # "9.1 Goodbye"
            next_state = "9.1 Goodbye"
            response_utterance = None

            if self.debug_mode:
                print("Entered State '9.1 Goodbye'")
        else:
            next_state = current_state
            if self.informal_utterances:
                response_utterance = "Sorry man, didn't get that can you rephrase it?"
            else:
                response_utterance = "I didn't understand, could you rephrase it in a different way?"

            if self.debug_mode:
                print(f"Entered no state with next_state: '{next_state}' and response_utterance: '{response_utterance}'")


        # To exit programme (DEBUG)
        if utterance == "exit" and self.debug_mode:
            next_state = "9.1 Goodbye"
            response_utterance = "Goodbye!"
        
            

        # Save state in state_history
        self.state_history.append(next_state)

        return next_state, response_utterance

if __name__ == "__main__":
    # expand csv to include 3 new columns: food_quality, crowdedness, length_of_stay
    expand_csv(Path("datasets/restaurant_info.csv"), Path("datasets/expanded_restaurant_info.csv"))

    agent = dialogAgent(model_path='saved_models/logistic_regression', debug_mode=True)
    agent.start_dialog()
