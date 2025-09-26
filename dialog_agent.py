import json
import random
import pandas as pd
from pathlib import Path
from infer import infer_utterance, load_artifacts
from keyword_extractor import extract_keywords
from expand_csv import expand_csv

# Shared greeting reused for both the initial turn and any restart.
WELCOME_MESSAGE = "Hello , welcome to the Cambridge restaurant system? You can ask for restaurants by area , price range or food type . How may I help you?"

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
        confirm_flag = False
        restart_flag = False
        if config_path.exists():
            try:
                with config_path.open("r", encoding="utf-8") as config_file:
                    config_data = json.load(config_file)
                    confirm_flag = config_data.get(confirm_key, False)
                    restart_flag = config_data.get(restart_key, False)
            except (json.JSONDecodeError, OSError):
                confirm_flag = False
                restart_flag = False

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
            print(f"Matches: {matches_dict}") # debug

        return matches_dict
        

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
        if classified_dialog_act == 'inform':
            output = extract_keywords(utterance)
            captured_entries = []

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
        
        # State transitions

        # State 0 to "1. Welcome"
        if current_state is None and utterance is None:
            # State "1. Welcome"
            next_state = "1. Welcome"
            response_utterance = WELCOME_MESSAGE

            if self.debug_mode:
                print("Entered State '1. Welcome'")
        
        # State "1. Welcome" or "2.2 Ask Area" to "2.2 Ask Area"
        elif current_state in ["1. Welcome", "2.2 Ask Area"] and self.area == None: # 2.1 Area Known?
            # State "2.2 Ask Area"
            next_state = "2.2 Ask Area"
            response_utterance = "What part of town do you have in mind?"

            if self.debug_mode:
                print("Entered State '2.2 Ask Area'")

        # State "1. Welcome" or "2.2 Ask Area" or "3.2 Ask price" to "3.2 Ask price"
        elif current_state in ["1. Welcome", "2.2 Ask Area", "3.2 Ask price"] and self.price == None: # 3.1 Price known?
            # State "3.2 Ask price"
            next_state = "3.2 Ask price"
            response_utterance = "How pricy would you like the restaurant to be?"

            if self.debug_mode:
                print("Entered State '3.2 Ask price'")
        
        # State "1. Welcome" or "2.2 Ask Area" or "3.2 Ask price" or "4,2 Ask Food type" to "4.2 Ask Food type"
        elif current_state in ["1. Welcome", "2.2 Ask Area", "3.2 Ask price", "4.2 Ask Food type"] and self.food == None: # 4.1 Food type known?
            # State "4.2 Ask Food type"
            next_state = "4.2 Ask Food type"
            response_utterance = "What kind of food would you like?"

            if self.debug_mode:
                print("Entered State '4.2 Ask Food type'")
        
        # State "1. Welcome" or "2.2 Ask Area" or "3.2 Ask price" or "4.2 Ask Food type" or "5.2 Change 1 of preferences" to "5.2 Change 1 of preferences" or "6.1 Suggest restaurant"
        elif current_state in ["1. Welcome", "2.2 Ask Area", "3.2 Ask price", "4.2 Ask Food type", "5.2 Change 1 of preferences"] and self.area != None and self.price != None and self.food != None: # 5.1 Is there a match
            # Look up possible restaurants that meet requirements
            possible_restaurants = self.__look_up_restaurants(self.area, self.price, self.food)
                
            possible_restaurant_count = len(possible_restaurants)

            if possible_restaurant_count == 0: # If there are no restaurants that meet requirements
                # State "5.2 Change 1 of preferences"
                next_state = "5.2 Change 1 of preferences"
                response_utterance = "There are no restaurants that meet your requirements. Would you like to change the area, pricerange or foodtype? And what would you like to change it to?"

                if self.debug_mode:
                    print("Entered State '5.2 Change 1 of preferences'")

            else:
                if possible_restaurant_count == 1: # If there is one restaurant that meet requirements
                    self.sugg_restaurant = possible_restaurants[0]
                    response_utterance = f"{self.sugg_restaurant['restaurantname']} is the only restaurant that meet your requirements. Would you like some infromation about this restaurant?"
                else: # If there are multiple restaurants that meet requirements
                    # Choose random restaurant and save the others
                    self.sugg_restaurant = random.choice(possible_restaurants)
                    self.restaurants = possible_restaurants
                    self.restaurants.remove(self.sugg_restaurant)

                    response_utterance = f"{self.sugg_restaurant['restaurantname']} is a restaurant that meet your requirements. Would you like some infromation about this restaurant or an alternative restaurant?"

                # State "6.1 Suggest restaurant"
                next_state = "6.1 Suggest restaurant"

                if self.debug_mode:
                    print("Entered State '6.1 Suggest restaurant'")

        # State "6.1 Suggest restaurant" to "6.1 Suggest restaurant"
        elif current_state == "6.1 Suggest restaurant" and classified_dialog_act == "reqalts":
            self.sugg_restaurant = random.choice(self.restaurants)
            self.restaurants.remove(self.sugg_restaurant)

            next_state = "6.1 Suggest restaurant"
            response_utterance = f"{self.sugg_restaurant['restaurantname']} is another restaurant that meet your requirements. Would you like some infromation about this restaurant or a different restaurant?" 

            if self.debug_mode:
                print("Entered State '6.1 Suggest restaurant'")

        # State "6.1 Suggest restaurant" or "7.1 Provide information questioned" or "8.1 Last restaurant statement" to "7.1 Provide information questioned"
        elif current_state in ["6.1 Suggest restaurant", "7.1 Provide information questioned", "8.1 Last restaurant statement"] and classified_dialog_act == "request":
            # Generate response utterance based on requested info, like phone number, address or postcode
            if "phone" in utterance:
                response_utterance = f"The phone number of restaurant {self.sugg_restaurant['restaurantname']} is {self.sugg_restaurant['phone']}"
            elif "address" in utterance:
                response_utterance = f"{self.sugg_restaurant['restaurantname']} is on {self.sugg_restaurant['addr']}"
            elif "postcode" in utterance:
                response_utterance = f"The postcode of {self.sugg_restaurant['restaurantname']} is {self.sugg_restaurant['postcode']}"
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
            response_utterance = f"Restaurant {self.sugg_restaurant['restaurantname']} is a great restaurant"

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
            response_utterance = "I didn't understand, could you rephrase it in a different way?"


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

    agent = dialogAgent(model_path='artifacts/dt', debug_mode=True)
    agent.start_dialog()
