import json
import random
import pandas as pd
from pathlib import Path
from infer import infer_utterance, load_artifacts
from keyword_extractor import extract_keywords
from expand_csv import expand_csv



class dialogAgent():
    def __init__(self, model_path=None, restaurant_path="datasets/restaurant_info.csv", debug_mode=False):
        """
        Initialize dialog agent
        """
        # Load in ML model or train ML model (based on input)
        self.model, self.vectorizer, self.label_encoder, self.metadata = load_artifacts(Path(model_path))

        # Save path to restaurant info file
        self.restaurant_info_path = restaurant_path
        
        # Load configuration toggles
        config_path = Path("config.json")
        confirm_key = "Ask confirmation for each preference or not"
        restart_key = "Allow dialog restarts or not"
        informal_key = "Informal langauge instead of formal"
        random_key = "Preferences are asked in a random order"
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

        # Confirmation toggle and temporary storage
        self.confirm_preferences = confirm_flag
        self.allow_restart = restart_flag
        self.informal_flag = informal_flag      
        self.random_flag = random_flag 
        self.pending_slot = None
        self.pending_value = None
        self.pending_state = None
        self.pending_prompt = None
        self.pending_message = None

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
        """
        chosen_value = "don't care" if value == "dontcare" else value
        message = f"You chose {slot} {chosen_value}. Is that correct?"

        self.pending_slot = slot
        self.pending_value = value
        self.pending_state = ask_state
        self.pending_prompt = prompt
        self.pending_message = message

        return "Confirm preference", message


    def __reset_dialog(self) -> None:
        """Reset collected preferences and confirmation state."""
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
            if self.informal_flag:
                response_utterance = "Eyoo, What's up? I'm gonna help you pick a restaurant to eat. Just tell me the area, price range and what type of food you like."
            else:
                response_utterance = "Hello , welcome to the Cambridge restaurant system? You may requests restaurants by area , price range or food type . How may I assist you?"
                
            
            self.state_history.append(next_state)
            return next_state, response_utterance

        confirm_intents = {"confirm", "affirm"}
        deny_intents = {"deny", "negate"}

        if self.confirm_preferences and current_state == "Confirm preference" and utterance:
            normalized_utt = utterance.lower().strip()
            yes_words = {"yes", "yeah", "yep", "sure", "correct", "absolutely", "affirmative", "right"}
            no_words = {"no", "nope", "nah", "negative", "not really", "don't"}

            if classified_dialog_act not in confirm_intents | deny_intents:
                if normalized_utt in yes_words:
                    if self.debug_mode:
                        print("Detected manual confirmation keyword.")
                    classified_dialog_act = "confirm"
                elif normalized_utt in no_words:
                    if self.debug_mode:
                        print("Detected manual denial keyword.")
                    classified_dialog_act = "deny"

        if self.confirm_preferences and current_state == "Confirm preference":
            if self.pending_slot is None:
                # No pending information; resume normal flow
                current_state = self.pending_state or current_state
            elif classified_dialog_act in confirm_intents:
                setattr(self, self.pending_slot, self.pending_value)

                if self.debug_mode:
                    print(f"{self.pending_slot.capitalize()} confirmed as: {self.pending_value}")

                current_state = self.pending_state or current_state

                self.pending_slot = None
                self.pending_value = None
                self.pending_state = None
                self.pending_prompt = None
                self.pending_message = None

                classified_dialog_act = ""
            elif classified_dialog_act in deny_intents:
                next_state = self.pending_state or "1. Welcome"
                response_utterance = self.pending_prompt or "Could you repeat that preference?"

                self.pending_slot = None
                self.pending_value = None
                self.pending_state = None
                self.pending_prompt = None
                self.pending_message = None

                self.state_history.append(next_state)
                return next_state, response_utterance
            else:
                next_state = "Confirm preference"
                response_utterance = self.pending_message or "Please answer yes or no so I can confirm."
                self.state_history.append(next_state)
                return next_state, response_utterance

        # Extract info based on dialog act (could be call to function)
        # only change value to 'dontcare' for the assiciated current state
        if classified_dialog_act == 'inform':
            output = extract_keywords(utterance)
            if output['area'] != None:
                if (output['area'] == 'dontcare' and current_state == "2. Fill slots") or output['area'] != 'dontcare':
                    if self.confirm_preferences:
                        next_state, response_utterance = self.__confirm_each_preference(
                            "area",
                            output['area'],
                            "2. Fill slots",
                            "What part of town do you have in mind?",
                        )
                        self.state_history.append(next_state)
                        return next_state, response_utterance

                    self.area = output['area']
            if output["pricerange"] != None:
                if (output['pricerange'] == 'dontcare' and current_state == "2. Fill slots") or output['pricerange'] != 'dontcare':
                    if self.confirm_preferences:
                        next_state, response_utterance = self.__confirm_each_preference(
                            "price",
                            output['pricerange'],
                            "2. Fill slots",
                            "How pricy would you like the restaurant to be?",
                        )
                        self.state_history.append(next_state)
                        return next_state, response_utterance

                    self.price = output['pricerange']
            if output["food"] != None:
                if (output['food'] == 'dontcare' and current_state == "2. Fill slots") or output['food'] != 'dontcare':
                    if self.confirm_preferences:
                        next_state, response_utterance = self.__confirm_each_preference(
                            "food",
                            output['food'],
                            "2. Fill slots",
                            "What kind of food would you like?",
                        )
                        self.state_history.append(next_state)
                        return next_state, response_utterance

                    self.food = output['food']

            if self.debug_mode:
                print(f"Area changed to: {self.area}") # debug
                print(f"Price changed to: {self.price}") # debug
                print(f"Food changed to: {self.food}") # debug
        
        # State transitions

        # State 0 to "1. Welcome"
        if current_state is None and utterance is None:
            # State "1. Welcome"
            next_state = "1. Welcome"
            if self.informal_flag:
                response_utterance = "Eyoo, What's up? I'm gonna help you pick a restaurant to eat. Just tell me the area, price range and what type of food you like."
            else:
                response_utterance = "Hello , welcome to the Cambridge restaurant system? You may requests restaurants by area , price range or food type . How may I assist you?"
                

            if self.debug_mode:
                print("Entered State '1. Welcome'")
        
        # State "1. Welcome" or "2.2 Ask Area" to "2.2 Ask Area"
        elif current_state in ["1. Welcome", "2. Fill slots"] and (self.area == None or self.price == None or self.food == None): # Go to the slots phase
            # Check witch slots need to be filled
            remaining_slots = []
            if self.area == None:
                remaining_slots.append("area")
            if self.price== None:
                remaining_slots.append("price")
            if self.food == None:
                remaining_slots.append("food")
            
            # Check if the order is set to random
            if len(remaining_slots) > 0:
                if self.random_flag:
                    next_slot = random.choice(remaining_slots)
                else:
                    next_slot = remaining_slots[0]
        
                # Ask Area
                if next_slot == "area":
                    next_state = "2. Fill slots"
                    if self.informal_flag:
                        next_state = "2. Fill slots"
                        response_utterance = "Where about in town do you wanna eat?"
                    else:
                        response_utterance= "Which part of town would you like the restaurant to be located?"

                # Ask Price
                if next_slot == "price":
                    next_state = "2. Fill slots"
                    if self.informal_flag:
                       response_utterance = "How pricy do you want to eat?"
                    else:
                       response_utterance= "What price range would you prefer for the restaurant?"
            
            # Ask Food type
                if next_slot == "food":
                    next_state = "2. Fill slots"
                    next_state = "2. Fill slots"
                    if self.informal_flag:
                       response_utterance = "What kind of food are you in de mood for?"
                    else:
                       response_utterance= "What type of food would you prefer?"

        
            if self.debug_mode:
                print(f"Entered State 'Fill Slots")
            
            
   
      
            
        
        # State "1. Welcome" or "2.2 "3.2 Change 1 of preferences" to "3.2 Change 1 of preferences" or "6.1 Suggest restaurant"
        elif current_state in ["1. Welcome", "2. Fill slots", "3.2 Change 1 of preferences"] and self.area != None and self.price != None and self.food != None: # 5.1 Is there a match
            # Look up possible restaurants that meet requirements
            possible_restaurants = self.__look_up_restaurants(self.area, self.price, self.food)
                
            possible_restaurant_count = len(possible_restaurants)

            if possible_restaurant_count == 0: # If there are no restaurants that meet requirements
                # State "3.2 Change 1 of preferences"
                next_state = "3.2 Change 1 of preferences"
                if self.informal_flag: 
                    response_utterance = "Sorry man, no restaurants meet your preference, do you wanna change the area, price range or food type?"
                else:
                    response_utterance= "Unfortunately there are no restaurants that meet your preferences. Would you like to change the area, price range or the foodtype?"


                if self.debug_mode:
                    print("Entered State '3.2 Change 1 of preferences'")

            else:
                if possible_restaurant_count == 1: # If there is one restaurant that meet requirements
                    self.sugg_restaurant = possible_restaurants[0]
                    if self.informal_flag:
                        response_utterance = f"{self.sugg_restaurant['restaurantname']} is the only match. Do you want to know something about this restaurant?"
                    else:
                        response_utterance= f"{self.sugg_restaurant['restaurantname']} is the only restaurant that meet your requirements. Would you like some infromation about this restaurant?"

                else: # If there are multiple restaurants that meet requirements
                    # Choose random restaurant and save the others
                    self.sugg_restaurant = random.choice(possible_restaurants)
                    self.restaurants = possible_restaurants
                    self.restaurants.remove(self.sugg_restaurant)
                    if self.informal_flag:
                        response_utterance = f"{self.sugg_restaurant['restaurantname']} is just what you where looking for. Do you want to know something about this restaurant?"
                    else:
                        response_utterance= f"{self.sugg_restaurant['restaurantname']} is the only restaurant that meet your requirements. Would you like some infromation about this restaurant?"


                    

                # State "6.1 Suggest restaurant"
                next_state = "6.1 Suggest restaurant"

                if self.debug_mode:
                    print("Entered State '6.1 Suggest restaurant'")

        # State "6.1 Suggest restaurant" to "6.1 Suggest restaurant"
        elif current_state == "6.1 Suggest restaurant" and classified_dialog_act == "reqalts":
            self.sugg_restaurant = random.choice(self.restaurants)
            self.restaurants.remove(self.sugg_restaurant)

            next_state = "6.1 Suggest restaurant"
            if self.informal_flag:
                response_utterance = f"{self.sugg_restaurant['restaurantname']} is another match. Do you want to know something about this restaurant?"
            else:
                response_utterance= f"{self.sugg_restaurant['restaurantname']} is another restaurant that meet your requirements. Would you like some infromation about this restaurant?"

           

            if self.debug_mode:
                print("Entered State '6.1 Suggest restaurant'")

        # State "6.1 Suggest restaurant" or "7.1 Provide information questioned" or "8.1 Last restaurant statement" to "7.1 Provide information questioned"
        elif current_state in ["6.1 Suggest restaurant", "7.1 Provide information questioned", "8.1 Last restaurant statement"] and classified_dialog_act == "request":
            # Generate response utterance based on requested info, like phone number, address or postcode
            if "phone" in utterance:
                if self.informal_flag:
                        response_utterance = "You can call {self.sugg_restaurant['restaurantname']} with {self.sugg_restaurant['phone']}"
                else:
                        response_utterance= "The phone number of restaurant {self.sugg_restaurant['restaurantname']} is {self.sugg_restaurant['phone']}"
                response_utterance = f"{self.sugg_restaurant['restaurantname']} is on {self.sugg_restaurant['addr']}"
            elif "postcode" in utterance:
                if self.informal_flag:
                    response_utterance = f"You can find {self.sugg_restaurant['restaurantname']} over here {self.sugg_restaurant['postcode']}"
                else:
                    response_utterance =f"The postal code of {self.sugg_restaurant['restaurantname']} is {self.sugg_restaurant['postcode']}"
            else:
                if self.informal_flag:
                    response_utterance = f"Here you have some information about {self.sugg_restaurant['restaurantname']} phone number: '{self.sugg_restaurant['phone']}', address: '{self.sugg_restaurant['addr']}' and postcode: '{self.sugg_restaurant['postcode']}'"
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
            if self.informal_flag:
                response_utterance =f"Restaurant {self.sugg_restaurant['restaurantname']} is great! You will love it!"
            else:
                response_utterance = f"Restaurant {self.sugg_restaurant['restaurantname']} is an outstanding restaurant"

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
            if self.informal_flag:
                response_utterance = "Sorry man, didn't get that can you rephrase it?"
            else:
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

    agent = dialogAgent(model_path='artifacts/dt')
    agent.start_dialog()
