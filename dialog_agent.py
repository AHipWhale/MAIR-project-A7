import random
import pandas as pd
from pathlib import Path
from infer import infer_utterance, load_artifacts
from keyword_extractor import extract_keywords

class dialogAgent():
    def __init__(self, model_path=None, restaurant_path="datasets/restaurant_info.csv"):
        """
        Initialize dialog agent
        """
        # Load in ML model or train ML model (based on input)
        self.model, self.vectorizer, self.label_encoder, self.metadata = load_artifacts(Path(model_path))

        # Save path to restaurant info file
        self.restaurant_info_path = restaurant_path
        
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

    def start_dialog(self):
        # Loop state_transition until certain state is reached (end state)
        state = None
        user_input = None
        while state != "9.1 Goodbye":
            # State transition
            state, system_utterance = self.__state_transition(state, user_input)

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

        # print(f"Matches: {matches_dict}") # debug
        return matches_dict
        

    def __state_transition(self, current_state: str, utterance: str) -> tuple:
        next_state = None
        response_utterance = None

        # Classify dialog act (could be call to function)
        if utterance != None:
            classified_dialog_act = infer_utterance(self.model, self.vectorizer, self.label_encoder, self.metadata, utterance)
            print(f"user utterance classified as: {classified_dialog_act}")
        else:
            classified_dialog_act = ""

        # Extract info based on dialog act (could be call to function)
        if classified_dialog_act == 'inform':
            output = extract_keywords(utterance)
            if output['area'] != None:
                self.area = output['area']
            if output["pricerange"] != None:
                self.price = output['pricerange']
            if output["food"] != None:
                self.food = output['food']

            print(f"Area changed to: {self.area}") # debug
            print(f"Price changed to: {self.price}") # debug
            print(f"Food changed to: {self.food}") # debug
        
        # State transitions

        # State 0 to "1. Welcome"
        if current_state is None and utterance is None:
            # State "1. Welcome"
            next_state = "1. Welcome"
            response_utterance = "Hello , welcome to the Cambridge restaurant system? You can ask for restaurants by area , price range or food type . How may I help you?"

            print("Entered State '1. Welcome'")
        
        # State "1. Welcome" or "2.2 Ask Area" to "2.2 Ask Area"
        elif current_state in ["1. Welcome", "2.2 Ask Area"] and self.area == None: # 2.1 Area Known?
            # State "2.2 Ask Area"
            next_state = "2.2 Ask Area"
            response_utterance = "What part of town do you have in mind?"

            print("Entered State '2.2 Ask Area'")

        # State "1. Welcome" or "2.2 Ask Area" or "3.2 Ask price" to "3.2 Ask price"
        elif current_state in ["1. Welcome", "2.2 Ask Area", "3.2 Ask price"] and self.price == None: # 3.1 Price known?
            # State "3.2 Ask price"
            next_state = "3.2 Ask price"
            response_utterance = "How pricy would you like the restaurant to be?"

            print("Entered State '3.2 Ask price'")
        
        # State "1. Welcome" or "2.2 Ask Area" or "3.2 Ask price" or "4,2 Ask Food type" to "4.2 Ask Food type"
        elif current_state in ["1. Welcome", "2.2 Ask Area", "3.2 Ask price", "4.2 Ask Food type"] and self.food == None: # 4.1 Food type known?
            # State "4.2 Ask Food type"
            next_state = "4.2 Ask Food type"
            response_utterance = "What kind of food would you like?"

            print("Entered State '4.2 Ask Food type'")
        
        # State "1. Welcome" or "2.2 Ask Area" or "3.2 Ask price" or "4.2 Ask Food type" or "5.2 Change 1 of preferences" to "5.2 Change 1 of preferences" or "6.1 Suggest restaurant"
        elif current_state in ["1. Welcome", "2.2 Ask Area", "3.2 Ask price", "4.2 Ask Food type", "5.2 Change 1 of preferences"] and self.area != None and self.price != None and self.food != None: # 5.1 Is there a match
            # Look up possible restaurants that meet requirements
            possible_restaurants = self.__look_up_restaurants(self.area, self.price, self.food)
                
            possible_restaurant_count = len(possible_restaurants)

            if possible_restaurant_count == 0:
                # State "5.2 Change 1 of preferences"
                next_state = "5.2 Change 1 of preferences"
                response_utterance = "There are no restaurants that meet your requirements. Would you like to change the area, pricerange or foodtype? And what would you like to change it to?"
            
                print("Entered State '5.2 Change 1 of preferences'")

            else:
                if possible_restaurant_count == 1:
                    self.sugg_restaurant = possible_restaurants[0]
                else:
                    # Choose random restaurant and save the others
                    self.sugg_restaurant = random.choice(possible_restaurants)
                    self.restaurants = possible_restaurants
                    self.restaurants.remove(self.sugg_restaurant)

                # State "6.1 Suggest restaurant"
                next_state = "6.1 Suggest restaurant"
                response_utterance = f"{self.sugg_restaurant['restaurantname']} is a restaurant that meet your requirements. Would you like some infromation about this restaurant or an alternative restaurant?"

                print("Entered State '6.1 Suggest restaurant'")

        # State "6.1 Suggest restaurant" to "6.1 Suggest restaurant"
        elif current_state == "6.1 Suggest restaurant" and classified_dialog_act == "reqalts":
            self.sugg_restaurant = random.choice(self.restaurants)
            self.restaurants.remove(self.sugg_restaurant)

            next_state = "6.1 Suggest restaurant"
            response_utterance = f"{self.sugg_restaurant['restaurantname']} is another restaurant that meet your requirements. Would you like some infromation about this restaurant or a different restaurant?" 

            print("Entered State '6.1 Suggest restaurant'")

        # State "6.1 Suggest restaurant" or "7.1 Provide information questioned" to "7.1 Provide information questioned"
        elif current_state in ["6.1 Suggest restaurant", "7.1 Provide information questioned"] and classified_dialog_act == "request":
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

            print("Entered State '7.1 Provide information questioned'")

        # State "6.1 Suggest restaurant" to "8.1 Last restaurant statement"    
        elif current_state in ["6.1 Suggest restaurant", "7.1 Provide information questioned"] and (classified_dialog_act == "confirm" or classified_dialog_act == "null"):
            # "8.1 Last restaurant statement"
            next_state = "8.1 Last restaurant statement"
            response_utterance = f"Restaurant {self.sugg_restaurant['restaurantname']} is a great restaurant"

            print("Entered State '8.1 Last restaurant statement'")
        
        # State "8.1 Last restaurant statement" or "7.1 Provide information questioned" to "9.1 Goodbye"
        elif current_state in ["7.1 Provide information questioned", "8.1 Last restaurant statement"] and classified_dialog_act == "bye":
            # "9.1 Goodbye"
            next_state = "9.1 Goodbye"
            response_utterance = None

            print("Entered State '9.1 Goodbye'")
        else:
            next_state = current_state
            response_utterance = "I didn't understand, could you rephrase it in a different way?"


        # To exit programme (DEBUG)
        if utterance == "exit":
            next_state = "9.1 Goodbye"
            response_utterance = "Goodbye!"
        
            

        # Save state in state_history
        self.state_history.append(next_state)

        return next_state, response_utterance

if __name__ == "__main__":
    agent = dialogAgent(model_path='artifacts/dt')
    agent.start_dialog()

