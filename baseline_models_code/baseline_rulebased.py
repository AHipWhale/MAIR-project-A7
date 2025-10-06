class BaselineRuleBased:
    def __init__(self):
        """
        Initialize keywords lists for each dialog act
        """
        self.ack = ['kay', 'okay']
        self.affirm = ['yes', 'right']
        self.bye = ['goodbye', 'bye']
        self.confirm = ['serve']
        self.deny = ['wrong', 'dont', 'not']
        self.hello = ['hi', 'hello', 'halo']
        self.inform = ['north', 'east', 'south', 'west', 'restaurant', 'priced', 'any', 'scottish', 'bistro', 'cheap', 'expensive', 'town', 'looking', 'mediterranean', 'seafood'] 
        self.negate = ['no']
        self.null = ['sil', 'cough', 'background_speech']
        self.repeat = ['again', 'repeat', 'back']
        self.reqalts = ['how', 'about', 'else']
        self.reqmore = ['more']
        self.request = ['address', 'postcode', 'phone', 'post']
        self.restart = ['reset', 'start']
        self.thankyou = ['thank']

    def predict(self, data: list) -> list:
        """Predict dialog act based on presence of keywords in utterance.

        Inputs:
            data: List of utterances to be classified.
        Returns:
            List of predicted dialog act labels corresponding to input utterances.
        """
        predicted_results = []

        for utterance in data:
            # Convert utterance to list of words
            words_list = utterance.split(" ")

            # Check for presence of keywords in utterance for every dialog act
            ack_res = any(elem in words_list for elem in self.ack)
            affirm_res = any(elem in words_list for elem in self.affirm)
            bye_res = any(elem in words_list for elem in self.bye)
            confirm_res = any(elem in words_list for elem in self.confirm)
            deny_res = any(elem in words_list for elem in self.deny)
            hello_res = any(elem in words_list for elem in self.hello)
            inform_res = any(elem in words_list for elem in self.inform)
            negate_res = any(elem in words_list for elem in self.negate)
            null_res = any(elem in words_list for elem in self.null)
            repeat_res = any(elem in words_list for elem in self.repeat)
            reqalts_res = any(elem in words_list for elem in self.reqalts)
            reqmore_res = any(elem in words_list for elem in self.reqmore)
            request_res = any(elem in words_list for elem in self.request)
            restart_res = any(elem in words_list for elem in self.restart)
            thankyou_res = any(elem in words_list for elem in self.thankyou)

            # Assign dialog act based on presence of keywords, in order of priority
            if inform_res:
                predicted_results.append("inform")
            elif request_res:
                predicted_results.append("request")    
            elif thankyou_res:
                predicted_results.append("thankyou")
            elif reqalts_res:
                predicted_results.append("reqalts")
            elif null_res:
                predicted_results.append("null")
            elif affirm_res:
                predicted_results.append("affirm")
            elif negate_res:
                predicted_results.append("negate")
            elif bye_res:
                predicted_results.append("bye")
            elif confirm_res:
                predicted_results.append("confirm")
            elif hello_res:
                predicted_results.append("hello")
            elif repeat_res:
                predicted_results.append("repeat")
            elif ack_res:
                predicted_results.append("ack")
            elif deny_res:
                predicted_results.append("deny")
            elif restart_res:
                predicted_results.append("restart")
            elif reqmore_res:
                predicted_results.append("reqmore")
            else:
                predicted_results.append("inform")
                   
        return predicted_results
