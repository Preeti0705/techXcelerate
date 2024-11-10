import random
import re

class RuleBot:
    negative_responses = ("np", "nope", "nah", "not a chance", "sorry")
    exit_commands = ("quit", "pause", "exit", "goodbye", "bye")
    random_questions = (
        "How may I help you today?",
        "What can I do for you?",
        "What is the problem that you are facing?",
        "Do you need any assistance regarding the medical case?",
    )

    def _init_(self):
        self.alienbabble = {
            'describe_case_intent': r'.\s*disease.',
            'answer_why_intent': r'.\s*why\sare.',
            'about_case': r'.\s*case.'
        }

    def greet(self):
        self.name = input("What is the patient's name?\n")
        will_help = input(f"{self.name}, I have noted, consider giving the details of the patient and about the medical case: ")
        if will_help in self.negative_responses:
            print("Let me know if you need any assistance.")
            return
        self.chat()

    def make_exit(self, reply):
        for command in self.exit_commands:
            if reply == command:
                print("Okay, have a nice day!")
                return True
        return False

    def chat(self):
        reply = input(random.choice(self.random_questions)).lower()
        while not self.make_exit(reply):
            reply = input(self.match_reply(reply)).lower()

    def match_reply(self, reply):
        for key, value in self.alienbabble.items():
            regex_pattern = value
            found_match = re.match(regex_pattern, reply)
            if found_match and key == 'describe_case_intent':
                return self.describe_case_intent()
            elif found_match and key == 'answer_why_intent':
                return self.answer_why_intent()
            elif found_match and key == 'about_case':
                return self.about_case()
        return self.no_match_intent()

    def describe_case_intent(self):
        responses = (
            "Can you please provide any CT scans or MRI scans for the given medical case (if necessary)?\n",
            "I’m sorry to hear about that. Based on the inputs, these are the problems that are identified.\n",
            "Are there any other problems that you are dealing with?"
        )
        return random.choice(responses)

    def answer_why_intent(self):
        response = ("I understand that receiving this information can be overwhelming. Tumors vary widely in type; some can be benign (non-cancerous) and some can be cancerous.\n"
                    "I suggest you kindly go through the health care reports and make sure to take guidance from the doctors.\n"
                    "For any help, feel free to ask me any questions.")
        return response

    def about_case(self):
        response = ("Here are some of the preventive measures and precautions for a brain tumor:\n"
                    "There are more than 120 types of brain tumors, classified based on the type of cells they are made of and where they occur.\n"
                    "The five-year relative survival rate for malignant brain tumors is 35.7%. For glioblastoma, the most common type of malignant brain tumor, the five-year relative survival rate is only 6.9%.\n"
                    "Maintaining a healthy lifestyle and avoiding tobacco is important.")
        return response

    def no_match_intent(self):
        responses = ("Please tell me about the case more.\n", "How May i improve\n")
        return random.choice(responses)
 

hospitalbot=RuleBot()
hospitalbot.greet()
hospitalbot.chat()
hospitalbot.match_reply()