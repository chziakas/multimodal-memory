from open_ai_completion import OpenAICompletion
from dotenv import load_dotenv
import os
import json




class Memory:

    # Pre-defined prompts for OpenAI's GPT model
    SYSTEM_MESSAGE = """ 
        You are an expert at extracting the information from a text and construct human memories based on that."
    """

    USER_MESSAGE_TEMPLATE = """
        Let's think step by step.
        1. Consider the text: {}.
        2. Generate a title that captures the usage of the provided text, labeled as 'title'
        3. Generate a human-like memory that could be used to retrieve the most relevant inforation later, labeled as 'memory'.
        4. Return a JSON object in the following format:'title':'memory'.
    """


    SYSTEM_MESSAGE_IMP = """ 
        You are an expert at evaluating the importance of a memory"
    """

    USER_MESSAGE_TEMPLATE_IMP = """
        Let's think step by step.
        1. Consider the text: {}.
        2. On the scale of 1 to 10, where 1 is purely mundane (e.g., brushing teeth, making bed) and 10 is extremely poignant (e.g., a break up, college acceptance), rate the likely poignancy of the following piece of memory, labeled as 'score'.
        4. Return a JSON object in the following format:'score':score.
    """

    def __init__(self, model:int, open_ai_key:str, agent_id:int, timestamp, ext_agent_ids):
        """
        Initialize the QuestionGenerator.
        """
        self.openAIcompletion = OpenAICompletion(model, open_ai_key)
        self.agent_id = agent_id
        self.timestamp = timestamp
        self.ext_agent_ids = ext_agent_ids

    def compute_importance(self, memory):
        user_message = self.USER_MESSAGE_TEMPLATE_IMP.format(memory)
        message = [
            {'role': 'system', 'content': self.SYSTEM_MESSAGE_IMP}, 
            {'role': 'user', 'content': user_message}
        ]

        openai_response = self.openAIcompletion.get_completion_from_messages(message)
        openai_response_json = self.openAIcompletion.extract_json_from_response(openai_response)
        self.importance = openai_response_json['score']


    def generate(self, text: str):
        """
        Generate a memory for that event.

        Args:
            text (str): The reference content used to generate questions.

        Returns:
            dict: A dictionary of generated questions with keys indicating the question order and values being the questions themselves.
        """
        user_message = self.USER_MESSAGE_TEMPLATE.format(text)
        message = [
            {'role': 'system', 'content': self.SYSTEM_MESSAGE}, 
            {'role': 'user', 'content': user_message}
        ]

        openai_response = self.openAIcompletion.get_completion_from_messages(message)
        openai_response_json = self.openAIcompletion.extract_json_from_response(openai_response)
        self.memory = openai_response_json
        self.compute_importance(text)

    

 


conversation = """
You: Hey, Jake! How was your day?

Jake: Oh, hey Anna! It was pretty hectic, to be honest. How about you?

You: Oh, you know, the usual. Had a couple of meetings at work, and then did some grocery shopping. Nothing too exciting. What made your day hectic?

Jake: I had this major project deadline at work and, you know, last-minute hiccups and all. Also, I had to drop off my car for servicing, so I had to rely on public transport. Which, by the way, made me realize how much I've taken my car for granted!

You: Oh, tell me about it! Public transport can be such a pain. But hey, on the brighter side, you get to see some unique characters on the bus or train.

Jake: Haha, true! Speaking of which, I saw a guy on the bus today who was dressed like he was going to a renaissance fair – complete with a feathered hat!

You: No way! That sounds hilarious. Maybe he was on his way to some themed party or event.

Jake: Possibly! Anyway, enough about my day. What's new with you? How's that new hobby of yours coming along?

You: Oh, the pottery class? It's going great! I made my first vase last week. It's a bit lopsided, but I'm proud of it.

Jake: That's awesome! You should totally show me sometime. Maybe you can make a custom mug for my coffee addiction.

You: Haha, I'll add it to my list of projects. By the way, are you up for grabbing a bite later tonight? I heard about this new sushi place downtown.

Jake: Sounds great! I'm always down for sushi. Let's do it.

You: Perfect! See you at 7 then?

Jake: 7 it is. Looking forward to it!"""

# Load environment variables from a .env file (if it exists)
load_dotenv()

# Fetch OpenAI API key from environment variables
openai_api_key = os.getenv("OPENAI_API_KEY")

agent_id = 1
external_user_id = [2]
timestamp = 'Monday'
memory_constructor = Memory("gpt-3.5-turbo-16k", openai_api_key,agent_id= agent_id,timestamp=timestamp, ext_agent_ids= external_user_id)
memory_constructor.generate(conversation)
print(memory_constructor.memory)
print(memory_constructor.importance)


