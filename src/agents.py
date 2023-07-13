from typing import List
from langchain.prompts import (
    ChatPromptTemplate,
    PromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder
)
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)
from langchain.chat_models import ChatOpenAI

from utils import return_agent_prompts, notes_prompts
from config import EXISTING_ROLES
from dotenv import load_dotenv

load_dotenv() 

import logging
log = logging.getLogger("my-logger")

# Unsure what's the best way to structure prompts
# What to put in system? What to keep in human prompt?
# How to prompt for notes? Should this be structured? 

# keeping track of vars here
WORD_COUNT=200  # how many words should the notes be

class NegotiationAgent:
    def __init__(self, agent_name: str) -> None:
        self.main_text, self.payoff = return_agent_prompts(agent_name) # AKA the stories
        self.init_notes_prompt, self.update_notes_prompt = notes_prompts() # determine how to write prompts

        self.context = self._combine_narratives(self.main_text, self.payoff)

        log.info("Creating systemp prompt")
        # self.system_prompt = ChatPromptTemplate.from_messages([SystemMessage(content=self.context), MessagesPlaceholder(variable_name="note_gen")])
        self._system_prompt = ChatPromptTemplate.from_messages([SystemMessage(content=self.context), MessagesPlaceholder(variable_name="note_gen")])
        
        self.init_notes_prompt = HumanMessage(content=self.init_notes_prompt.replace("{word_count}", str(WORD_COUNT)))
        self.update_notes_prompt = HumanMessagePromptTemplate.from_template(self.update_notes_prompt)

        log.info("Initializing Chat model")
        self.chat_model = ChatOpenAI()
        
        self.history = []

    def initialize_notes(self):
        chat_prompt = self._system_prompt.format_prompt(note_gen=[self.init_notes_prompt]).to_messages()
        resp = self.chat_model(chat_prompt)
        self.notes = resp.content
        self.history += AIMessage(content=self.notes)
        self.system_prompt = SystemMessagePromptTemplate.from_template(self._combine_notes(self.context, self.notes))


    def initialize_negotations(self, initialization_text="Start negotations."):
        """
        Method to be used if this agent begins negotations.

        Determine the correct initialization text.
        """
        chat_prompt = ChatPromptTemplate.from_messages([
                    self.system_prompt, 
                    HumanMessagePromptTemplate.from_template(initialization_text)]
                )
        print(chat_prompt)
        self.chat_model = ChatOpenAI()

        initial_negotation = self.chat_model(chat_prompt).content
        self.history += AIMessage(content=initial_negotation)
        return initial_negotation
    
    def step(self, new_message):
        self.history += HumanMessage(content=new_message)
        update_prompt = ChatPromptTemplate.from_messages(self.system_prompt, self.history)
        next_negotiation = self.chat_model(update_prompt)
        self.history += next_negotiation
        return next_negotiation.content
        

    def complete_negotation(self):
        return self.notes

    def update_notes(self):
        pass 

    @staticmethod
    def _combine_narratives(main_text, payoff):
        # TODO: think if this the best way to combine?
        return main_text + "\n\nPayoffs:\n\n" + payoff
    
    @staticmethod
    def _combine_notes(main, notes):
        return main + "\n\nNotes:\n\n" + notes

class CollaborativeAgents:
    def __init__(self, agent_list: List[NegotiationAgent]):
        pass


if __name__=="__main__":
    # preliminatry code running
    # check init
    na = NegotiationAgent("cpc")

    # check note-taking process
    na.initialize_notes()
    
    # check initialize negotations 
    na.initialize_negotations()