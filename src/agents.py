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
from langchain import LLMChain

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
        """
        Class creates agents for negotation tasks. Currently, there are 
        two possible agent names, but will implement more as more
        negotation stories emerge. 
        

        Args:
            agent_name (str): Name of agent name. Payoff table must exist. 
        """
        
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
        
        
        # somehow need to keep track of notes evolution and 
        # history of conversation to keep prompts coming.
        self.history = []
        self.notes_history = []

    def initialize_notes(self) -> None: 
        """
        Initializes notes for agents prior to negotiations. 
        """
        chat_prompt = self._system_prompt.format_prompt(note_gen=[self.init_notes_prompt]).to_messages()
        resp = self.chat_model(chat_prompt)
        self.notes = resp.content
        self.notes_history += AIMessage(content=self.notes)
        
        # update system prompt to also have notes.
        self.system_prompt = SystemMessagePromptTemplate.from_template(self._combine_notes(self.context, self.notes))


    def initialize_negotations(self, initialization_text: str) -> str:
        """
        Method to be used if this agent begins negotations.
        

        TODO: Determine the correct initialization text.
        """
        
        # create chat prompt
        chat_prompt = ChatPromptTemplate.from_messages([
                    self.system_prompt, 
                    HumanMessagePromptTemplate.from_template(initialization_text)]
                )
        
        # initiialize openai model. not sure if i have to do this. 
        self.chat_model = ChatOpenAI(max_tokens=128,model_name="gpt-4")
        
        # creating an LLM chain due to various bugs in just using chat model. 
        llm = LLMChain(llm=self.chat_model, prompt=chat_prompt)
        
        # TODO: figure out why you have to add an input to the llm run function.
        initial_negotation = llm.run(input="")          # outputs a string!(???)
        
        # TODO: Determine how to keep track of history.
        self.history.append(AIMessage(content=initial_negotation))
        return initial_negotation
    
    def step(self, new_message: str) -> str:
        """Takes the message from the previous output and has this agent
        outputs an a response.

        Args:
            new_message (str): Message from (other) agent

        Returns:
            str: New message from (this) agent.
        """
        # we will first add the other output to this agents history.
        self.history.append(HumanMessage(content=new_message))
        update_prompt = ChatPromptTemplate.from_messages([self.system_prompt] + self.history)
        # initiialize openai model. not sure if i have to do this. 
        self.chat_model = ChatOpenAI(max_tokens=128,model_name="gpt-4")
        
        # creating an LLM chain due to various bugs in just using chat model. 
        llm = LLMChain(llm=self.chat_model, prompt=update_prompt)
        response = llm.run(input="")          # outputs a string!(???)

        self.history.append(AIMessage(content=response))
        return response

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
        # Another weird function to combien two strings. 
        # This must be better designed.
        return main + "\n\nNotes:\n\n" + notes

class CollaborativeAgents:
    def __init__(self):
        # instantiate the two agents
        self.cpc = NegotiationAgent("cpc")
        self.costa = NegotiationAgent("hp_costa")
        
        # define their negotation intialization text.
        self.cpc_negotiation_text = "You are meeting with H.P. Costa of Rio Copa Foods to begin the negotations. You will start the discussions."
        self.chp_costa_negotiation_text = "You are meeting with P.J. Green of CPC to begin the negotations. You will start the discussions."

        # have them come up with their notes
        self.cpc.initialize_notes()
        self.costa.initialize_notes()
        
    def run(self, num_steps=2):
        f = open("outputs", "a")
        # let's get the ball rolling
        output_text = self.cpc.initialize_negotations(self.cpc_negotiation_text)
        f.write(output_text)
        # if cpc begins, then we must feed it into costa next
        for _ in range(2):
            step_text = self.costa.step(output_text)
            output_text = self.cpc.step(step_text)
            f.write(step_text)
            f.write(output_text)
            
        

if __name__=="__main__":
    ca = CollaborativeAgents()
    
    ca.run()