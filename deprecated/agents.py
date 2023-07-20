from typing import List
from dlabchain import AIMessage, SystemMessage, HumanMessage, ChatModel
from utils import return_agent_prompts, notes_prompts
from config import EXISTING_ROLES
import logging

log = logging.getLogger("my-logger")


from dotenv import load_dotenv  # --> tim-note to @venia: what is this..?

load_dotenv()

# Unsure what's the best way to structure prompts
#  What to put in system? What to keep in human prompt?
# How to prompt for notes? Should this be structured?
#  When should I keep track of financing?
#
# keeping track of vars here
WORD_COUNT=200  # how many words should the notes be


class NegotiationAgent:
    def __init__(self, agent_name: str, initialization_text: str, talkative: int = 512,
                 model_name="gpt-4") -> None:
        """
        Class creates agents for negotation tasks. Currently, there are
        two possible agent names, but will implement more as more
        negotation stories emerge.


        Args:
            agent_name (str): Name of agent name. Payoff table must exist.
            talkative (str) : Determines how much the agent will talk and think
        """
        self.max_tokens = talkative
        self.model_name = model_name

        self.initialization_text = initialization_text

        self.main_text, self.payoff = return_agent_prompts(agent_name) # AKA the stories
        self.init_notes_prompt, self.update_notes_prompt = notes_prompts() # determine how to write prompts

        self.context = self._combine_narratives(self.main_text, self.payoff)

        log.info("Creating systemp prompt")
        self._system_prompt = SystemMessage(self.context)

        self.init_notes_prompt = HumanMessage(self.init_notes_prompt).format_prompt("word_count", str(WORD_COUNT))
        self.update_notes_prompt = HumanMessage(self.update_notes_prompt).format_prompt("word_count", str(WORD_COUNT))


        log.info("Initializing Chat model")
        self.chat_model = ChatModel(model_name=self.model_name, max_tokens=self.max_tokens)
        # somehow need to keep track of notes evolution and
        # history of conversation to keep prompts coming.
        self.history = []
        self.notes_history = []

    def initialize_notes(self) -> None:
        """
        Initializes notes for agents prior to negotiations.
        """
        # provide agent with task and previous notes
        chat_prompt = [self._system_prompt, self.init_notes_prompt]
        print(f"Negotitation Cost will be: {self.chat_model.estimate_cost(chat_prompt, self.max_tokens)}")

        resp = self.chat_model(chat_prompt)

        # save most recent notes in self.notes.
        self.notes = resp.content

        # self.notes_history holds all notes the agent has taken.
        self.notes_history.append(resp)

        # update system prompt to also have notes.
        self.system_prompt = SystemMessage(self._combine_notes(self.context, self.notes))


    def initialize_negotations(self) -> str:
        """
        Method to be used if this agent begins negotations.

        TODO: Determine the correct initialization text.
        """

        # create chat prompt
        chat_prompt = [self.system_prompt, HumanMessage(self.initialization_text)]

        # initiialize openai model. not sure if i have to do this.
        self.chat_model = ChatModel(model_name=self.model_name, max_tokens=self.max_tokens)
        initial_negotation = self.chat_model(chat_prompt)

        # TODO: Determine how to keep track of history.
        self.history.append(initial_negotation)
        return initial_negotation.content

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
        update_prompt = [self.system_prompt]+self.history

        # initiialize openai model. not sure if i have to do this.
        self.chat_model = ChatModel(model_name=self.model_name, max_tokens=self.max_tokens)

        print("Estimated cost: ", self.chat_model.estimate_cost(update_prompt, self.max_tokens))

        # creating an LLM chain due to various bugs in just using chat model.
        response = self.chat_model(update_prompt).content

        self.history.append(AIMessage(content=response))
        return response

    def update_notes(self) -> None:
        """
        After k rounds of negotiations we will exit the
        step loop and update our `internal notes`.

        Provide agent with old notes, task description,
        conversation history.
        Note update task should be added in the negotations.

        The prompt should be structured
        - Task to update notes: {}
        - Previous notes: {}
        - Conversation history: {}

        OR  (implemented)
        - Same history and system prompt
        - Append a message asking to update notes.
        """
        # new prompt
        update_prompt = [self.system_prompt] + self.history + [self.update_notes_prompt]

        new_notes = self.chat_model(update_prompt)

        # keep memory of notes here
        self.notes_history.append(new_notes)
        self.notes = new_notes.content
        # update system prompt to also have notes.
        self.system_prompt = SystemMessage(self._combine_notes(self.context, self.notes))

    def complete_negotation(self):
        return self.notes_history


    @staticmethod
    def _combine_narratives(main_text, payoff):
        # TODO: think if this the best way to combine?
        return main_text + "\n\nPayoffs:\n\n" + payoff

    @staticmethod
    def _combine_notes(main, notes):
        # Another weird function to combien two strings.
        # This must be better designed.
        return main + "\n\nNotes:\n\n" + notes

class CostaCPCNegotiations:
    def __init__(self):
        #  define their negotation intialization text.
        cpc_negotiation_text = "You are meeting with H.P. Costa of Rio Copa Foods to begin the negotations. You will start the discussions."
        hp_costa_negotiation_text = "You are meeting with P.J. Green of CPC to begin the negotations. You will start the discussions."

        # instantiate the two agents
        self.cpc = NegotiationAgent("cpc", cpc_negotiation_text)
        self.costa = NegotiationAgent("hp_costa", hp_costa_negotiation_text)

        #  have them come up with their notes
        self.cpc.initialize_notes()
        self.costa.initialize_notes()

    def run(self, num_steps=2):
        f = open("outputs", "w")
        # let's get the ball rolling
        f.write("Costa original notes: " + self.costa.notes)
        f.write("\n\nCPC original notes: " + self.cpc.notes)

        output_text = self.cpc.initialize_negotations()
        f.write("\n\nInitial negotiations: " + output_text)
        # if cpc begins, then we must feed it into costa next
        for _ in range(2):
            for _ in range(num_steps):
                step_text = self.costa.step(output_text)
                output_text = self.cpc.step(step_text)
                f.write("\n\nCosta step:" + step_text)
                f.write("\n\nCPC step:" + output_text)

            self.costa.update_notes()
            f.write("\n\nCosta new notes: " + self.costa.notes)
            self.cpc.update_notes()
            f.write("\n\nCPC new notes: " + self.cpc.notes)

        f.write("===================")


class BuyerSellerNegotiations:
    def __init__(self):
        #  define their negotation intialization text.
        buyer_negotiation_text = "You will put in the request to purchase the object."
        seller_negotiation_text = "You will offer buyer a price to sell the object for. "

        # instantiate the two agents
        self.buyer = NegotiationAgent("buyer", buyer_negotiation_text)
        self.seller = NegotiationAgent("seller", seller_negotiation_text)

        #  have them come up with their notes
        self.buyer.initialize_notes()
        self.seller.initialize_notes()

    def run(self, num_steps=2):
        f = open("outputs", "w")
        # let's get the ball rolling
        f.write("Buyer original notes: " + self.buyer.notes)
        f.write("\n\nSeller original notes: " + self.cpc.notes)

        output_text = self.seller.initialize_negotations()
        f.write("\n\nInitial negotiations: " + output_text)
        # if cpc begins, then we must feed it into costa next
        for _ in range(2):
            for _ in range(num_steps):
                step_text = self.buyer.step(output_text)
                output_text = self.seller.step(step_text)
                f.write("\n\Buyer step:" + step_text)
                f.write("\n\Seller step:" + output_text)

            self.buyer.update_notes()
            f.write("\n\Buyer new notes: " + self.buyer.notes)
            self.seller.update_notes()
            f.write("\n\Seller new notes: " + self.seller.notes)

        f.write("===================")

if __name__=="__main__":
    ca = BuyerSellerNegotiations()

    ca.run()
