import json
import os
import tiktoken
import pandas as pd
from model_utils import AIMessage, HumanMessage, SystemMessage, ChatModel
from utils import get_api_key

class EvaluateNegotiations:
    """
    - read note history 
    - provide summary statistics on both sides
    - look at doc for advice https://docs.google.com/document/d/1H9fGwmUllIBkFj2_KFPLiJKi4T8DMeJNFO8Wv_f-wQM/edit
    """
    def __init__(self, save_dir, game, file_name="negotiations.csv", check_message_for_offers=False):
        self.save_dir=save_dir
        self.check_message_for_offers=check_message_for_offers
        negotiations_path = os.path.join(save_dir, file_name)
        self.neg_hist = pd.read_csv(negotiations_path)
        self.game = game 
        self.issues = game.issues
        self.n_agents = len(self.game.sides)
        self.enc = None
    
    def compute_metrics(self):
        """
        - Absolute Payoff Outcomes (individual, joined, per-issue-type)
        - Number of rounds needed for outcome
        - Length of notes/messages
        - Number of tokens generated
        - Word/sentence analysis
            E.g., action words, placating words, threats etc.

        """
        n_rounds = len(self.neg_hist)
        self.neg_hist["note_length"] = self.neg_hist.apply(
            lambda x: self.estimate_tokens(x, "note"), axis=1
        )
        self.neg_hist["msg_length"] = self.neg_hist.apply(
            lambda x: self.estimate_tokens(x, "message"), axis=1
        )
        print(self.neg_hist.groupby("agent_name")[["note_length", "msg_length"]].mean())
        self.neg_hist["payoffs"] = self.neg_hist.apply(
            lambda x: self.label_to_payoff(x["issues_state"], x["agent_id"]), axis=1
        )
        if self.check_message_for_offers:
            self.neg_hist["offers_in_message"] = self.neg_hist["message"].apply(lambda x: self.check_message_for_offers(x))

        self.neg_hist[
            ["total_payoff", "normalized_payoff", "issue_payoff", "normalized_issue_payoff"]
        ] = self.neg_hist.apply(lambda x: self.payoff_analysis(x["payoffs"], x["agent_id"]), axis=1).apply(pd.Series)
        print(self.neg_hist)
        output_path = os.path.join(self.save_dir, "processed_negotiation.csv")
        # TODO implement saving of the file
        self.neg_hist.to_csv(output_path, index=False)


    def language_analysis(self, vocab_path):
        """
        vocab_path (str): a json file with different vocabularies for different constructs.
        """
        pass

    def label_to_payoff(self, issue_state, agent_id):
        """
        issue_state (dict): issue-offer pairs

        returns: payoff list [{agent_id: {issue_name: payoff}}]
        """
        payoffs = []
        # ensure dictionary
        if type(issue_state)==str:
            issue_state = eval(issue_state)

        try:
            # for agent_id in range(self.n_agents):
            for key, value in issue_state.items():
                issue = self.game.get_issue(key)
                issue_payoffs = issue.payoffs[agent_id]
                min_payoff = min(issue_payoffs)
                max_payoff = max(issue_payoffs)
                issue_payoff_labels = issue.payoff_labels[agent_id]

                idx = issue_payoff_labels.index(value)
                payoff = issue_payoffs[idx]
                payoffs.append({str(agent_id): {key: [payoff, min_payoff, max_payoff]}})
        except: 
            print(f"'{key}' not found in issues.")
        return payoffs

    def payoff_analysis(self, payoffs, agent_id):
        """
        Take the last row of the negotiations and returns a summary of the payoffs
        (individual, joined, per-issue-type)
        * compare to maximum possible payoff, i.e. normalized. 
        """
        print(payoffs)

        total_payoff = 0
        total_max_payoff = 0
        total_min_payoff = 0
        issue_payoffs = []
        normalized_issue_payoffs = []

        # iterate over final issues and measure performance
        for agent_issue in payoffs:
            for agent_id, issue_info in agent_issue.items():
                for issue_name, value in issue_info.items():
                    payoff, min_payoff, max_payoff = value
                    total_payoff += payoff
                    total_max_payoff += max_payoff
                    total_min_payoff += min_payoff
                    issue_payoffs.append([agent_id, issue_name, payoff])
                    normalized_issue_payoffs.append([agent_id, issue_name, payoff / (max_payoff - min_payoff)])
        
        try:
            normalized_total_payoff = total_payoff / (total_max_payoff - total_min_payoff)
        except ZeroDivisionError:
            return 0

        return total_payoff, normalized_total_payoff, issue_payoffs, normalized_issue_payoffs

    def beautify_summary(self):
        """
        Should print a nice summary of the negotiations.
        """
        pass

    def check_message_for_offers(self, message, model_name="gpt-3.5-turbo"):
        key = get_api_key(key="OPENAI_API_KEY")
        model = ChatModel(model_name=model_name,model_key=key)
        issues = ", ".join([issue.name for issue in self.issues])
        message_offer_prompt = """
These are the issues being discusses {issues}. 
In the following message, if an issue is being discussed extract the offer being provided

acceptable format:
```json{
    "issue_name_0": "<stated offer>",
    "issue_name_1": "<stated offer>",
    ...
}

Example 1: 
Comment: After considering your offer of $7,200, I believe we can reach a mutually beneficial agreement. How about we settle on $6,800? 

{
    "new car": "$6,800"
}

```

Message: {message}
""".replace("{issues}", issues).replace("{message}", message)
        print(message_offer_prompt)
        try:
            output = model([HumanMessage(message_offer_prompt)])
        except:
            output = {}
        return output

    def estimate_tokens(self, message, text_col="note"):
        message = len(message[text_col].split())
        return message

