# import json
import os
# import tiktoken
import pandas as pd
from utils import get_api_key
from model_utils import check_text_for_offers
from games import Game
from attr import define, field

@define
class EvaluateNegotiations:
    """
    - read note history 
    - provide summary statistics on both sides
    - look at doc for advice https://docs.google.com/document/d/1H9fGwmUllIBkFj2_KFPLiJKi4T8DMeJNFO8Wv_f-wQM/edit
    """
    save_dir: str
    game: Game
    file_name: str = "negotiations.csv"
    processed_file_name: str = "processed_negotiation.csv"
    processed_exists: bool = False
    check_message_for_offers: bool = False
    n_agents: int = 2
    issues: list = field(factory=list)
    neg_hist: pd.DataFrame = field(default=None)
    model_provider: str = 'openai'
    model_key: str = field(default=None)
    model_name: str = 'gpt-3.5-turbo'

    def __attrs_post_init__(self):
        negotiations_path = os.path.join(self.save_dir, self.file_name)
        processed_path = os.path.join(self.save_dir, self.processed_file_name)
        if os.path.exists(processed_path):
            self.neg_hist = pd.read_csv(processed_path)
            self.processed_exists = True
        else:
            self.neg_hist = pd.read_csv(negotiations_path)
        self.n_agents = len(self.game.sides)
        self.issues = self.game.issues

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

        self.neg_hist["offers_in_message"] = self.neg_hist["message"].apply(
            lambda x: check_text_for_offers(x, self.issues))
        
        self.neg_hist["offers_in_note"] = self.neg_hist["note"].apply(
            lambda x: check_text_for_offers(x, self.issues))

        self.neg_hist[
            ["total_payoff", "normalized_payoff", "issue_payoff", "normalized_issue_payoff"]
        ] = self.neg_hist.apply(lambda x: self.payoff_analysis(x["payoffs"], x["agent_id"]), axis=1).apply(pd.Series)

        output_path = os.path.join(self.save_dir, "processed_negotiation.csv")
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
        if type(issue_state) == str:
            issue_state = eval(issue_state)

        key = None
        # for agent_id in range(self.n_agents):
        for key, value in issue_state.items():
            try:
                issue = self.game.get_issue(key)
                issue_payoffs = issue.payoffs[agent_id]
                min_payoff = min(issue_payoffs)
                max_payoff = max(issue_payoffs)
                issue_payoff_labels = issue.payoff_labels[agent_id]
                idx = self.fuzzy_index_matching(issue_payoff_labels, value)
                payoff = issue_payoffs[idx]
                payoffs.append({str(agent_id): {key: [payoff, min_payoff, max_payoff]}})
            except (NotImplementedError) as e:
                print(f"{e}")
        # except Exception as e:
        #     print(f"'{key}' not found in issues. - {e}")
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
    
    @staticmethod
    def fuzzy_index_matching(lst, value):
        '''
        Function to find the index of an element in a list.
        If the element is not found, it returns the index of the element with closest string lexicographically.
        '''
        
        # Try to find the index in the list
        try:
            return lst.index(value)
        
        # If value is not in the list
        except ValueError:
            min_diff = float('inf')
            min_index = -1
            for i, el in enumerate(lst):
                # Calculate lexicographical difference
                diff = abs(ord(value[0]) - ord(el[0]))
        
                if diff < min_diff:
                    min_diff = diff
                    min_index = i
            # If value not found, return lexicographically closest element's index
            return min_index

    def estimate_tokens(self, message, text_col="note"):
        # TODO: use estimate token from ChatModel class
        message = len(message[text_col].split())
        return message
