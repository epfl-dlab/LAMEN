import json
import os
import tiktoken
import pandas as pd

class EvaluateNegotiations:
    """
    - read note history 
    - provide summary statistics on both sides
    - look at doc for advice https://docs.google.com/document/d/1H9fGwmUllIBkFj2_KFPLiJKi4T8DMeJNFO8Wv_f-wQM/edit
    """
    def __init__(self, save_dir, game, file_name="negotiations.csv"):
        self.save_dir=save_dir
        negotiations_path = os.path.join(save_dir, file_name)
        # import negotiations_history
        self.neg_hist = pd.read_csv(negotiations_path)
        # load the game
        self.game = game 
        self.issues = game.issues
        # maybe problematic if n_sides != n_agents
        self.n_agents = len(self.game.sides)
        self.enc = None

    def save_summary(self, file_name="results_summary.json"):
        output_path = os.path.join(self.save_path, file_name)
        # TODO implement saving of the file

    
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
        self.neg_hist["note_length"] = self.neg_hist.apply(lambda x: self.estimate_tokens(x, "note"), axis=1)
        self.neg_hist["msg_length"] = self.neg_hist.apply(lambda x: self.estimate_tokens(x, "message"), axis=1)
        print(self.neg_hist.groupby("agent_name")[["note_length", "msg_length"]].mean())
        self.neg_hist["payoffs"] = self.neg_hist.apply(lambda x: self.label_to_payoff(x["issues_state"]), axis=1)
        print(self.neg_hist)
        

    def language_analysis(self):
        pass

    def label_to_payoff(self, issue_state):
        """
        issue_state (dict): issue-offer pairs

        returns: payoff list [{agent_id: {issue_name: payoff}}]
        """
        payoffs = []
        # ensure dictionary
        if type(issue_state)==str:
            issue_state = eval(issue_state)

        try:
            for agent_id in range(self.n_agents):
                for key, value in issue_state.items():
                    issue = self.game.get_issue(key)
                    issue_payoffs = issue.payoffs[agent_id]
                    issue_payoff_labels = issue.payoff_labels[agent_id]

                    idx = issue_payoff_labels.index(value)
                    payoff = issue_payoffs[idx]
                    payoffs.append({str(agent_id): {key: payoff}})
        except: 
            print(f"'{key}' not found in issues.")
        return payoffs

    def payoff_anaylsis(self):
        """
        Take the last row of the negotiations and returns a summary of the payoffs
        (individual, joined, per-issue-type)
        """
        final_row = self.neg_hist.tail(1)
        payoffs = final_row["payoff"]

        for agent_issue in payoffs:


    def beautify_summary(self):
        """
        Should print a nice summary of the negotiations.
        """
        pass


    def estimate_tokens(self, message, text_col="note"):
        if self.enc == None: self.enc = tiktoken.encoding_for_model(message["model_name"])
        input_tokens = len(self.enc.encode(message[text_col]))
        return input_tokens

    

    