# import json
import os
import re
import pandas as pd
from utils import extract_dictionary, fuzzy_index_matching
from models.model_offer_extraction import do_offer_extraction
from games import Game
from protocols import NegotiationProtocol
from attr import define, field
import tiktoken
import copy


def _partial_agreement(is1, is2, partial=True):
    # check if internal representations are conflicting
    agreed = False
    try:
        if not any(is1) or not any(is2):
            agreed, agreed_issues = False, []
        else:
            if partial:
                keys1 = set(is1.keys())
                keys2 = set(is2.keys())

                intersecting_keys = keys1.intersection(keys2)
                agreed_issues = [k for k in intersecting_keys if is1.get(k) == is2.get(k)]
                agreed = len(agreed_issues) == len(intersecting_keys)
            else:
                is_keys = is2.keys()
                agreed_issues = [k for k in is_keys if is1.get(k) == is2.get(k)]
                agreed = len(agreed_issues) == len(is_keys)
        return agreed
    except TypeError as e:
        print(f'error in _partial_agreement() - {e}')
        return agreed


@define
class ProcessTranscript:
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
    n_agents: int = 2
    issues: list = field(factory=list)
    _df: pd.DataFrame = field(default=None)
    model_provider: str = field(default='azure')
    model_key: str = field(default=None)
    model_name: str = field(default='gpt-3.5-turbo')
    offer_extraction_model_provider: str = field(default='azure')
    offer_extraction_model_name: str = field(default='gpt-3.5-turbo')
    encoder: tiktoken.Encoding = None
    update: bool = False
    agreement_prompt: str = field(default='We agree on all issues.')

    def __attrs_post_init__(self):
        # TODO: change name + model init type
        self.encoder = tiktoken.encoding_for_model("gpt-4")
        negotiations_path = os.path.join(self.save_dir, self.file_name)
        processed_path = os.path.join(self.save_dir, self.processed_file_name)
        if os.path.exists(processed_path):
            self._df = pd.read_csv(processed_path)
            self.processed_exists = True
        else:
            self._df = pd.read_csv(negotiations_path)
        self.n_agents = len(self.game.sides)
        self.issues = self.game.issues

    @staticmethod
    def get_metric_headers(return_as_dict=False):
        headers = ['note_words', 'msg_words', 'msg_tokens', 'note_tokens', 'number_square_brackets_in_message',
                   'payoffs', 'total_payoff', 'normalized_payoff', 'issue_payoff', 'normalized_issue_payoff',
                   'propogated_payoffs', 'completion_reason', 'offers_in_message', 'offers_in_note']
        if return_as_dict:
            return {f"c_{v}": v for v in headers}

        return headers

    def compute_metrics(self):
        """
        - Absolute Payoff Outcomes (individual, joined, per-issue-type)
        - Number of rounds needed for outcome
        - Length of notes/messages
        - Number of tokens generated
        - Word/sentence analysis
            E.g., action words, placating words, threats etc.

        """
        cs = self.get_metric_headers(return_as_dict=True)
        n_cs = NegotiationProtocol.get_save_headers(return_as_dict=True)
        c_agent_id, c_completion_reason = n_cs['c_agent_id'], n_cs['c_completion_reason']
        c_note, c_message = n_cs['c_note'], n_cs['c_message']
        c_note_words, c_msg_words = cs['c_note_words'], cs['c_msg_words']
        c_note_tokens, c_msg_tokens = cs['c_note_tokens'], cs['c_msg_tokens']
        c_offers_in_note, c_offers_in_msg = cs['c_offers_in_note'], cs['c_offers_in_message']
        c_payoffs, c_propogated_payoffs = cs['c_payoffs'], cs['c_propogated_payoffs']
        c_total_payoff, c_normalized_payoff = cs['c_total_payoff'], cs['c_normalized_payoff']
        c_issue_payoff, c_normalized_issue_payoff = cs['c_issue_payoff'], cs['c_normalized_issue_payoff']

        c_number_square_brackets_in_message = cs['c_number_square_brackets_in_message']

        self._df[c_note_words] = self._df[c_note].str.split().str.len()
        self._df[c_msg_words] = self._df[c_message].str.split().str.len()

        self._df[c_msg_tokens] = self._df.apply(lambda x: self.estimate_tokens(x, c_message), axis=1)
        self._df[c_note_tokens] = self._df.apply(lambda x: self.estimate_tokens(x, c_note), axis=1)

        # if there is no processed file, extract message and note using llms.
        if not self.processed_exists:
            self._df[c_offers_in_msg] = self._df.apply(
                lambda x: self._parse_json(x, c_message, is_note=False, c_agent_id=c_agent_id), axis=1)
            self._df[c_offers_in_note] = self._df.apply(
                lambda x: self._parse_json(x, c_note, is_note=True, c_agent_id=c_agent_id), axis=1)
        # otherwise, parse out the dictionary from the string
        else:
            self._df[c_offers_in_msg] = self._df[c_offers_in_msg].apply(lambda x: self._convert_dict_to_string(x))
            self._df[c_offers_in_note] = self._df[c_offers_in_note].apply(lambda x: self._convert_dict_to_string(x))
        # add a column for completion reason
        self._update_completion_reason(c_message, c_offers_in_note, c_completion_reason)

        self._df[c_payoffs] = self._df.apply(
            lambda x: self._label_to_payoff(x[c_offers_in_note], x[c_agent_id]), axis=1)
        # propogates payoffs forward
        self._propogate_payoffs_forward(c_payoffs, c_agent_id, c_propogated_payoffs)

        # TODO: UPDATE WITH NEW FUNCTION LATER
        self._df[c_number_square_brackets_in_message] = self._df[c_message].apply(
            lambda x: self._count_square_brackets_colons(x))

        self._df[[c_total_payoff, c_normalized_payoff, c_issue_payoff, c_normalized_issue_payoff]] = (
            self._df.apply(lambda x: self.payoff_analysis(x[c_propogated_payoffs], x[c_agent_id]), axis=1).apply(
                pd.Series))

        # TODO: should we drop all rows after they reach full agreement?
        if self.update:
            # drop all rows after they agree 
            index_list = self._df[self._df[c_completion_reason] == NegotiationProtocol.get_full_agreement_string].index
            if not index_list.empty:
                drop_index = index_list[0]

                # drop all rows after this index
                self._df = self._df[:drop_index + 1]

        output_path = os.path.join(self.save_dir, self.processed_file_name)
        self._df.to_csv(output_path, index=False)

    @staticmethod
    def _convert_dict_to_string(str_dict):
        if isinstance(str_dict, str):
            try:
                out_dict = eval(str_dict)
            except Exception as e:
                out_dict = {}
                print(f'[error]: problem extracting dictionary from string - {e}')
            return out_dict
        elif isinstance(str_dict, dict):
            return str_dict
        else:
            raise NotImplementedError('offers in message / note must be string or dict')

    def _propogate_payoffs_forward(self, c_payoffs, c_agent_id, c_propogated_payoffs):
        payoffs_all_rows = []
        payoffs_agent_1 = []
        payoffs_agent_2 = []

        for i, row in self._df.iterrows():
            payoffs = row[c_payoffs]
            agent_id = row[c_agent_id]
            new_payoffs = {key: val for p in payoffs for key, val in p[str(agent_id)].items()}

            if agent_id == 0:
                last_row_1 = copy.deepcopy(payoffs_agent_1[-1]) if len(payoffs_agent_1) > 0 else {}
                last_row_1.update(new_payoffs)
                payoffs_all_rows.append(last_row_1)
                payoffs_agent_1.append(last_row_1)

            else:
                last_row_2 = copy.deepcopy(payoffs_agent_2[-1]) if len(payoffs_agent_2) > 0 else {}
                last_row_2.update(new_payoffs)
                payoffs_all_rows.append(last_row_2)
                payoffs_agent_2.append(last_row_2)

        self._df[c_propogated_payoffs] = payoffs_all_rows

    def _update_completion_reason(self, c_message, c_offers_in_note, c_completion_reason):
        # TODO: make sure this is the same function as in the protocol
        reasons = []
        completion_reason = 'in-progress'
        for i, row in self._df.iterrows():
            if i == 0:
                prev_note_offer = row[c_offers_in_note]
                prev_message = row[c_message]
                reasons.append(completion_reason)
            else:
                note_offer = row[c_offers_in_note]
                message = row[c_message]

                if (self.agreement_prompt.lower() in str(message).lower()) and (
                        self.agreement_prompt.lower() in str(prev_message).lower()):
                    completion_reason = 'disagreeing internal states, textual agreement'

                    partial_agreement = _partial_agreement(prev_note_offer, note_offer, partial=False)
                    if partial_agreement:
                        completion_reason = 'full agreement'
                else:
                    full_agreement = _partial_agreement(prev_note_offer, note_offer, partial=False)
                    if full_agreement:
                        completion_reason = 'aligning internal states, textual disagreement'

                reasons.append(completion_reason)
                prev_note_offer = note_offer
                prev_message = message
            completion_reason = 'in-progress'

        self._df[c_completion_reason] = reasons

    def _parse_json(self, x, text_col, is_note, c_agent_id):
        # temporary
        # for a re-run without using chatgpt to extract col
        pre_processed = x[text_col]
        agent_id = x[c_agent_id]

        state_dict = extract_dictionary(pre_processed)

        if state_dict is None:
            try:
                state_dict = do_offer_extraction(pre_processed, issues=self.issues, idx=agent_id, is_note=is_note,
                                                 model_name=self.offer_extraction_model_name,
                                                 model_provider=self.offer_extraction_model_provider)
            except Exception as e:
                print(f'error: unable to retrieve valid state from notes - {e}')

        return state_dict

    @staticmethod
    def _count_square_brackets_colons(message):
        # temporary script to see if the model goes off
        # the rails and begins having an dialogue with itself
        # TODO: think of a more complete way to conduct this analysis.
        try:
            num_breaks = len(re.findall("(\[.+?\])", message))
            num_colons = len(message.split(":")) - 1
            return num_breaks + num_colons
        except Exception as e:
            print(f'[error]: calculate square brackets failed - {e}')
            return None

    def language_analysis(self, vocab_path):
        """
        vocab_path (str): a json file with different vocabularies for different constructs.
        """
        pass

    def _label_to_payoff(self, issue_state, agent_id):
        """
        issue_state (dict): issue-offer pairs

        returns: payoff list [{agent_id: {issue_name: payoff}}]
        """
        payoffs = []
        # ensure dictionary
        if isinstance(issue_state, str):
            issue_state = eval(issue_state)
        for key, value in issue_state.items():
            try:
                issue = self.game.get_issue(key)
                issue_payoffs = issue.payoffs[agent_id]
                min_payoff = min(issue_payoffs)
                max_payoff = max(issue_payoffs)
                issue_payoff_labels = issue.payoff_labels[agent_id]
                idx = fuzzy_index_matching(issue_payoff_labels, value)
                payoff = 0 if idx is None else issue_payoffs[idx]
                payoffs.append({str(agent_id): {key: [payoff, min_payoff, max_payoff]}})
            except Exception as e:
                print(f"[error] unable to convert label to payoff - {e}")
        return payoffs

    def payoff_analysis(self, payoffs, agent_id):
        """
        Take the last row of the negotiations and returns a summary of the payoffs
        (individual, joined, per-issue-type)
        * compare to maximum possible payoff, i.e. normalized. 
        """
        total_payoff = 0.
        total_max_payoff = 0.
        total_min_payoff = 0.
        issue_payoffs = []
        normalized_issue_payoffs = []

        # iterate over final issues and measure performance
        for issue_name, value in payoffs.items():
            payoff, min_payoff, max_payoff = value
            total_payoff += payoff
            total_max_payoff += max_payoff
            total_min_payoff += min_payoff
            issue_payoffs.append([agent_id, issue_name, payoff])
            normalized_issue_payoffs.append([agent_id, issue_name, payoff / (max_payoff - min_payoff)])

        try:
            normalized_total_payoff = total_payoff / (total_max_payoff - total_min_payoff)
        except ZeroDivisionError:
            normalized_total_payoff = 0

        return total_payoff, normalized_total_payoff, issue_payoffs, normalized_issue_payoffs


    @staticmethod
    def estimate_words(message, text_col="note"):
        num_words = len(message[text_col].split())
        return num_words

    def estimate_tokens(self, message, text_col="note"):
        try:
            num_tokens = len(self.encoder.encode(message[text_col]))
            return num_tokens
        except TypeError as e:
            print(f'Type error in estimating tokens - {e}')
            return None
