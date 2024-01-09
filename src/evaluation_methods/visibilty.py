from attr import define, field
from .abstract_eval import AbstractEval
import pandas as pd 

@define
class SelfPlayEval(AbstractEval):
    # how many points to sample in cross play
    sample_n: int = field(default=None)
    # limit games the agents by number of issues
    limit_issues: int = field(default=2)
    run_analysis: str = field(default="all")

    def run(self):
        self._preprocess()


    def _preprocess(self): 
        """preprocess the run to make it fit for cross-play eval"""
        pass