from convlab2.util.analysis_tool.analyzer import Analyzer
from convlab2.nlu.jointBERT.multiwoz import BERTNLU
from convlab2.policy.rule.multiwoz import RulePolicy
from convlab2.nlg.template.multiwoz import TemplateNLG
from convlab2.dialog_agent import Agent, PipelineAgent
from pipelines import AuGPTConversation
import data
from utils import seed


class ConvLabWrapper(Agent):
    def __init__(self, pipeline):
        super().__init__('soloist')
        self.pipeline = pipeline
        self.init_session()

    def init_session(self):
        self.conversation = AuGPTConversation()

    def response(self, observation):
        self.conversation.add_user_input(observation)
        self.conversation = self.pipeline(self.conversation)
        return self.conversation.generated_responses[-1]

    def get_in_da(self):
        return None

    def get_out_da(self):
        return None


class ConvLabAnalyzer(Analyzer):
    def __init__(self, dataset='multiwoz-test'):
        user_nlu = BERTNLU(mode='sys', config_file='multiwoz_sys_context.json',
                           model_file='https://convlab.blob.core.windows.net/convlab-2/bert_multiwoz_sys_context.zip')
        user_dst = None
        user_policy = RulePolicy(character='usr')
        user_nlg = TemplateNLG(is_user=True)
        user_agent = PipelineAgent(user_nlu, user_dst, user_policy, user_nlg, name='user')
        dataset, _ = data.split_name(dataset)
        super().__init__(user_agent, dataset)

    def __call__(self, pipeline, num_dialogs=1000):
        agent = ConvLabWrapper(pipeline)
        seed(20200202)
        result = dict()
        result['complete_rate'], result['success_rate'], result['precision'], \
            result['recall'], result['f1'], result['match'], result['avg_turns'] = \
            self.comprehensive_analyze(agent, agent.name, total_dialog=num_dialogs)
        return result
