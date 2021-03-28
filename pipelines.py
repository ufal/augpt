from typing import Union, List, Optional
import logging
import uuid
from uuid import UUID
import transformers
from functools import partial
from collections import OrderedDict
from model import ModelPredictor
from data import BeliefParser
from utils import AutoDatabase, AutoLexicalizer


logger = logging.getLogger()


def get_context_from_conversation(user, system):
    context = []
    user.reverse()
    system.reverse()
    user.append(None)
    system.append(None)
    for user_input, system_response in zip(user, system):
        if user_input is not None:
            context.append(user_input)
        if system_response is not None:
            context.append(system_response)
    context.reverse()
    return context


# TODO: upgrade to newer transformers
if not hasattr(transformers, 'Conversation'):
    class Conversation:
        """
        Utility class containing a conversation and its history. This class is meant to be used as an input to the
        :class:`~transformers.ConversationalPipeline`. The conversation contains a number of utility function to manage the
        addition of new user input and generated model responses. A conversation needs to contain an unprocessed user input
        before being passed to the :class:`~transformers.ConversationalPipeline`. This user input is either created when
        the class is instantiated, or by calling :obj:`conversional_pipeline.append_response("input")` after a conversation
        turn.

        Arguments:
            text (:obj:`str`, `optional`):
                The initial user input to start the conversation. If not provided, a user input needs to be provided
                manually using the :meth:`~transformers.Conversation.add_user_input` method before the conversation can
                begin.
            conversation_id (:obj:`uuid.UUID`, `optional`):
                Unique identifier for the conversation. If not provided, a random UUID4 id will be assigned to the
                conversation.

        Usage::

            conversation = Conversation("Going to the movies tonight - any suggestions?")

            # Steps usually performed by the model when generating a response:
            # 1. Mark the user input as processed (moved to the history)
            conversation.mark_processed()
            # 2. Append a mode response
            conversation.append_response("The Big lebowski.")

            conversation.add_user_input("Is it good?")
        """

        def __init__(self, text: str = None, conversation_id: UUID = None):
            if not conversation_id:
                conversation_id = uuid.uuid4()
            self.uuid: UUID = conversation_id
            self.past_user_inputs: List[str] = []
            self.generated_responses: List[str] = []
            self.history: List[int] = []
            self.new_user_input: Optional[str] = text

        def add_user_input(self, text: str, overwrite: bool = False):
            """
            Add a user input to the conversation for the next round. This populates the internal :obj:`new_user_input`
            field.

            Args:
                text (:obj:`str`): The user input for the next conversation round.
                overwrite (:obj:`bool`, `optional`, defaults to :obj:`False`):
                    Whether or not existing and unprocessed user input should be overwritten when this function is called.
            """
            if self.new_user_input:
                if overwrite:
                    logger.warning(
                        'User input added while unprocessed input was existing: "{}" was overwritten with: "{}".'.format(
                            self.new_user_input, text
                        )
                    )
                    self.new_user_input = text
                else:
                    logger.warning(
                        'User input added while unprocessed input was existing: "{}" new input ignored: "{}". '
                        "Set `overwrite` to True to overwrite unprocessed user input".format(self.new_user_input, text)
                    )
            else:
                self.new_user_input = text

        def mark_processed(self):
            """
            Mark the conversation as processed (moves the content of :obj:`new_user_input` to :obj:`past_user_inputs`) and
            empties the :obj:`new_user_input` field.
            """
            if self.new_user_input:
                self.past_user_inputs.append(self.new_user_input)
            self.new_user_input = None

        def append_response(self, response: str):
            """
            Append a response to the list of generated responses.

            Args:
                response (:obj:`str`): The model generated response.
            """
            self.generated_responses.append(response)

        def set_history(self, history: List[int]):
            """
            Updates the value of the history of the conversation. The history is represented by a list of :obj:`token_ids`.
            The history is used by the model to generate responses based on the previous conversation turns.

            Args:
                history (:obj:`List[int]`): Historyof tokens provided and generated for this conversation.
            """
            self.history = history

        def __repr__(self):
            """
            Generates a string representation of the conversation.

            Return:
                :obj:`str`:

                Example:
                Conversation id: 7d15686b-dc94-49f2-9c4b-c9eac6a1f114
                user >> Going to the movies tonight - any suggestions?
                bot >> The Big Lebowski
            """
            output = "Conversation id: {} \n".format(self.uuid)
            for user_input, generated_response in zip(self.past_user_inputs, self.generated_responses):
                output += "user >> {} \n".format(user_input)
                output += "bot >> {} \n".format(generated_response)
            if self.new_user_input is not None:
                output += "user >> {} \n".format(self.new_user_input)
            return output
else:
    Conversation = transformers.Conversation


class AuGPTConversation(Conversation):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.generated_belief = None
        self.database_results = None
        self.raw_response = None
        self.oracle_belief = None
        self.oracle_database_results = None

    def add_user_input(self, *args, **kwargs):
        super().add_user_input(*args, **kwargs)
        self.generated_belief = None
        self.database_results = None
        self.raw_response = None
        self.oracle_belief = None
        self.oracle_database_results = None


class AuGPTConversationalPipeline(transformers.Pipeline):
    """
    Multi-turn conversational pipeline.

    This conversational pipeline can currently be loaded from :func:`~transformers.pipeline` using the following
    task identifier: :obj:`"augpt-conversational"`.

    AuGPTConversationalPipeline is similar to `transformers.ConversationalPipeline`,
    but supports database and lexicalization. The interface could be the same, or if `AuGPTConversation`
    type is passed as the input, additional fields are filled by the Pipeline.

    The models that this pipeline can use are models that have been fine-tuned on a multi-turn conversational task,
    currently: `'jkulhanek/augpt-mw-21'`.

    Usage::

        conversational_pipeline = pipeline("conversational")

        conversation_1 = Conversation("Going to the movies tonight - any suggestions?")
        conversation_2 = Conversation("What's the last book you have read?")

        conversational_pipeline([conversation_1, conversation_2])

        conversation_1.add_user_input("Is it an action movie?")
        conversation_2.add_user_input("What is the genre of this book?")

        conversational_pipeline([conversation_1, conversation_2])
    """

    def __init__(self, lexicalizer=None, database=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.lexicalizer = lexicalizer
        if isinstance(lexicalizer, str):
            self.lexicalizer = AutoLexicalizer.load(lexicalizer)
        self.database = database
        if isinstance(database, str):
            self.database = AutoDatabase.load(database)
        self.predictor = ModelPredictor(self.model, self.tokenizer, device=self.device)
        self.parse_belief = BeliefParser()

    def __call__(self, conversations: Union[AuGPTConversation, Conversation, List[Union[AuGPTConversation, Conversation]]]):
        r"""
        Generate responses for the conversation(s) given as inputs.

        Args:
            conversations (a :class:`~transformers.Conversation` or a list of :class:`~transformers.Conversation`):
                Conversations to generate responses for. If `AuGPTConversation` instances are passed as the input,
                additional information is returned from the system, e.g., database results, belief state and
                delexicalized response.

        Returns:
            :class:`~transformers.Conversation` or a list of :class:`~transformers.Conversation`: Conversation(s) with
            updated generated responses for those containing a new user input.
        """

        # Input validation
        if isinstance(conversations, list):
            for conversation in conversations:
                assert isinstance(
                    conversation, Conversation
                ), "AuGPTConversationalPipeline expects a Conversation or list of Conversations as an input"
                if conversation.new_user_input is None:
                    raise ValueError(
                        "Conversation with UUID {} does not contain new user input to process. "
                        "Add user inputs with the conversation's `add_user_input` method".format(
                            type(conversation.uuid)
                        )
                    )
            assert (
                self.tokenizer.pad_token_id is not None or self.tokenizer.eos_token_id is not None
            ), "Please make sure that the tokenizer has a pad_token_id or eos_token_id when using a batch input"
        elif isinstance(conversations, Conversation):
            conversations = [conversations]
        else:
            raise ValueError("AuGPTConversationalPipeline expects a Conversation or list of Conversations as an input")

        with self.device_placement():
            contexts = AuGPTConversationalPipeline._get_contexts_from_conversations(conversations)
            original_belief_strs = self.predictor.predict_belief(contexts)
            oracle_beliefs = [getattr(x, 'oracle_belief', None) for x in conversations]
            oracle_dbs_results = [getattr(x, 'oracle_database_results', None) for x in conversations]
            beliefs = [oracle_belief if oracle_belief is not None else self.parse_belief(belief_str)
                       for oracle_belief, belief_str in zip(oracle_beliefs, original_belief_strs)]
            dbs_results = [oracle_db if oracle_db is not None else self.database(bs, return_results=True)
                           for oracle_db, bs in zip(oracle_dbs_results, beliefs)]
            dbs = [OrderedDict((k, x[0] if isinstance(x, tuple) else x) for k, x in db.items()) for db in dbs_results]
            delex_responses = self.predictor.predict_response(contexts, original_belief_strs, dbs)
            responses = [self._lexicalise(response, db, bf, ctx)
                         for response, bf, db, ctx in zip(delex_responses, beliefs, dbs_results, contexts)]

            output = []
            for conversation_index, (conversation, response, belief, db, delex) \
                    in enumerate(zip(conversations, responses, original_belief_strs, dbs_results, delex_responses)):
                conversation.mark_processed()
                conversation.append_response(response)
                if hasattr(conversation, 'generated_belief'):
                    conversation.generated_belief = belief
                if hasattr(conversation, 'database_results'):
                    conversation.database_results = db
                if hasattr(conversation, 'raw_response'):
                    conversation.raw_response = delex
                output.append(conversation)
            if len(output) == 1:
                return output[0]
            else:
                return output

    def save_pretrained(self, save_directory):
        super().save_pretrained(save_directory)
        if self.lexicalizer is not None:
            self.lexicalizer.save(save_directory)
        self.database.save(save_directory)

    @staticmethod
    def _get_contexts_from_conversations(conversations):
        def _get_context_from_conversation(conversation):
            user = list(conversation.past_user_inputs)
            if conversation.new_user_input is not None:
                user.append(conversation.new_user_input)
            system = list(conversation.generated_responses)
            return get_context_from_conversation(user, system)
        return list(map(_get_context_from_conversation, conversations))

    def _lexicalise(self, response, db, belief, context):
        if self.lexicalizer is None:
            return response
        return self.lexicalizer(response, db, belief=belief, context=context)


# Registering the pipeline with transformers if desired
transformers.pipelines.SUPPORTED_TASKS["augpt-conversational"] = {
    "impl": AuGPTConversationalPipeline,
    "tf": transformers.TFAutoModelForCausalLM if transformers.is_tf_available() else None,
    "pt": transformers.AutoModelForCausalLM if transformers.is_torch_available() else None,
    "default": {"model": {"pt": "jkulhanek/augpt-mw-21", "tf": "jkulhanek/augpt-mw-21"}}
}

# Utility function for transformers to call `pipeline('augpt-conversational')` with default model.
__old_pipeline = transformers.pipeline


def augpt_pipeline(task: str, model: Optional = None, *args, **kwargs) -> transformers.Pipeline:  # noqa
    if task == 'augpt-conversational':
        lexicalizer = kwargs.get('lexicalizer', 'default')
        database = kwargs.get('database', 'default')
        config = kwargs.get('config', None)
        model_name = model
        if model_name is None:
            model_name = 'jkulhanek/augpt-mw-21'

        # Try to infer database and lexicalizer from model or config name (if provided as str)
        if lexicalizer == 'default':
            if isinstance(model_name, str):
                lexicalizer = model_name
            elif isinstance(config, str):
                lexicalizer = config
            else:
                # Impossible to guest what is the right tokenizer here
                raise Exception(
                    "Impossible to guess which lexicalizer to use. "
                )
            kwargs['lexicalizer'] = lexicalizer

        if database == 'default':
            if isinstance(model_name, str):
                database = model_name
            elif isinstance(config, str):
                database = config
            else:
                # Impossible to guest what is the right tokenizer here
                raise Exception(
                    "Impossible to guess which database to use. "
                )
            kwargs['database'] = database

    return __old_pipeline(task, model, *args, **kwargs)


transformers.pipeline = augpt_pipeline
transformers.pipelines.pipeline = augpt_pipeline
