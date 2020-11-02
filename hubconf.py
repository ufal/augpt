dependencies = ['torch', 'transformers', 'fuzzywuzzy']
import pipelines  # noqa
import transformers  # noqa
from model import ModelPredictor  # noqa


def augpt_conversational_pipeline(model='jkulhanek/augpt-mw-21', **kwargs):
    """
    Loads the AuGPT conversational pipeline, which could be used as a dialogue system.
    Args:
        model (str): Either the name of the model to load or the local path to the model.
            Currently supported models are 'jkulhanek/augpt-mw-21' and 'jkulhanek/augpt-mw-20' which
            are AuGPT dialogue system trained on MultiWOZ 2.1 and MultiWOZ 2.0 respectively.

    Usage:
        pipeline = augpt_conversational_pipeline()

        conversation_1 = Conversation("Going to the movies tonight - any suggestions?")
        conversation_2 = Conversation("What's the last book you have read?")

        conversational_pipeline([conversation_1, conversation_2])

        conversation_1.add_user_input("Is it an action movie?")
        conversation_2.add_user_input("What is the genre of this book?")

        conversational_pipeline([conversation_1, conversation_2])

    Returns:
        AuGPTConversationalPipeline is returned, which is similar to `transformers.ConversationalPipeline`,
        but supports database and lexicalization. The interface could be the same, or if `AuGPTConversation`
        type is passed as the input, additional fields are filled by the Pipeline.

    """
    return transformers.pipeline('augpt-conversational', model=model)


def augpt_predictor(model='jkulhanek/augpt-mw-21', **kwargs):
    """
    Returns the AuGPT model predictor, which contains LM model and tokenizer.
    Args:
        model (str): Either the name of the model to load or the local path to the model.
            Currently supported models are 'jkulhanek/augpt-mw-21', 'jkulhanek/augpt-mw-20' and
            'jkulhanek/augpt-bigdata'
    Returns:
        ModelPredictor instance.
    """
    tokenizer = transformers.AutoTokenizer.from_pretrained(model)
    model = transformers.AutoModelForCausalLM.from_pretrained(model)
    return ModelPredictor(model, tokenizer)
