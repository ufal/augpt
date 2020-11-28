import pytest
import pipelines  # noqa
import transformers  # noqa
from model import ModelPredictor  # noqa


@pytest.mark.parametrize('model', ['jkulhanek/augpt-mw-21', 'jkulhanek/augpt-mw-20'])
def test_conversational_pipeline(model):
    from pipelines import AuGPTConversation, Conversation
    pipeline = transformers.pipeline('augpt-conversational', model=model)
    conversation_1 = AuGPTConversation("I need a hotel.")
    conversation_2 = Conversation("I need a train.")

    pipeline([conversation_1, conversation_2])

    assert isinstance(conversation_1.generated_responses[-1], str)
    assert len(conversation_1.generated_responses[-1]) > 0
    assert isinstance(conversation_2.generated_responses[-1], str)
    assert len(conversation_2.generated_responses[-1]) > 0
    response1 = conversation_1.generated_responses[-1]

    conversation_1.add_user_input("Another one.")
    conversation_2.add_user_input("Another input.")

    pipeline([conversation_1, conversation_2])

    assert isinstance(conversation_1.generated_responses[-1], str)
    assert len(conversation_1.generated_responses[-1]) > 0
    assert isinstance(conversation_2.generated_responses[-1], str)
    assert len(conversation_2.generated_responses[-1]) > 0
    assert response1 != conversation_1.generated_responses[-1]
