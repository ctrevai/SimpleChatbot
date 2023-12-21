
from langchain.memory import ConversationSummaryBufferMemory
from langchain.llms.bedrock import Bedrock
from langchain.chains import ConversationChain


def get_llm():

    model_kwargs = {  # anthropic
        "max_tokens_to_sample": 512,
        "temperature": 0,
        "top_k": 250,
        "top_p": 1,
        "stop_sequences": ["\n\nHuman:"]
    }

    llm = Bedrock(
        credentials_profile_name="default",
        region_name="us-east-1",
        model_id="anthropic.claude-v2:1",  # set the foundation model
        model_kwargs=model_kwargs,)  # configure the properties for Claude

    return llm


def get_memory():  # create memory for this chat session

    # ConversationSummaryBufferMemory requires an LLM for summarizing older messages
    # this allows us to maintain the "big picture" of a long-running conversation
    llm = get_llm()

    # Maintains a summary of previous messages
    memory = ConversationSummaryBufferMemory(
        llm=llm, max_token_limit=1024, ai_prefix="Assistant")

    return memory


def get_chat_response(input_text, memory):  # chat client function

    llm = get_llm()

    conversation_with_summary = ConversationChain(  # create a chat client
        llm=llm,  # using the Bedrock LLM
        memory=memory,  # with the summarization memory
        verbose=True  # print out some of the internal states of the chain while running
    )

    # pass the user message and summary to the model
    chat_response = conversation_with_summary.predict(input=input_text)

    return chat_response
