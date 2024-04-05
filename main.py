from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain.chains.llm import LLMChain
from langchain_core.prompts import PromptTemplate

from output_pasers import review_intel_parser


def product_review(review: str) -> str:
    # template
    product_review_template = """
    given this product review {review} I want you to breakdown the:
    1: sentiment
    2: emotion
    3: product name
    4: problem
    \n{format_instructions}
    """

    # prompt template config
    review_prompt_template = PromptTemplate(
        input_variables=["review"],
        template=product_review_template,
        partial_variables={"format_instructions": review_intel_parser.get_format_instructions()}
    )

    # create an instance of OpenAI
    llm = ChatOpenAI(temperature=0, model_name='gpt-3.5-turbo')
    # creates a chain passing in the llm and template
    chain = LLMChain(llm=llm, prompt=review_prompt_template)
    # calls the invoke method on the chain to execute the template on the llm
    res = chain.invoke(input={"review": review})

    # res is a dictionary with 2 indexes ["review"] and ["text"].
    # review is the information sent to the llm and text is the response from the llm.
    return res["text"]


if __name__ == "__main__":
    # loads environment variables.
    load_dotenv()

    # Chat-GPT-3.5 generated example customer reviews.
    customer_review_positive = """
    Absolutely love my new wireless headphones! The sound quality is fantastic, and the 
    Bluetooth connectivity makes it so convenient. Plus, the battery life is impressive. Definitely worth the 
    purchase!
    """

    customer_review_neutral = """
    This coffee maker gets the job done, but it's nothing extraordinary. It brews a 
    decent cup of coffee, but the taste is pretty average. The design is simple, and it's easy to use, but it doesn't 
    offer any standout features.
    """

    customer_review_negative = """
    I was really disappointed with this blender. It claimed to be powerful, but it struggled 
    to blend even soft fruits and veggies properly. Plus, it was loud and started leaking after just a few uses. 
    Definitely not worth the money.
    """

    # passes a product review to be assessed by the llm giving a templated response.
    llm_response = product_review(customer_review_negative)
    print(llm_response)
