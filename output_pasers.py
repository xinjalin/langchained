from typing import Optional

from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field


# Defines a Pydantic model to structure the output data
class ReviewIntel(BaseModel):
    sentiment: str = Field(
        description="The sentiment field indicates the overall tone or feeling conveyed in the review"
    )
    emotion: str = Field(
        description="The emotion field captures the predominant emotional state conveyed in the review"
    )
    product_name: str = Field(
        description="The product_name field contains the name or title of the product being reviewed."
    )
    problem: Optional[str] = Field(
        description="The problem field captures any identified issues or problems mentioned in the review."
    )

    # Defines a method to convert the model instance to a dictionary for easier processing
    def to_dict(self):
        return {
            "sentiment": self.sentiment,
            "emotions": self.emotion,
            "product_name": self.product_name,
            "problems": self.problem
        }


# Create a parser instance to convert output data into ReviewIntel model instances
review_intel_parser: PydanticOutputParser = PydanticOutputParser(pydantic_object=ReviewIntel)
