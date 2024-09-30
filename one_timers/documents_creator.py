from langchain.prompts import PromptTemplate
from langchain_ollama.llms import OllamaLLM
from tqdm import tqdm


def create_stories_from_prompt(creator):
    with open("documents_creation_prompt.txt", "r", encoding="utf-8") as f:
        template = f.read()

    prompt = PromptTemplate(
        input_variables=["creator"],
        template=template,
    )

    model = OllamaLLM(model="mistral")
    chain = prompt | model
    response = chain.invoke({"creator": creator})
    return response


def save_story_in_md(story, index):
    with open(
        f"../generated_stories/story_{index}.md",
        "w+",
        encoding="utf-8"
    ) as f:
        f.write(story)


def create_stories():
    authors = [
        "Jorge Luis Borges",
        "Gabriel García Márquez",
        "Julio Cortázar",
        "Mario Vargas Llosa",
        "Pablo Neruda",
        "Isabel Allende",
        "Agatha Christie",
        "Stephen King",
        "J.K. Rowling",
        "George R.R. Martin",
        "Haruki Murakami",
    ]

    for index, author in tqdm(enumerate(authors)):
        story = create_stories_from_prompt(author)
        save_story_in_md(story, index)


if __name__ == "__main__":
    create_stories()
