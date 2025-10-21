from model_setup import model
from langchain_core.prompts import ChatPromptTemplate 

system_template="You are a translator of {language} language,Translate from {from_lang} to {language}. If you didnt found translation for certain words, write it in english word. "

prompt_template=ChatPromptTemplate.from_messages(
        [
        ("system",system_template),
        ("user","{text}")
        ]
        )
prompt=prompt_template.invoke({
    "from_lang":"English",
    "language":"Newari",
    })
print(
prompt.to_messages()
      )
