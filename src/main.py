import streamlit as st
import pandas as pd
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain.agents.agent_types import AgentType
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv


def main():

    load_dotenv()

    st.set_page_config(page_title="Ask your CSV file")
    st.header("Ask your CSV")

    user_csv = st.file_uploader("Upload your CSV file", type="csv")

    if user_csv is not None:
        df = pd.read_csv(user_csv)
        st.write("Data Preview:")
        st.dataframe(df.head())

        user_question = st.text_input("ask your question about your CSV")

        # need to initilize the language model
        llm = ChatOpenAI(temperature=0, model="gpt-4-0125-preview")
        agent = create_pandas_dataframe_agent(
            llm,
            df,
            verbose=True,
            agent_type=AgentType.OPENAI_FUNCTIONS,  # type: ignore
        )

        if user_question is not None and user_question != "":
            response = agent.run(user_question)
            st.write(response)


if __name__ == "__main__":
    main()
