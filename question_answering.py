import ast
import os
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, GoogleGenerativeAI, ChatGoogleGenerativeAI
from langchain.tools.retriever import create_retriever_tool
from langchain.agents import create_react_agent, Tool, AgentExecutor
from langchain import hub
from langchain_text_splitters import TokenTextSplitter
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnableSequence
from langchain_groq import ChatGroq
from langchain.agents import load_tools ,initialize_agent

import json
import re
# from langchain_core.documents import Document

from temp import resume_content
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
os.environ['TAVILY_API_KEY'] = os.getenv("TAVILY_API_KEY")
os.environ['GOOGLE_API_KEY'] = os.getenv("GEMINI_API_KEY")
os.environ['GROQ_API_KEY'] = os.getenv("GROQ_API_KEY")

os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ["LANGCHAIN_TRACING"] = "false"
os.environ["LANGCHAIN_API_KEY"] = ""

agent_base_prompt = hub.pull("hwchase17/react")

class QuestionAnsweringAgent:
    def __init__(self, resume_content: str):
        """
        Initialize the question-answering agent with LLM, tools, and vector store based on the provided resume content.
        """
        self.llm = GoogleGenerativeAI(model="gemini-pro")
        self.llm_chat = ChatGoogleGenerativeAI(model="gemini-pro")
        self.llm_grok_chat=ChatGroq(
            model="llama3-70b-8192",
            temperature=0,
            max_tokens=None,
            timeout=None,
            max_retries=2,
            # other params...
        )
        self.search = TavilySearchResults(max_results=1)
        self.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        # self.llm_transformer = LLMGraphTransformer(llm=self.llm)
        self.resume_content=self.contextual_resume_content(resume_content)
        # self.resume_knowledge_base=self.get_knowledge_base()

        # Initialize Google Search Tool
        self.google_tool = Tool(
            name="google-search",
            description=(
                "Leverage Google Search to find recent and reliable external information. "
                "Use it for queries that require real-time or general knowledge beyond the resume or your knowledge base."
            ),
            func=self.search.run,
            return_direct=True,
        )

        # Load the REACT prompt from LangChain hub
        self.prompt = agent_base_prompt
        # Split and index resume content
        # text_splitter = TokenTextSplitter(chunk_size=200, chunk_overlap=20)
        # texts = text_splitter.split_text(self.resume_content)
        self.vectorstore = FAISS.from_texts(self.resume_content, embedding=self.embeddings)

        # Create retriever tool for resume content
        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 3})

        self.retriever_tool = create_retriever_tool(
            self.retriever,
            name="resume_question_answering",
            description=(
                "Use this tool to answer questions about the resume."
                "It retrieves relevant information about the candidate's experience, skills, projects, cgpa, percentage, personal information, achievements, etc."
            ),
        )

        # Simple LLM Call Tool
        self.simple_llm_tool = Tool(
            name="simple-llm",
            description=(
                "Use this tool to get a straightforward answer from the LLM for any query. "
                "It is ideal for basic questions or when additional processing is unnecessary."
            ),
            func=self.llm_grok_chat.invoke,
        )

        # Combine tools
        self.tools = [self.google_tool, self.retriever_tool, self.simple_llm_tool]


        # Create REACT agent
        self.agent = create_react_agent(self.llm_grok_chat, self.tools, self.prompt)
        self.agent_executor = AgentExecutor(
            agent=self.agent, 
            tools=self.tools, 
            verbose=True,     
            # max_execution_time=3,
            max_iterations=4,
            return_intermediate_steps=True
            )


    def extract_list_from_response(self, response_text):
        """
        Extracts a Python list from a response string formatted with triple backticks.

        Args:
            response_text (str): The response text containing a Python list in triple backticks.

        Returns:
            list: The extracted list, or an empty list if parsing fails.
        """
        try:
            # Find the content within the triple backticks
            start_index = response_text.find("```") + 3
            end_index = response_text.rfind("```")
            
            if start_index == -1 or end_index == -1:
                raise ValueError("No triple backticks found in the response.")
            
            # Extract and clean the list text
            list_text = response_text[start_index:end_index].strip()
            
            # Safely evaluate the extracted text to a Python list
            extracted_list = ast.literal_eval(list_text)
            
            if not isinstance(extracted_list, list):
                raise ValueError("Extracted content is not a list.")
            
            return extracted_list
        except (ValueError, SyntaxError) as e:  # Catch specific exceptions
            print(f"Error while parsing the response: {e}")
            return []
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            return []

    
    def contextual_resume_content(self, resume_content):
        BASE_PROMPT = """
        I have an unstructured resume text that I need to convert into a structured list of meaningful strings for storage in a vector database. The task is to follow steps:

        step 1. Extract all relevant details from the resume without losing any information, even a single character.
        step 2. Split the content into meaningful, self-contained strings that capture individual pieces of information. Each string should provide enough context to be independently understandable and retrievable.
        step 3. Ensure that all sections (e.g., Education, Experience, Projects, Achievements, Skills, etc.) are represented in the list.
        step 4. Preserve the original formatting and content integrity while making the strings concise yet descriptive.

        Here is the unstructured resume text:
        {input}

        << OUTPUT FORMATTING >>
        strictly return the output as a Python list of strings, where each string represents a detailed, self-contained unit of information from the resume, formatted as a complete sentence. Ensure the description is meaningful, detailed, and provides enough context to be independently understandable and retrievable.

        << OUTPUT (must include ```[ at the start of the response) >>
        << OUTPUT (must end with ]```) >>

        NOTE: Do not include any other text before starting triple backticks
        """

        # Create the prompt template
        prompt = PromptTemplate(
            template=BASE_PROMPT,
            input_variables=["input"]
        )

        try:
            # Generate the response using the prompt and the model
            prompt_text = prompt.format(input=resume_content)
            response = self.llm_grok_chat.invoke(prompt_text)

            # Parse the response (assumed to be a string representation of a list)
            structured_content = self.extract_list_from_response(response.content)  
            
            # Validate the output is a list
            if not isinstance(structured_content, list):
                raise ValueError("LLM did not return a list as expected.")
            
            print("Extracted Resume Content: ", structured_content)
            return structured_content

        except Exception as e:
            print(f"Error during processing: {e}")
            return None

            
    # def get_knowledge_base(self):
    #     documents = [Document(page_content=self.resume_content)]
    #     graph_documents = self.llm_transformer.convert_to_graph_documents(documents)
    #     print(f"Nodes:{graph_documents[0].nodes}")
    #     print(f"Relationships:{graph_documents[0].relationships}")

    def parse_questions(self, response: str):        
        match = re.search(r"```(json)?(.*)", response, re.DOTALL)

        # If no match found, assume the entire string is a JSON string
        if match is None:
            json_str = response
        else:
            # If match found, use the content within the backticks
            json_str = match.group(2)

        # Strip whitespace and newlines from the start and end
        json_str = json_str.strip().strip("`")

        # Parse the JSON string into a Python dictionary
        parsed = ast.literal_eval(json_str)

        return parsed['questions']

    def transcripts_question(self, transcript: str):
        BASE_PROMPT = """\
        Given a raw text input (TRANSCRIPT), identify all the explicit questions included within the input transcript. Your task is to \
        extract and return only the exact questions present in the transcript as a list.

        << FORMATTING >>
        Return a markdown code snippet with a JSON object with key as questions and whose value is list containing questions,formatted to look like:
        {{output_schema}}

        << TRANSCRIPT >>
        {transcript}

        << OUTPUT (must include ```json at the start of the response) >>
        << OUTPUT (must end with ```) >>
        """

        OUTPUT_SCHEMA = """
        ```json
        {
        "questions": [
            {% for question in (possible question from transcript) %}
            "{{ question }}"{{ ',' if not loop.last else '' }},
            {% endfor %}
        ]
        }
        ```
        """

        prompt = PromptTemplate(
            template=BASE_PROMPT,
            input_variables=["transcript"],
            partial_variables={"output_schema": OUTPUT_SCHEMA}
        )

        # Format the final prompt using the given transcript
        final_prompt = prompt.format(transcript=transcript)

        # Print the final prompt
        print("Final Prompt Passed to LLM:")
        print(final_prompt)

        # Use the recommended API
        chain = RunnableSequence(prompt | self.llm_chat)
        
        try:
            llm_response = chain.invoke({"transcript": transcript})
            questions = self.parse_questions(llm_response.content)
            return questions
        except Exception as e:
            print(f"Error during processing: {e}")
            return None

    def answer_question(self, transcript: str) -> str:
        """
        Use the REACT agent to answer a question based on the transcript and resume content.
        """
        print('trnascript: ', transcript)
        questions=self.transcripts_question(transcript)
        if(len(questions) >= 1):
            try:
                input_query = f"""
                Answering the following questions delimited by triple backticks: ```{questions}``` \nNOTE:Your response should not exceed more than 50 words. Process each question at a time.
                """
                response = self.agent_executor.invoke({"input": input_query})
                return questions , response["output"]
            except Exception as e:
                return f"Error during response generation: {str(e)}"
        else: 
            return "Do not found any question.", "I apologize, but I'm unable to assist you as I do not found any relevant question to answer."

# Example Usage
if __name__ == "__main__":
    # Initialize the QA Agent
    qa_agent = QuestionAnsweringAgent(resume_content)

    # Sample transcript and question
    sample_transcript = "Hi nice to meet, i see in your resume you have work with AI projects and recommendation systems. can you Tell me about the experience with AI projects and also what is cpga in class 10th and 12th"

    # Get the answer
    question, answer = qa_agent.answer_question(sample_transcript)
    print("Answer:", answer)
