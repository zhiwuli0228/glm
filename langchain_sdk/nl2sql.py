from operator import itemgetter

from langchain.chains.sql_database.query import create_sql_query_chain
from langchain_community.tools import QuerySQLDataBaseTool
from langchain_community.utilities import SQLDatabase
from dotenv import load_dotenv, find_dotenv
import os

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI

_ = load_dotenv(find_dotenv())
os.environ[
    'USER_AGENT'] = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36'
api_key = os.environ['ZHIPUAI_API_KEY']

llm = ChatOpenAI(
    model='glm-4-plus',
    temperature=0.8,
    api_key=api_key,
    base_url="https://open.bigmodel.cn/api/paas/v4/"
)

HOST = '47.119.150.113'
POET = 3306
USER = 'root'
PASSWORD = 'root'
TABLESPACE = 'user'

mysql_url = 'mysql+mysqldb://{}:{}@{}:{}/{}?charset=utf8mb4'.format(USER, PASSWORD, HOST, POET, TABLESPACE)

db = SQLDatabase.from_uri(mysql_url)

# print(db.get_usable_table_names())
# print(db.run('select * from student;'))

test_chain = create_sql_query_chain(llm, db)
# resp = test_chain.invoke({'question': '请问：学生表中有多少条数据'})

# print(resp)

# 创建一个执行sql语句的工具
execute_sql_tool = QuerySQLDataBaseTool(db=db)

# 创建一个模版
answer_prompt = PromptTemplate.from_template(
    """
    给定一下用户问题，SQL语句和SQL执行后的结果，中的回答用户问题。
    Question: {question}
    SQL Query: {query}
    SQL Result: {result}
    回答：
    """
)

chain = (RunnablePassthrough.assign(query=test_chain).assign(result=itemgetter('query') | execute_sql_tool)
         | answer_prompt
         | llm
         | StrOutputParser()
         )

resp = chain.invoke({'question': '请问学生表中有多少个学生'})

print(resp)
