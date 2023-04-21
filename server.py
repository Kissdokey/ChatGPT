from flask import Flask
from flask import render_template
from flask import request
# from langchain.document_loaders import UnstructuredURLLoader
import os
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import TokenTextSplitter
from langchain.chains import ChatVectorDBChain
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import DirectoryLoader
from langchain.prompts import PromptTemplate
import jieba as jb
import openai
import pdfplumber
import docx
# 用pdf文件解析器读取文件
with pdfplumber.\
    open(r'D:\\chatGPT\\test\document.ai\\code\\server\\data\\default.pdf') \
        as f:
    for page in f.pages:
        text = page.extract_text()
        txt_f = open(r'./data/111.txt', mode='a', encoding='utf-8')  # 创建txt文件
        txt_f.write(text)  
        txt_f.close()


# docx测试
f2 = docx.\
    Document(r"D:\\chatGPT\\test\document.ai\\code\\server\\data\\222.docx")
for para in f2.paragraphs:   # 用for循环读取文件每一段
    txt_f2 = open(r'./data/222.txt', mode='a', encoding='utf-8')  # 创建txt文件
    txt_f2.write(para.text)     # 写入txt文件
    txt_f2.close()

openai.api_base = "https://ai-proxy.ksord.com/api.openai.com/v1"
os.environ["OPENAI_API_KEY"] = "UkibOXaswamuXpn3DlcoISs8OvNH2x9O"
app = Flask(__name__)
template = """您是提供有用建议的AI助手。你得到了以下长文档的提取部分和一个问题。根据提供的上下文进行交流式回答。
  您应该只提供引用以下上下文的超链接。不要编造超链接。
  如果您在下面的上下文中找不到答案，只需说“嗯，我不确定。”不要试图编造答案。
  如果问题与上下文无关，请礼貌地回答。你只能回答与上下文相关的问题。我要求你返回的结果全部用中文。
Question: {question}
=========
{context}
=========
Answer in Markdown:
"""
prompt_template = PromptTemplate(input_variables=["question", "context"],
                                 template=template)


def findtxt(path, ret, type):
    """Finding the *.txt file in specify path"""
    filelist = os.listdir(path)
    for filename in filelist:
        de_path = os.path.join(path, filename)
        if os.path.isfile(de_path):
            if de_path.endswith(type):
                ret.append(filename)
        else:
            findtxt(de_path, ret, type)


ret = []
type = ".txt"
root = "D:\\chatGPT\\test\\document.ai\\code\\server\\data"
findtxt(root, ret, type)
for file in ret:
    print(file)
    my_file = f"./data/{file}"
    with open(my_file, "r", encoding='utf-8') as f:
        data = f.read()

    # 对中文文档进行分词处理
    cut_data = " ".join([w for w in list(jb.cut(data))])
    # 分词处理后的文档保存到data文件夹中的cut子文件夹中
    cut_file = f"./cut-data/cut/cut_{file}"
    with open(cut_file, 'w', encoding='utf-8') as f:
        f.write(cut_data)
        f.close()
        # 加载文档
loader = DirectoryLoader('./cut-data/cut', glob='**/*.txt')
docs = loader.load()
# 测试url
# url1 = "http://en.people.cn/n3/2023/0420/c90000-20008343.html"
# url2 = "http://en.people.cn/n3/2023/0420/c90000-20008284.html"

# 文档切块
text_splitter = TokenTextSplitter(chunk_size=1000, chunk_overlap=0)
doc_texts = text_splitter.split_documents(
    # UnstructuredURLLoader(urls=[url1, url2]).load())
    docs)
# 调用openai Embeddings
os.environ["OPENAI_API_KEY"] = "UkibOXaswamuXpn3DlcoISs8OvNH2x9O"
embeddings = OpenAIEmbeddings(openai_api_key=os.environ["OPENAI_API_KEY"])
# 向量化
vectordb = Chroma.from_documents(doc_texts,
                                 embeddings,
                                 persist_directory="./cut-data/cut")
vectordb.persist()
# 创建聊天机器人对象chain
chain = ChatVectorDBChain.from_llm(ChatOpenAI(temperature=0,
                                   model_name="gpt-3.5-turbo"),
                                   vectordb,
                                   qa_prompt=prompt_template,
                                   return_source_documents=True)


def query(question):
    chat_history = []
    result = chain({"question": question, "chat_history": chat_history})
    return result


@app.route('/')
def hello_world():
    return render_template('index.html')


@app.route('/search', methods=['POST'])
def search():
    data = request.get_json()
    search = data['search']

    res = query(search)

    return {
        "code": 200,
        "data": {
            "search": search,
            "answer": res["answer"],
        }
    }


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3001)
