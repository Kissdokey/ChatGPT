项目名称：基于本地知识库的ai查询

涉及技术：flask，langchain框架，python-docx

具体内容：通过python后端的flask框架，以及AI方向的langchain框架，实现了一个基于本地知识库的ai问询，前端仅仅由简单的html网页组成，后端通过python编写，使用python-docx将文档转换为文本，使用了langchain框架可以快速搭建向AI大模型发送请求的过程，其中包括为文档切片，建立索引，嵌入embedding，向模型发送请求，这些全部都集成在了langchain中并且暴露出相应的接口，最后通过flask框架分发请求，主要是页面的请求，返回前端页面的html文件，还有search请求，将向大模型返回的数据转发给前端。

解决了像chat-gpt无法有针对性地对大规模数据进行查询的问题。
# 服务端

服务端的提示词需要按照你的需求自定义，代码中的提示词是我自己的需求，你可以根据自己的需求修改。

## 安装依赖

`pip install -r requirements.txt`

## 运行

`python server.py`

## 访问

`http://localhost:3001`
