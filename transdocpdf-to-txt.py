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