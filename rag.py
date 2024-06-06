from text2vec import SentenceModel, semantic_search
import sentence_transformers
import torch
import numpy as np
import os
import re
import json
import random
from http import HTTPStatus
import dashscope
import time
import gradio as gr
import py2neo
import configparser
from llm_api import llm_init,call_stream_with_messages,call_with_messages

def list_filenames(folder_path):
    """
    列出指定文件夹路径下的所有文件名。
    
    参数:
    folder_path -- 目标文件夹的路径字符串
    
    返回值:
    一个包含所有文件名的列表
    """
    # 确保传入的是一个字符串
    if not isinstance(folder_path, str):
        raise ValueError("folder_path 必须是字符串类型")
    
    # 初始化一个空列表来存储文件名
    filenames = []
    
    # 使用os.walk()函数遍历文件夹
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            # 构建完整的文件路径
            full_path = os.path.join(root, file)
            # 将文件名添加到列表中
            filenames.append(os.path.basename(full_path))
    
    return filenames

class RAG_BOT:
    
    def __init__(self,llm,document_name,top_k=5,block="bychunk",chunk_len=20,neo4j_user='neo4j',neo4j_password='neo4j'):
        self.llm = llm
        self.document_name = document_name
        self.top_k = top_k
        self.block = block
        self.chunk_len = chunk_len #分块大小
        self.max_kg_entity_neigh = 20 #最多查询的邻居数目
        self.max_single_entity_neigh = 10 #单一实体最多查询的邻居数目
        self.kg_neigh_topk = 2
        self.encoder_model  = SentenceModel('shibing624/text2vec-base-chinese')
        self.neo4j_graph = py2neo.Graph(
                            "http://localhost:7474", 
                            auth=(neo4j_user,neo4j_password)
                        )
        self.llm_model,self.llm_tokenizer = llm_init(self.llm)
        self.init_rag()
        self.init_kg_entity_rag()

    def init_rag(self):
        document_path='./data/'+self.document_name+'.txt'            
        if self.block=="byrow":#按行分割，每一行就是一个句子，适合整齐的文本数据
            vector_path = './data/'+self.document_name+"_"+self.block+'.pt'
            with open(document_path, 'r', encoding='utf-8') as file:
                document = file.readlines()
            self.document = [line.strip() for line in document]
        elif self.block=="bychunk": #按块分割，适合杂乱数据
            vector_path = './data/'+self.document_name+"_"+self.block+"_"+str(self.chunk_len)+'.pt'
            with open('./data/pipe.txt','r', encoding='utf-8') as file:
                doc_text = file.read()
            def split_string_by_length(string, length):
                return [string[i:i+length] for i in range(0, len(string), length)]
            self.document = split_string_by_length(doc_text, self.chunk_len)
            
        if os.path.exists(vector_path):
            self.embeddings = torch.load(vector_path)
            print("already load document vector")
        else:
            self.embeddings = self.encoder_model.encode(self.document, convert_to_tensor=True)
            torch.save(self.embeddings, vector_path)
    
    def init_kg_entity_rag(self):
        kg_entity_vector_path = './data/kg_entity.pt'
        self.kg_entity_list = self.neo4j_graph.run("MATCH (n) return n.name as title").data()
        self.kg_entity_list = [e["title"] for e in self.kg_entity_list]
        if os.path.exists(kg_entity_vector_path):
            self.kg_entity_embeddings = torch.load(kg_entity_vector_path)
            print("already load entity vector")
        else:
            self.kg_entity_embeddings = self.encoder_model.encode(self.kg_entity_list, convert_to_tensor=True)
            torch.save(self.kg_entity_embeddings, kg_entity_vector_path)

    def prompt_entity_extraction(self,sentence):
        entity_type = ",".join(["巡查项目","巡查内容","巡查方法","巡查周期"])
        entity_format="[实体1,实体2,实体3,...]"
        return f"\
说明：从给定的输入文本中提取实体，这些实体可能的类型包括“{entity_type}”,提取的实体最后以{entity_format}的格式回答。\n\
示例如三个反引号内所示：\n\
```\n\
输入文本：“管道泄水及冲洗水的封堵应该怎么做？”\n\
输出结果：[管道,泄水及冲洗水,封堵]\n\
```\n\
注意：输出结果是从用户的输入文本中提取实体,输出结果严格以{entity_format}的格式回答,不需要生成其他任何多余的内容。\n\
此时用户的输入文本是：“{sentence}”\n\
输出结果："

    def extract_entity(self,text):
        entities=[]
        entity_list = re.findall(r'\[(.*?)\]', text)
        if not entity_list:
            return entities
        entity_list = entity_list[0].split(",")
        for entity in entity_list:
            try:
                entity = entity.strip()
                entity = entity.strip("'")
                entity = entity.strip('"')
                # if entity_name in sentence:
                if entity:
                    entities.append(entity)
            except Exception as result:
                print(result)   
        return entities
    
    def entity_neigh_cypher(self,entity):
        result_child = self.neo4j_graph.run("MATCH (n {name:'"+ entity+"'})-[r]->(m)  \
                        RETURN id(r) As id, type(r) As type, \
                        n.name As title1 ,m.name As title2,\
                        id(n) As id1 ,id(m) As id2,\
                        head(labels(n)) As type1, head(labels(m)) As type2").data()
        result_parent = self.neo4j_graph.run("MATCH (n)-[r]->(m {name:'"+ entity+"'})  \
                                RETURN id(r) As id, type(r) As type, \
                                n.name As title1 ,m.name As title2,\
                                id(n) As id1 ,id(m) As id2,\
                                head(labels(n)) As type1, head(labels(m)) As type2").data()
        return result_child+result_parent
    
    def extract_entity_neigh(self,query,extract_mode="llm"):
        self.entity_neigh = []
        if not query:
            return None
        if extract_mode=="llm" or extract_mode=="llm+rag":
            prompt = self.prompt_entity_extraction(query)
            full_content = call_with_messages(prompt,self.llm,self.llm_model,self.llm_tokenizer)
            entities = self.extract_entity(full_content)
            #没有抽取到实体
            if not entities:
                return None
            result_list = []
            entity2neigh_num = {}
            for entity in entities:
                entity_neigh = self.entity_neigh_cypher(entity)
                entity2neigh_num[entity] = len(entity_neigh)
                result_list.append(entity_neigh[0:self.max_single_entity_neigh])
            #穿插合并
            result = []
            max_length = max(len(lst) for lst in result_list)
            for i in range(max_length):
                for k in range(0,len(result_list)):
                    if i < len(result_list[k]):
                        result.append(result_list[k][i])
            self.entity_neigh = result

        if extract_mode=="llm+rag":
            max_neigh = self.max_kg_entity_neigh
            if len(self.entity_neigh)<max_neigh:
                kg_neigh_topk = self.kg_neigh_topk #这里只检索最语义相近的节点
                for entity in entities:
                    if entity2neigh_num[entity]<self.max_single_entity_neigh:
                        query_embedding = self.encoder_model.encode(entity, convert_to_tensor=True)
                        sim_scores = sentence_transformers.util.cos_sim(query_embedding, self.kg_entity_embeddings)[0]
                        top_k_cat = torch.topk(sim_scores, k=kg_neigh_topk)
                        top_k_score,top_k_idx = top_k_cat[0],top_k_cat[1]
                        top_k_text = [self.kg_entity_list[top_k_idx[i]] for i in range(0,kg_neigh_topk)]
                        for k in range(0,kg_neigh_topk):
                            for neigh in self.entity_neigh_cypher(top_k_text[k]):
                                if neigh not in self.entity_neigh:
                                    if entity2neigh_num[entity]<self.max_single_entity_neigh:
                                        self.entity_neigh.append(neigh)
                                        entity2neigh_num[entity]+=1

    def ranking_api(self,query):
        if not query: 
            return None,None,None
        query_embedding = self.encoder_model.encode(query, convert_to_tensor=True)
        sim_scores = sentence_transformers.util.cos_sim(query_embedding, self.embeddings)[0]
        top_k_cat = torch.topk(sim_scores, k=self.top_k)
        top_k_score,top_k_idx = top_k_cat[0],top_k_cat[1]
        top_k_text = [self.document[top_k_idx[i]] for i in range(0,self.top_k)]
        return top_k_text,top_k_score,top_k_idx
    
    def generate_prompt(self,query):
        top_k_text,top_k_score,top_k_idx = self.ranking_api(query)
        knowledge_base = "\n".join(top_k_text)
        begin_prompt = f"\
你是一个专业领域的ai助手，你需要根据提供的知识库来与用户对话。\n\
提供的数据库如三个反引号内所示：\n\
```\n\
{knowledge_base}\n\
```\n"
        end_prompt=f"\
用户说的是：“{query}”\n\
给出对用户的回答："
        prompt_kg = ""
        if self.entity_neigh:
            max_len = self.max_kg_entity_neigh
            prompt_kg = "提供的知识图谱库如三个短横线内所示：\n---\n"
            result = ["->".join([r['title1'],r["type"],r["title2"]]) for r in self.entity_neigh[0:max_len]]
            prompt_kg += "\n".join(result)
            prompt_kg += "\n---\n"
        self.answer_prompt = begin_prompt+prompt_kg+end_prompt

    def answer(self,query):
        self.extract_entity_neigh(query,extract_mode="llm+rag")
        self.generate_prompt(query)
        yield from call_stream_with_messages(self.answer_prompt,self.llm,self.llm_model,self.llm_tokenizer)

def config_init(llm):
    config = configparser.ConfigParser()
    config.read('data/config.ini')
    neo4j_user = config['Neo4j']['user']
    neo4j_password = config['Neo4j']['password']
    if llm == "qwen":
        llm_api_key = config['Qwen']['api_key']
        dashscope.api_key = llm_api_key
    return neo4j_user,neo4j_password

def create_demo_single_round(rag_bot):
    with gr.Blocks(title="检索增强单轮对话demo", css="footer {visibility: hidden}", theme="default") as demo:
        gr.HTML("""<h1 align="center">管廊运维RAG</h1>""")
        with gr.Row():
            with gr.Column():
                query = gr.Textbox(label="用户输入", placeholder="请输入...", max_lines=30, lines=5,interactive=True)
                btn_answer = gr.Button("生成回答", variant="primary")
                prompt = gr.Textbox(label="对应的Prompt", placeholder="输入后确认", max_lines=30, lines=15, interactive=True)
            with gr.Column():
                answer = gr.Textbox(label="AI回复", placeholder="输入后确认", max_lines=30, lines=25, interactive=True)
        history = gr.State([])
        past_key_values = gr.State(None)
        btn_answer.click(rag_bot.answer,inputs=[query],outputs=[prompt,answer,past_key_values, history])
    return demo

def create_demo_multi_round(rag_bot):
    import mdtex2html
    def postprocess(self, y):
        if y is None:
            return []
        for i, (message, response) in enumerate(y):
            y[i] = (
                None if message is None else mdtex2html.convert((message)),
                None if response is None else mdtex2html.convert(response),
            )
        return y


    gr.Chatbot.postprocess = postprocess


    def parse_text(text):
        """copy from https://github.com/GaiZhenbiao/ChuanhuChatGPT/"""
        lines = text.split("\n")
        lines = [line for line in lines if line != ""]
        count = 0
        for i, line in enumerate(lines):
            if "```" in line:
                count += 1
                items = line.split('`')
                if count % 2 == 1:
                    lines[i] = f'<pre><code class="language-{items[-1]}">'
                else:
                    lines[i] = f'<br></code></pre>'
            else:
                if i > 0:
                    if count % 2 == 1:
                        line = line.replace("`", "\`")
                        line = line.replace("<", "&lt;")
                        line = line.replace(">", "&gt;")
                        line = line.replace(" ", "&nbsp;")
                        line = line.replace("*", "&ast;")
                        line = line.replace("_", "&lowbar;")
                        line = line.replace("-", "&#45;")
                        line = line.replace(".", "&#46;")
                        line = line.replace("!", "&#33;")
                        line = line.replace("(", "&#40;")
                        line = line.replace(")", "&#41;")
                        line = line.replace("$", "&#36;")
                    lines[i] = "<br>"+line
        text = "".join(lines)
        return text


    def predict(input, chatbot, history, past_key_values):
        chatbot.append((parse_text(input), ""))
        for prompt, response, past_key_values, history in rag_bot.answer(input):
            chatbot[-1] = (parse_text(input), parse_text(response))

            yield prompt, chatbot, history, past_key_values

    def reset_user_input():
        return gr.update(value='')
    
    def reset_state():
        return [], [], None
    
    with gr.Blocks(title="检索增强多轮对话demo") as demo:
        gr.HTML("""<h1 align="center">管廊运维RAG</h1>""")

        chatbot = gr.Chatbot()
        with gr.Row():
            with gr.Column(scale=4):
                with gr.Column(scale=8):
                    user_input = gr.Textbox(show_label=False, placeholder="Input...", max_lines=10, lines=10)
                with gr.Column(min_width=32, scale=1):
                    submitBtn = gr.Button("Submit", variant="primary")
            with gr.Column(scale=1):
                prompt = gr.Textbox(label="Prompt", max_lines=10, lines=10)
                emptyBtn = gr.Button("Clear History")

        history = gr.State([])
        past_key_values = gr.State(None)

        submitBtn.click(predict,inputs=[user_input, chatbot, history, past_key_values],outputs=[prompt, chatbot, history, past_key_values], show_progress=True)
        submitBtn.click(reset_user_input, [], [user_input])

        emptyBtn.click(reset_state, outputs=[chatbot, history, past_key_values], show_progress=True)
    
    return demo

def main():
    llm = "glm"#从这里选择模型@@@
    neo4j_user,neo4j_password = config_init(llm)
    # rag_bot = RAG_BOT(llm = llm, document_name = 'pipe',top_k=5,block="bychunk",chunk_len=200,neo4j_user=neo4j_user,neo4j_password=neo4j_password)
    rag_bot = RAG_BOT(llm = llm, document_name = 'somuut',top_k=10,block="byrow",neo4j_user=neo4j_user,neo4j_password=neo4j_password)
    demo = create_demo_single_round(rag_bot)
    # demo = create_demo_multi_round(rag_bot)
    demo.launch(server_port=9010, share=True, show_api=False)

if __name__=='__main__':
    main()
