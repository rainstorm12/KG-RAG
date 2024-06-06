import gradio as gr
from rag_bot import RAG_BOT,config_init

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

def main():
    llm = "qwen"#从这里选择模型@@@glm/qwen
    neo4j_user,neo4j_password = config_init(llm)
    # rag_bot = RAG_BOT(llm = llm, document_name = 'pipe',top_k=5,block="bychunk",chunk_len=200,neo4j_user=neo4j_user,neo4j_password=neo4j_password)
    rag_bot = RAG_BOT(llm = llm, document_name = 'somuut',top_k=10,block="byrow",neo4j_user=neo4j_user,neo4j_password=neo4j_password)
    demo = create_demo_single_round(rag_bot)
    demo.launch(server_port=9010, share=True, show_api=False)

if __name__=='__main__':
    main()
