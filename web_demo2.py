import gradio as gr
from rag_bot import RAG_BOT,config_init

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
        print("history:",history)
        # print("psv:", past_key_values)
        for prompt, response, past_key_values, history in rag_bot.answer(input,history, past_key_values):
            chatbot[-1] = (parse_text(input), parse_text(response))
            history[-1] = (input,response)
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
    llm = "glm"#从这里选择模型@@@glm/qwen
    neo4j_user,neo4j_password = config_init(llm)
    # rag_bot = RAG_BOT(llm = llm, document_name = 'pipe',top_k=5,block="bychunk",chunk_len=200,neo4j_user=neo4j_user,neo4j_password=neo4j_password)
    rag_bot = RAG_BOT(llm = llm, document_name = 'somuut',top_k=10,block="byrow",neo4j_user=neo4j_user,neo4j_password=neo4j_password)
    demo = create_demo_multi_round(rag_bot)
    demo.launch(server_port=9010, share=True, show_api=False)

if __name__=='__main__':
    main()
