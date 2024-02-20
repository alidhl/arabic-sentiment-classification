from model import Model
import gradio as gr

model = Model()

interface = gr.Interface(fn=model.predict,
                         inputs=gr.Textbox(lines=2, placeholder="Type Arabic text here..."),
                         outputs="text",
                         title="Arabic Sentiment Classification",
                         description="Enter an Arabic text to predict its sentiment.")
interface.launch()
