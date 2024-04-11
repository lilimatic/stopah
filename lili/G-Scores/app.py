from shiny import App, render, ui
from glille import *

app_ui = ui.page_fluid(
    ui.panel_title("Generalized Lili Score (G-Lille)"),
    #ui.input_switch("s","Treatment"),
    ui.input_numeric("x1","Age",value=0),
    ui.input_numeric("x2","Albumin",value=0),
    ui.input_numeric("x3","Day 0 Bilirubin",value=0),
    ui.input_numeric("x4","Prothrombin time",value=0),
    ui.input_numeric("x5","Day 7 Bilirubin",value=0),
    ui.input_numeric("x6","Renal insufficiency",value=0),
    ui.output_text_verbatim("txt"),
)


def server(input, output, session):
    @render.text
    def txt():
        df = pd.DataFrame({'Feature1': [input.x1()],'Feature2': [input.x2()],'Feature3': [input.x3()],'Feature4':[input.x4()],
                           'Feature5': [input.x5()-input.x3()], 'Feature6': [input.x6()]})
        return f"G-Lille Score {df.sum().sum()}"
    
##gamcv.predict_proba(user_data)[0]


app = App(app_ui, server)


