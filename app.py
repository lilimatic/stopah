from shiny import App, render, ui
from glille import *

app_ui = ui.page_fluid(
    ui.panel_title("Generalized Lili Score calculator (G-Lille)"),
    ui.input_switch("s","Treatment",0),
    ui.input_numeric("x1","Age",value=round(df['Age.at.randomisation..calc.'].mean(),0)),
    ui.input_numeric("x2","Albumin",value=round(df['Albumin...Merged'].mean(),2)),
    ui.input_numeric("x3","Day 0 Bilirubin",value=round(df['Bilirubin.Merged'].mean(),2)),
    ui.input_numeric("x4","Prothrombin time",value=round(df['Prothrombin.Time..patient....Merged'].mean(),2)),
    ui.input_numeric("x5","Day 7 Bilirubin",value=round(df['Bilirubin.day.7'].mean(),2)),
    ui.input_numeric("x6","Creatinine",value=round(df1['Creatinine..mg.dL....Merged'].mean(),2)),
    ui.output_text_verbatim("txt"),
)


def server(input, output, session):
    @render.text
    def txt():
        df1 = pd.DataFrame({'Feature1': [input.x1()],'Feature2': [input.x2()],'Feature3': [input.x3()],'Feature4':[input.x4()],
                           'Feature5': [input.x5()-input.x3()], 'Feature6': [int(input.x6() > 1.3)]})
        if input.s() == 1:
            return f"Probability of survival {round((gam_cv.predict_proba(df1)[0])/(1-gam_cv.predict_proba(df1)[0]),2)}" 
        else: return f"Probability of survivalß {round((gam_cv0.predict_proba(df1)[0])/(1-gam_cv0.predict_proba(df1)[0]),2)}"
    
app = App(app_ui, server)

