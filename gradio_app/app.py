import gradio as gr
import pandas as pd
import mlflow
from genai import generate_explanation

MODEL_NAME = "WineQualityModel"
model = mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}/1")


feature_names = [
    "fixed acidity", "volatile acidity", "citric acid", "residual sugar",
    "chlorides", "free sulfur dioxide", "total sulfur dioxide", "density",
    "pH", "sulphates", "alcohol"
]

MODEL_METRICS = "RMSE: 0.62 | MAE: 0.48 | R²: 0.61"

def predict_wine_quality(*inputs):
    input_dict = {name: val for name, val in zip(feature_names, inputs)}
    df = pd.DataFrame([input_dict])

    
    pred = model.predict(df)[0]
    
    explanation = generate_explanation(input_dict, pred)
    
    return f"Predicción de calidad: **{pred:.2f}** (escala 0-10)", explanation

inputs = [gr.Number(label=name, value=6.0 if "acidity" in name else 0.5 if name == "chlorides" else 10.0 if name == "alcohol" else 0.99 if name == "density" else 3.0 if name == "pH" else 0.5) for name in feature_names]

demo = gr.Interface(
    fn=predict_wine_quality,
    inputs=inputs,
    outputs=[
        gr.Markdown(label="Resultado"),
        gr.Textbox(label="Explicación Generada por IA", lines=4)
    ],
    title="Predicción de Calidad de Vino Blanco",
    description=f"Introduce las características químicas y obtén la calidad predicha junto a una explicación automática.",
    examples=[
        [7.0, 0.27, 0.36, 20.7, 0.045, 45, 170, 1.001, 3.0, 0.45, 8.8],  
        [6.5, 0.30, 0.32, 10.5, 0.050, 30, 120, 0.995, 3.3, 0.50, 10.0]
    ]
)

if __name__ == "__main__":
    demo.launch(server_name="localhost", server_port=7860)