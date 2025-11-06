import mlflow
import ollama 

def generate_explanation(features_dict, prediction):
    features_text = "\n".join([f"- {k}: {v:.2f}" for k, v in features_dict.items()])
    prompt = f"""
    Eres un enólogo experto. Basado en las siguientes características de un vino blanco:
    {features_text}
    se predijo una calidad de {prediction:.2f} en una escala de 0 a 10.
    Proporciona una explicación clara, breve y técnica de por qué este vino tiene esa calidad.
    No inventes datos. Sé objetivo y basado en atributos comunes de calidad (equilibrio, acidez, alcohol, etc.).
    Retorna la explicación en español.
    """

    try:
        response = ollama.chat(model='llama3', messages=[{'role': 'user', 'content': prompt}])
        explanation = response['message']['content'].strip()
    except Exception as e:
        explanation = f"[Error en Gen AI: {str(e)}]. Usando explicación por defecto."
        explanation += f" Predicción: {prediction:.2f} basada en atributos del vino."

    with mlflow.start_run(nested=True):
        mlflow.set_tag("genai_model", "llama3")
        mlflow.log_text(explanation, "genai_explanation.txt")

    return explanation