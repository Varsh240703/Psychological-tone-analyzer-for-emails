import pickle
import os

# Load the pipeline
def load_pipeline():
    pipeline_path = 'models/tone/tone_pipeline.pkl'

    # Check if pipeline exists
    if not os.path.exists(pipeline_path):
        raise FileNotFoundError("Pipeline model not found. Please train first.")

    # Load the trained pipeline
    with open(pipeline_path, 'rb') as pipeline_file:
        pipeline = pickle.load(pipeline_file)

    return pipeline

def predict_tone(text):
    try:
        pipeline = load_pipeline()

        # Make the prediction using the pipeline
        prediction = pipeline.predict([text])

        return prediction[0]

    except FileNotFoundError as e:
        print(f"Error: {e}")
        return "Error"
    except Exception as e:
        print(f"An error occurred during prediction: {e}")
        return "Error"

# Example usage
if __name__ == "__main__":
    sample_texts = [
        "I am feeling very happy today!",
        "This is a stressful situation."
    ]
    
    for text in sample_texts:
        result = predict_tone(text)
        print(f"Predicted Tone for '{text}': {result}")
