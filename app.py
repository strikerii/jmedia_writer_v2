from flask import Flask, request, jsonify
from flask_cors import CORS
import google.generativeai as genai
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Initialize Flask application
app = Flask(__name__)
CORS(app)  # Enable Cross-Origin Resource Sharing for all routes

# Load the API key from environment variables
api_key = os.getenv("GENAI_API_KEY")
if not api_key:
    raise EnvironmentError("API key for Google Generative AI not set in .env file.")

# Configure Google Generative AI with the API key
try:
    genai.configure(api_key=api_key)
except Exception as e:
    raise RuntimeError(f"Failed to configure Google Generative AI: {e}")

# Generation settings for the AI model
generation_config = {
    "temperature": 1,
    "top_p": 0.95,
    "top_k": 40,
    "max_output_tokens": 8192,
    "response_mime_type": "text/plain",
}

# Initialize the AI model
try:
    model = genai.GenerativeModel(
        model_name="gemini-2.0-flash-exp",  # Replace with the correct model name if needed
        generation_config=generation_config,
    )
except Exception as e:
    raise RuntimeError(f"Failed to initialize the AI model: {e}")

# Initialize a chat session with an empty history
chat_session = model.start_chat(history=[])

@app.route('/process', methods=['POST'])
def process_request():
    """Handles incoming POST requests for generating or summarizing text."""
    global chat_session
    try:
        data = request.json
        if not data:
            return jsonify({'error': 'Request body must be in JSON format.'}), 400

        action = data.get('action', '').lower()

        if action == 'reset':
            # Reset the chat session
            chat_session = model.start_chat(history=[])
            return jsonify({'message': 'Chat history reset successfully'}), 200

        elif action == 'generate':
            # Generate text based on the provided prompt
            prompt = data.get('keywords', '').strip()
            if not prompt:
                return jsonify({'error': 'Keywords (prompt) are required for text generation.'}), 400

            response = chat_session.send_message(prompt)
            return jsonify({
                'length': len(response.text.split()),
                'generated_text': response.text
            }), 200

        elif action == 'summarize':
            # Summarize the provided article
            article = data.get('article', '').strip()
            if not article:
                return jsonify({'error': 'Article text is required for summarization.'}), 400

            summary_prompt = f"Summarize the following article:\n{article}"
            response = chat_session.send_message(summary_prompt)
            return jsonify({'summary': response.text}), 200

        else:
            return jsonify({'error': 'Invalid action. Use "generate", "summarize", or "reset".'}), 400

    except Exception as e:
        return jsonify({'error': f"An error occurred: {str(e)}"}), 500


# Entry point for the Flask application
if __name__ == '__main__':
    # Run the app in debug mode (not recommended for production)
    app.run(host='0.0.0.0', port=5000, debug=True)
