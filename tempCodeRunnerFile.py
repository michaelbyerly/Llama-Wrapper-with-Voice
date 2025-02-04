            client = ollama.Client()
            models_response = ollama.list()
            
            # Log the type and content of the 'models_response'
            logging.info(f"Models response: {models_response} (Type: {type(models_response)})")
            
            # Check if 'models' key exists in the response and is a list
            if isinstance(models_response, dict) and 'models' in models_response:
                models = models_response['models']  # Access the list of models
                model_names = [model['name'] for model in models]
                eel.receive_ollama_models(model_names)
                logging.info("Ollama models sent to frontend.")
                return True
            else:
                logging.error("The response from ollama.list() does not contain a 'models' list.")
                return False