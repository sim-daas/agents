import os
import re
from datetime import datetime
import google.generativeai as genai

# Directory to store chat histories
HISTORY_DIR = "chat_histories"

def ensure_history_dir():
    """Ensures the history directory exists."""
    if not os.path.exists(HISTORY_DIR):
        try:
            os.makedirs(HISTORY_DIR)
            print(f"Created directory: {HISTORY_DIR}")
        except OSError as e:
            print(f"Error creating directory {HISTORY_DIR}: {e}")
            # If directory creation fails, we might not be able to save/load
            # For simplicity, we'll proceed but warn the user.

def save_chat_history(chat_session_history, filename):
    """Saves the chat history to a file."""
    ensure_history_dir()
    filepath = os.path.join(HISTORY_DIR, filename)
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            for message in chat_session_history:
                # message.parts is a list, usually with one Part containing text
                if message.parts and message.parts[0].text.strip():
                    role = message.role
                    # Replace newlines in message text with a placeholder (e.g., \\n)
                    # to keep each message on a single line in the file.
                    text = message.parts[0].text.replace('\n', '\\n')
                    f.write(f"role - {role}: {text}\n")
        print(f"Chat history saved to {filepath}")
    except IOError as e:
        print(f"Error saving chat history to {filepath}: {e}")
    except Exception as e:
        print(f"An unexpected error occurred while saving history: {e}")


def load_chat_history_from_file(filename):
    """Loads chat history from a file and prepares it for the SDK."""
    filepath = os.path.join(HISTORY_DIR, filename)
    history_for_sdk = []
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line_number, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue

                # Regex to capture role and text
                # Example: "role - user: Hello there"
                match = re.match(r"role - (user|model): (.*)", line, re.DOTALL)
                if match:
                    role = match.group(1)
                    # Restore newlines from the placeholder
                    text_content = match.group(2).replace('\\n', '\n')
                    history_for_sdk.append({'role': role, 'parts': [{'text': text_content}]})
                else:
                    print(f"Warning: Malformed line {line_number} in history file '{filepath}': {line}")
        print(f"Successfully loaded history from {filepath}")
        return history_for_sdk
    except FileNotFoundError:
        print(f"History file not found: {filepath}")
        return []
    except IOError as e:
        print(f"Error loading chat history from {filepath}: {e}")
        return []
    except Exception as e:
        print(f"An unexpected error occurred while loading history: {e}")
        return []

def list_history_files():
    """Lists available chat history files."""
    ensure_history_dir()
    try:
        files = [f for f in os.listdir(HISTORY_DIR) if f.startswith("chat_") and f.endswith(".txt")]
        return sorted(files, reverse=True) # Show newest first
    except FileNotFoundError:
        # This can happen if HISTORY_DIR was just created and listdir is called too quickly
        # or if there are permission issues.
        print(f"Could not list files in {HISTORY_DIR}. It might be empty or inaccessible.")
        return []
    except Exception as e:
        print(f"Error listing history files: {e}")
        return []


def run_chatbot():
    # 0. Ensure history directory exists at the start
    ensure_history_dir()

    # 1. Configure API Key
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        api_key = input("Please enter your Gemini API key: ")

    if not api_key:
        print("Error: GEMINI_API_KEY not provided. Exiting.")
        return

    try:
        genai.configure(api_key=api_key)
    except Exception as e:
        print(f"Error configuring Gemini API: {e}")
        return

    # 2. Ask to load history
    initial_sdk_history = []
    load_choice = input(f"Do you want to load a previous chat session? (y/N): ").strip().lower()
    if load_choice == 'y':
        available_files = list_history_files()
        if not available_files:
            print(f"No saved chat histories found in '{HISTORY_DIR}'.")
        else:
            print("\nAvailable chat histories:")
            for i, f_name in enumerate(available_files):
                print(f"{i+1}. {f_name}")

            while True:
                try:
                    file_num_choice_str = input(f"Enter the number of the history to load (or 0 to skip): ").strip()
                    if not file_num_choice_str: # User just pressed Enter
                        print("Skipping history load.")
                        break
                    file_num_choice = int(file_num_choice_str)
                    if file_num_choice == 0:
                        print("Skipping history load.")
                        break
                    if 1 <= file_num_choice <= len(available_files):
                        selected_file = available_files[file_num_choice - 1]
                        initial_sdk_history = load_chat_history_from_file(selected_file)
                        break
                    else:
                        print("Invalid selection. Please try again.")
                except ValueError:
                    print("Invalid input. Please enter a number or 0.")
                except Exception as e:
                    print(f"An error occurred during selection: {e}")
                    break # exit selection loop on unexpected error
    print("-" * 30)

    # 3. Create the model and start a chat session
    model_name = "gemini-1.5-flash-latest" # Or "gemini-pro"
    try:
        model = genai.GenerativeModel(model_name)
        # Start chat with the loaded (or empty) history
        chat = model.start_chat(history=initial_sdk_history)
        print(f"Chatbot initialized with model: {model_name}")
        if initial_sdk_history:
            print(f"Loaded {len(initial_sdk_history)//2} previous exchanges.") # Each exchange is user + model
        print("Type 'quit', 'exit', or 'bye' to end the chat.")
        print("Type 'history' to view the conversation history.")
        print("-" * 30)
    except Exception as e:
        print(f"Error initializing model or chat: {e}")
        return

    # 4. Chat loop
    try:
        while True:
            user_input = input("You: ").strip()

            if not user_input:
                continue

            if user_input.lower() in ["quit", "exit", "bye"]:
                print("Gemini: Goodbye!")
                break

            if user_input.lower() == "history":
                print("\n--- Chat History ---")
                if not chat.history: # chat.history is the live history object from the SDK
                    print("No history yet for this session (or none loaded).")
                else:
                    for message in chat.history:
                        if message.parts and message.parts[0].text.strip():
                            print(f'role - {message.role}: {message.parts[0].text}')
                print("--- End of History ---\n")
                continue

            try:
                print("Gemini: ", end="", flush=True)
                response_stream = chat.send_message(user_input, stream=True)
                for chunk in response_stream:
                    print(chunk.text, end="", flush=True)
                print()
            except Exception as e:
                print(f"\nError communicating with Gemini: {e}")

    finally:
        # 5. Save chat history on exit (even if loop breaks due to error)
        if chat and chat.history: # Ensure chat object exists and has history
            # Filter out any initial empty "user" part if history was empty and first input was made.
            # This can sometimes happen if the very first message is from the user.
            # The SDK usually handles this well, but a safeguard.
            valid_history_to_save = [m for m in chat.history if m.parts and m.parts[0].text.strip()]

            if valid_history_to_save:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"chat_{timestamp}.txt"
                save_chat_history(valid_history_to_save, filename)
            else:
                print("No chat interactions to save.")
        else:
            print("No chat session active or no history to save.")

        # 6. (Optional) Display final history after loop ends
        print("\n--- Final Chat Session Summary ---")
        if chat and chat.history:
            for message in chat.history:
                if message.parts and message.parts[0].text.strip():
                    print(f'role - {message.role}: {message.parts[0].text}')
        else:
            print("No history was recorded for this session.")
        print("--- End of Chat ---")


if __name__ == "__main__":
    print("Starting Gemini Chatbot...")
    # Before running, make sure you have the library installed:
    # pip install google-generativeai
    run_chatbot()
