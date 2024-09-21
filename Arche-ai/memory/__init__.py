import os
import threading
import logging
import time
from llms import Cohere  # Replace with your actual LLM library

HISTORY_FOLDER = "MEMORIES"

class Memory:
    """Handles prompt generation, conversation history management, and memory summarization."""

    def __init__(
        self,
        llm: Cohere,
        status: bool = True,
        max_tokens: int = 8000,
        memory_filepath: str = os.path.join(HISTORY_FOLDER, "memory.txt"),
        chat_filepath: str = os.path.join(HISTORY_FOLDER, "chat.txt"),
        update_file: bool = True,
        history_offset: int = 10250,
        system_prompt: str = "You are a helpful AI assistant",
    ):
        self.status = status
        self.llm = llm
        self.max_tokens_to_sample = max_tokens
        self.chat_history = ""
        self.history_format = "\n%(role)s: %(content)s"
        self.update_file = update_file
        self.history_offset = history_offset
        self.prompt_allowance = 10
        self.memory_filepath = memory_filepath
        self.chat_filepath = chat_filepath
        self.system_prompt = system_prompt

        # Ensure history folder exists
        os.makedirs(HISTORY_FOLDER, exist_ok=True) 

        self.memory = self._load_memory(memory_filepath)
        self._load_conversation(chat_filepath)

        if self.system_prompt:
            self._write_to_chat_file(self.system_prompt + "\n", mode="w") # Start with a fresh chat file

        # Initialize 5-minute chat saving and summarization
        self.chat_buffer = []
        self.save_interval = 300  # 5 minutes in seconds
        self.summarization_thread = threading.Thread(target=self._summarize_and_save_chat)
        self.summarization_thread.daemon = True
        self.summarization_thread.start()

    def _write_to_chat_file(self, content: str, mode: str = "a") -> None:
        """Writes content to the chat file."""
        if self.chat_filepath:
            try:
                with open(self.chat_filepath, mode, encoding="utf-8") as fh:
                    fh.write(content)
            except IOError as e:
                logging.error(f"Error writing to chat file: {e}")

    def _load_conversation(self, filepath: str) -> None:
        """Loads the conversation history from a file."""
        if os.path.isfile(filepath):
            logging.debug(f"Loading conversation from '{filepath}'")
            try:
                with open(filepath, encoding="utf-8") as fh:
                    self.chat_history = fh.read().strip()
            except IOError as e:
                logging.error(f"Error loading conversation: {e}")
        else:
            logging.debug(f"Creating new chat-history file - '{filepath}'")
            open(filepath, "w", encoding="utf-8").close()

    def _load_memory(self, filepath: str) -> str:
        """Loads the memory from a file, each summary on a new line."""
        if os.path.exists(filepath):
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    return f.read().strip()
            except IOError as e:
                logging.error(f"Error loading memory: {e}")
        return ""

    def _trim_chat_history(self, chat_history: str, intro: str) -> str:
        """Trims the chat history to fit within the token limit."""
        total_length = len(intro) + len(chat_history)

        if total_length > self.history_offset:
            # Find the starting position of the second "User:" section
            first_user_index = chat_history.find("\nUser:")
            second_user_index = chat_history.find("\nUser:", first_user_index + 1)

            if second_user_index != -1:
                truncate_at = second_user_index + self.prompt_allowance
                return "... " + chat_history[truncate_at:]
        return chat_history

    def gen_complete_prompt(self, prompt: str, intro: str = "") -> str:
        """Generates a complete prompt using chat history and memory."""
        if self.status:
            incomplete_chat_history = self.chat_history + self.history_format % dict(
                role="User", content=prompt
            )
            trimmed_history = self._trim_chat_history(incomplete_chat_history, intro)

            # Include memory in the prompt
            complete_prompt = intro + "\n" + trimmed_history 
            if self.memory:
                complete_prompt += "\nMemory:\n" + self.memory 
            return complete_prompt 
        return prompt

    def update_chat_history(self, role: str, content: str, force: bool = False) -> None:
        """Updates chat history, adding timestamps only for user messages."""
        if not self.status and not force:
            return

        new_history = f"{role}: {content}"

        if self.update_file:
            self._write_to_chat_file(new_history + "\n")

        self.chat_history += new_history
        self.chat_buffer.append(new_history) 

    def add_message(self, role: str, content: str) -> None:
        """Adds a message to the chat history."""
        self.update_chat_history(role, content)

    def _summarize_and_save_chat(self):
        """Periodically summarizes and saves the chat buffer to memory."""
        while True:
            time.sleep(self.save_interval)
            if self.chat_buffer:  # Only summarize if there are messages in the buffer
                chat_summary = self._summarize_chat(self.chat_buffer)
                self._save_memory(chat_summary)
                self.chat_buffer.clear()

                # Clear the chat file after summarizing 
                try:
                    with open(self.chat_filepath, "w", encoding="utf-8") as chat_file:
                        # Write the system prompt again for a fresh start
                        if self.system_prompt:
                            chat_file.write(self.system_prompt + "\n")
                except IOError as e:
                    logging.error(f"Error clearing chat file: {e}")

    def _summarize_chat(self, chat_log: list) -> str:
        """Summarizes the chat log into a concise summary."""
        full_chat = "".join(chat_log)
        prompt = f"""
        You are a highly advanced AI assistant tasked with summarizing a conversation.
        Given the following conversation, create a concise summary, focusing on user requests or preferences, actions taken, and important information exchanged. 
        Limit your summary to 250 words. 

        Conversation:
        {full_chat}

        Summary:
        """
        summary = self.llm.run(prompt)
        return summary.strip()

    def _save_memory(self, summary: str) -> None:
        """Saves the memory summary to the file, each summary on a new line."""
        try:
            with open(self.memory_filepath, "a", encoding="utf-8") as f:  # Append to the file
                f.write(summary + "\n") 
            self.memory = self._load_memory(self.memory_filepath) # Reload memory to include the new summary
        except IOError as e:
            logging.error(f"Error saving memory: {e}")