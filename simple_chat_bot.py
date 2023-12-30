# Assuming necessary packages are installed and model is properly set up.
import os
from typing import List, Optional

import fire

from llama import Dialog, Llama


def main(
        ckpt_dir: str = "/mnt/d/models/llama-2/chat/llama-2-7b-chat", # /mnt/d/models/llama-2/chat/llama-2-7b-chat #D:\\models\\llama-2\\chat\\llama-2-7b-chat
        tokenizer_path: str = "/mnt/d/models/llama-2/chat/tokenizer.model", #/mnt/d/models/llama-2/chat/tokenizer.model #D:\\models\\llama-2\\chat\\tokenizer.model
        temperature: float = 0.6,
        top_p: float = 0.9,
        max_seq_len: int = 512,
        max_batch_size: int = 1,  # Single batch since single GPU
        max_gen_len: Optional[int] = None,
):
    """
    Entry point of the program for generating text using a pretrained model.
    """
    # Initialize the generator with the given parameters.
    generator = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
    )

    try:
        # Initial system prompt, if any
        system_prompt = [{
            "role": "system",
            "content": """Your name is PabloBot. You are wholesome bot trained on many wholesome subreddit. And Pornhub comments. You will not State your an AI or assistant, you are a _Bot_'"""}]

        while True:
            # Collecting user input to initiate dialog.
            user_content = input("Enter your message (Ctrl+C to exit): ")
            if user_content.strip() == "":
                print("Please enter a valid message.")
                continue

            # Constructing the dialog for this iteration
            dialogs: List[Dialog] = [
                system_prompt + [{"role": "user", "content": user_content}]
            ]

            # Generating dialog completion using the model for the given dialog.
            results = generator.chat_completion(
                dialogs,  # Dialog with the user's latest message.
                max_gen_len=max_gen_len,
                temperature=temperature,
                top_p=top_p,
            )

            # Extract and print the generated response
            assistant_response = results[0]['generation']['content']
            print("\nAssistant: ", assistant_response)

            print("\n==================================\n")

    except KeyboardInterrupt:
        print("\nExiting the script. Goodbye!")


if __name__ == "__main__":
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29500")

    fire.Fire(main)
