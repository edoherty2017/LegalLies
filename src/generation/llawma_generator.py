"""
Mistral 7B generator for legal question answering.

Uses Mistral 7B Instruct to fit in 16GB VRAM without quantization.
No quantization is used to preserve full weight access for steering.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .base_generator import BaseGenerator, GenerationConfig


class LlawmaGenerator(BaseGenerator):
    """
    Mistral 7B Instruct generator.

    Uses Mistral's 7B instruction-tuned model.
    Loads in float16 (~14GB VRAM) for steering compatibility.
    """

    def _load_model(self):
        """Load Mistral 7B in float16 (no quantization)."""
        # Load tokenizer
        print(f"Loading tokenizer from {self.config.model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name,
            trust_remote_code=True,
        )

        # Set padding token if not set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load model in float16 (NO quantization for steering)
        print(f"Loading model from {self.config.model_name}...")
        print("Loading in float16 (no quantization for steering compatibility)...")

        torch_dtype = torch.float16 if self.config.torch_dtype == "float16" else torch.bfloat16

        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            device_map="auto",
            torch_dtype=torch_dtype,
            trust_remote_code=True,
        )

        # Set to evaluation mode
        self.model.eval()

        print(f"Model loaded on: {self.model.device}")
        print(f"Memory footprint: ~{self.model.get_memory_footprint() / 1e9:.2f} GB")

    def _generate_response(self, prompt: str) -> str:
        """
        Generate response using Mistral 7B.

        Uses chat template for proper instruction formatting.

        Args:
            prompt: User message (will be wrapped in chat template)

        Returns:
            Generated answer text
        """
        # Build messages for chat template
        messages = [
            {"role": "user", "content": f"{self.config.system_prompt}\n\n{prompt}"}
        ]

        # Apply chat template
        input_ids = self.tokenizer.apply_chat_template(
            messages,
            return_tensors="pt",
            add_generation_prompt=True,
        ).to(self.model.device)

        # Create attention mask
        attention_mask = torch.ones_like(input_ids)

        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=self.config.max_new_tokens,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                top_k=self.config.top_k,
                do_sample=self.config.do_sample,
                repetition_penalty=self.config.repetition_penalty,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        # Decode response (skip input prompt)
        generated_text = self.tokenizer.decode(
            outputs[0][input_ids.shape[1]:],
            skip_special_tokens=True,
        )

        return generated_text.strip()


def create_llawma_generator(
    model_name: str = "mistralai/Mistral-7B-Instruct-v0.2",
    **kwargs
) -> LlawmaGenerator:
    """
    Factory function to create Mistral 7B Instruct generator.

    Args:
        model_name: HuggingFace model name (default: Mistral 7B Instruct v0.2)
        **kwargs: Additional GenerationConfig parameters

    Returns:
        Configured LlawmaGenerator

    Example:
        >>> generator = create_llawma_generator()
        >>> result = generator.generate("What is the statute of limitations for breach of contract?")
        >>> print(result.generated_answer)

    Note:
        Using Mistral 7B Instruct as the "legal model" alternative.
        No quantization is used to maintain full weight access for
        future steering implementations. Uses ~14GB VRAM in float16.
    """
    # Use legal-specific system prompt
    system_prompt = kwargs.pop("system_prompt", (
        "You are a legal AI assistant. Provide precise, well-reasoned legal analysis. "
        "Cite relevant statutes, cases, and legal principles when possible. "
        "If uncertain, acknowledge limitations."
    ))

    config = GenerationConfig(
        model_name=model_name,
        system_prompt=system_prompt,
        **kwargs
    )

    return LlawmaGenerator(config)
