"""
Chat-only LLM config for HuggingFace + LangGraph.

- Prefers LOCAL_MODEL_PATH (no downloads; local_files_only=True).
- Optional fallback to LOCAL_MODEL_NAME (HF repo id) if no path provided.
- Optional online fallback via HuggingFaceHub when enabled.
- Injects a minimal chat_template if missing.
- Returns ChatHuggingFace with tokenizer passed in.

ENV (.env):
  USE_LOCAL_MODEL=true|false
  LOCAL_MODEL_PATH= D:/models/TinyLlama            # preferred local folder (contains config.json, tokenizer files, weights)
  LOCAL_MODEL_NAME= TinyLlama/TinyLlama-1.1B-Chat-v1.0
  ALLOW_ONLINE_FALLBACK=true|false
  HUGGINGFACEHUB_API_TOKEN= hf_...
  ONLINE_MODEL_NAME= HuggingFaceH4/zephyr-7b-beta
  TEMPERATURE=0.2
  MAX_NEW_TOKENS=256
  HF_DEVICE=cpu | 0  (optional; cpu or GPU index)
"""

import os
from pathlib import Path
from dotenv import load_dotenv
import torch

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    pipeline as hf_pipeline,
)

from langchain_huggingface.llms import HuggingFacePipeline
from langchain_huggingface import ChatHuggingFace
from langchain_community.llms import HuggingFaceHub

load_dotenv()

# -------------------- Flags --------------------
# Load environment variables and set default values for model configuration
USE_LOCAL_MODEL = os.getenv("USE_LOCAL_MODEL", "true").lower() == "true"
LOCAL_MODEL_PATH = os.getenv("LOCAL_MODEL_PATH", "").strip()
LOCAL_MODEL_NAME = os.getenv("LOCAL_MODEL_NAME", "TinyLlama/TinyLlama-1.1B-Chat-v1.0").strip()

ALLOW_ONLINE_FALLBACK = os.getenv("ALLOW_ONLINE_FALLBACK", "false").lower() == "true"
HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN", "").strip()
ONLINE_MODEL_NAME = os.getenv("ONLINE_MODEL_NAME", "HuggingFaceH4/zephyr-7b-beta").strip()

TEMPERATURE = float(os.getenv("TEMPERATURE", "0.2"))
MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", "256"))

# Optional device pin: HF_DEVICE="cpu" or "0"
_device_env = os.getenv("HF_DEVICE", "").strip()
if _device_env.lower() == "cpu":
    _DEVICE = -1
elif _device_env.isdigit():
    _DEVICE = int(_device_env)
else:
    _DEVICE = 0 if torch.cuda.is_available() else -1

# Check if accelerate is available for device mapping
try:
    import accelerate  # noqa: F401
    _HAS_ACCELERATE = True
except Exception:
    _HAS_ACCELERATE = False

# Cache for the chat model
_chat_model = None


# -------------------- Helpers --------------------
def _has_local_model_folder(p: str) -> bool:
    """
    Check if the given path looks like a valid HuggingFace model folder.

    Args:
        p (str): The path to check.

    Returns:
        bool: True if the path contains required model files, False otherwise.
    """
    if not p:
        return False
    root = Path(p)
    if not root.exists() or not root.is_dir():
        return False
    # Check for minimal required files in the folder
    needed = ["config.json", "tokenizer_config.json"]
    return all((root / f).exists() for f in needed)


def _resolve_local_source() -> str | None:
    """
    Resolve the local model folder if available.

    Returns:
        str | None: The path to the local model folder or None if not found.
    """
    if _has_local_model_folder(LOCAL_MODEL_PATH):
        return LOCAL_MODEL_PATH
    return None


def _ensure_chat_template(tokenizer):
    """
    Ensure the tokenizer has a chat_template attribute. Injects a minimal template if missing.

    Args:
        tokenizer: The tokenizer to check and modify.

    Returns:
        The tokenizer with a chat_template attribute.
    """
    if not getattr(tokenizer, "chat_template", None):
        tokenizer.chat_template = (
            "{% for message in messages %}"
            "{{ message['role'] | capitalize }}: {{ message['content'] }}\n"
            "{% endfor %}Assistant:"
        )
    return tokenizer


def _load_local_model_and_tokenizer():
    """
    Load a local model and tokenizer.

    Returns:
        tuple: The loaded model and tokenizer.

    Raises:
        RuntimeError: If the local model cannot be loaded.
    """
    folder = _resolve_local_source()
    model_source = folder if folder else LOCAL_MODEL_NAME

    print(
        f"[llm_config] Loading LOCAL model from "
        f"{'folder' if folder else 'repo'}: {model_source} "
        f"(device={_DEVICE}, accelerate={_HAS_ACCELERATE})"
    )

    tok_kwargs = {}
    mdl_kwargs = {}

    # If a local folder is used, force no network access
    if folder:
        tok_kwargs["local_files_only"] = True
        mdl_kwargs["local_files_only"] = True

    # Use GPU mapping if available and accelerate is present
    if torch.cuda.is_available() and _HAS_ACCELERATE:
        mdl_kwargs["device_map"] = "auto"
        try:
            mdl_kwargs["torch_dtype"] = torch.float16
        except Exception:
            pass

    tokenizer = AutoTokenizer.from_pretrained(model_source, **tok_kwargs)
    tokenizer = _ensure_chat_template(tokenizer)

    model = AutoModelForCausalLM.from_pretrained(model_source, **mdl_kwargs)
    return model, tokenizer


def _build_pipeline(model, tokenizer):
    """
    Build a text-generation pipeline with the given model and tokenizer.

    Args:
        model: The model to use.
        tokenizer: The tokenizer to use.

    Returns:
        The text-generation pipeline.
    """
    return hf_pipeline(
        task="text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=MAX_NEW_TOKENS,
        temperature=TEMPERATURE,
        device=_DEVICE,  # -1 = CPU, N = GPU index
    )


def _wrap_as_chat(pipe, tokenizer) -> ChatHuggingFace:
    """
    Wrap the pipeline as a ChatHuggingFace model.

    Args:
        pipe: The text-generation pipeline.
        tokenizer: The tokenizer with a chat_template.

    Returns:
        ChatHuggingFace: The wrapped chat model.
    """
    base_llm = HuggingFacePipeline(pipeline=pipe)
    return ChatHuggingFace(llm=base_llm, tokenizer=tokenizer)


# -------------------- Public API --------------------
def get_chat_model():
    """
    Get or initialize a ChatHuggingFace model.

    Returns:
        ChatHuggingFace: The chat model instance.

    Raises:
        RuntimeError: If no model can be initialized.
    """
    global _chat_model
    if _chat_model is not None:
        return _chat_model

    if USE_LOCAL_MODEL:
        try:
            model, tok = _load_local_model_and_tokenizer()
            pipe = _build_pipeline(model, tok)
            _chat_model = _wrap_as_chat(pipe, tok)
            return _chat_model
        except Exception as e:
            if not ALLOW_ONLINE_FALLBACK:
                raise RuntimeError(
                    "Failed to load local chat model. "
                    "Tips:\n"
                    f" - Set LOCAL_MODEL_PATH to your exact local model directory (currently: '{LOCAL_MODEL_PATH or '(empty)'}').\n"
                    " - Or leave LOCAL_MODEL_PATH empty and allow one-time download via LOCAL_MODEL_NAME.\n"
                    " - Or set ALLOW_ONLINE_FALLBACK=true to use HuggingFaceHub."
                ) from e

    # Online fallback (HuggingFaceHub)
    if ALLOW_ONLINE_FALLBACK:
        if not HUGGINGFACEHUB_API_TOKEN:
            raise RuntimeError("ALLOW_ONLINE_FALLBACK=true but HUGGINGFACEHUB_API_TOKEN is not set.")
        model_id = ONLINE_MODEL_NAME or LOCAL_MODEL_NAME
        print(f"[llm_config] Using ONLINE model via HuggingFaceHub: {model_id}")

        # Load & patch tokenizer for chat template, then hand it to ChatHuggingFace
        tok = AutoTokenizer.from_pretrained(model_id, use_auth_token=HUGGINGFACEHUB_API_TOKEN)
        tok = _ensure_chat_template(tok)

        hub_llm = HuggingFaceHub(
            repo_id=model_id,
            huggingfacehub_api_token=HUGGINGFACEHUB_API_TOKEN,
            model_kwargs={"temperature": TEMPERATURE, "max_new_tokens": MAX_NEW_TOKENS},
        )
        _chat_model = ChatHuggingFace(llm=hub_llm, tokenizer=tok)
        return _chat_model

    raise RuntimeError("No chat model could be initialized. Check your .env settings.")
