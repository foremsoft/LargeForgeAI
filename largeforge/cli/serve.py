"""Inference server CLI commands."""

from typing import Optional

import click

from largeforge.utils import get_logger

logger = get_logger(__name__)


@click.command()
@click.argument("model_path", type=str)
@click.option("--host", "-h", type=str, default="0.0.0.0", help="Host to bind to")
@click.option("--port", "-p", type=int, default=8000, help="Port to bind to")
@click.option("--backend", type=click.Choice(["auto", "transformers", "vllm"]),
              default="auto", help="Inference backend")
@click.option("--dtype", type=click.Choice(["auto", "float16", "bfloat16", "float32"]),
              default="auto", help="Model data type")
@click.option("--device", type=str, default="auto", help="Device to use (auto, cpu, cuda, cuda:0)")
@click.option("--tensor-parallel", type=int, default=1, help="Tensor parallel size (vLLM)")
@click.option("--gpu-memory", type=float, default=0.9, help="GPU memory utilization (vLLM)")
@click.option("--quantization", type=click.Choice(["none", "4bit", "8bit", "awq", "gptq"]),
              default="none", help="Quantization method")
@click.option("--workers", "-w", type=int, default=1, help="Number of workers")
@click.option("--reload/--no-reload", default=False, help="Enable auto-reload for development")
@click.option("--trust-remote-code/--no-trust-remote-code", default=False,
              help="Trust remote code in model")
@click.pass_context
def start(
    ctx: click.Context,
    model_path: str,
    host: str,
    port: int,
    backend: str,
    dtype: str,
    device: str,
    tensor_parallel: int,
    gpu_memory: float,
    quantization: str,
    workers: int,
    reload: bool,
    trust_remote_code: bool,
) -> None:
    """Start an inference server for a model.

    MODEL_PATH: HuggingFace model ID or local path

    The server provides OpenAI-compatible endpoints:
    - POST /v1/completions
    - POST /v1/chat/completions
    - GET /v1/models

    Example:
        largeforge serve start meta-llama/Llama-2-7b-chat-hf --port 8000
    """
    click.echo(f"Starting inference server...")
    click.echo(f"Model: {model_path}")
    click.echo(f"Backend: {backend}")
    click.echo(f"Endpoint: http://{host}:{port}")

    # Build config
    from largeforge.config import InferenceConfig

    config_kwargs = {
        "backend": backend if backend != "auto" else "transformers",
        "dtype": dtype,
        "trust_remote_code": trust_remote_code,
    }

    if quantization != "none":
        if quantization == "4bit":
            config_kwargs["quantization"] = "4bit"
        elif quantization == "8bit":
            config_kwargs["quantization"] = "8bit"
        else:
            config_kwargs["quantization"] = quantization

    if backend == "vllm" or (backend == "auto" and tensor_parallel > 1):
        config_kwargs["tensor_parallel_size"] = tensor_parallel
        config_kwargs["gpu_memory_utilization"] = gpu_memory

    config = InferenceConfig(**config_kwargs)

    # Start server
    click.echo("Loading model and starting server...")
    try:
        from largeforge.inference.server import run_server

        run_server(
            model_path=model_path,
            host=host,
            port=port,
            backend=backend,
            config=config,
            workers=workers,
            reload=reload,
        )
    except ImportError as e:
        raise click.ClickException(
            f"Missing dependencies: {e}\n"
            "Install with: pip install fastapi uvicorn"
        )
    except Exception as e:
        logger.error(f"Server failed: {e}")
        raise click.ClickException(str(e))


@click.command()
@click.argument("model_path", type=str)
@click.option("--prompt", "-p", type=str, help="Input prompt (or use stdin)")
@click.option("--max-tokens", "-m", type=int, default=256, help="Maximum tokens to generate")
@click.option("--temperature", "-t", type=float, default=0.7, help="Sampling temperature")
@click.option("--top-p", type=float, default=0.9, help="Top-p sampling")
@click.option("--top-k", type=int, default=50, help="Top-k sampling")
@click.option("--repetition-penalty", type=float, default=1.1, help="Repetition penalty")
@click.option("--stream/--no-stream", default=True, help="Stream output tokens")
@click.option("--backend", type=click.Choice(["auto", "transformers", "vllm"]),
              default="auto", help="Inference backend")
@click.option("--device", type=str, default="auto", help="Device to use")
@click.option("--dtype", type=click.Choice(["auto", "float16", "bfloat16"]),
              default="auto", help="Model data type")
@click.option("--system", "-s", type=str, help="System prompt for chat mode")
@click.option("--chat/--no-chat", default=False, help="Interactive chat mode")
@click.pass_context
def generate(
    ctx: click.Context,
    model_path: str,
    prompt: Optional[str],
    max_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    repetition_penalty: float,
    stream: bool,
    backend: str,
    device: str,
    dtype: str,
    system: Optional[str],
    chat: bool,
) -> None:
    """Generate text from a model.

    MODEL_PATH: HuggingFace model ID or local path

    Examples:
        # Single generation
        largeforge serve generate gpt2 -p "Once upon a time"

        # Interactive chat
        largeforge serve generate meta-llama/Llama-2-7b-chat-hf --chat

        # Pipe input
        echo "Hello, world!" | largeforge serve generate gpt2
    """
    import sys

    # Get prompt from argument or stdin
    if prompt is None:
        if not sys.stdin.isatty():
            prompt = sys.stdin.read().strip()
        elif not chat:
            raise click.ClickException("Please provide --prompt or use stdin")

    # Initialize generator
    click.echo(f"Loading model: {model_path}", err=True)

    from largeforge.inference import TextGenerator

    generator = TextGenerator(
        model_path=model_path,
        backend=backend,
        device=device,
        dtype=dtype,
    )

    try:
        generator.load()
        click.echo("Model loaded!", err=True)

        if chat:
            # Interactive chat mode
            _run_chat(generator, system, max_tokens, temperature, top_p, stream)
        else:
            # Single generation
            _generate_once(
                generator, prompt, max_tokens, temperature, top_p,
                top_k, repetition_penalty, stream
            )

    finally:
        generator.unload()


def _generate_once(
    generator,
    prompt: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    repetition_penalty: float,
    stream: bool,
) -> None:
    """Generate text once."""
    import sys

    if stream:
        for chunk in generator.generate_stream(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
        ):
            click.echo(chunk, nl=False)
            sys.stdout.flush()
        click.echo()  # Final newline
    else:
        text = generator.generate(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
        )
        click.echo(text)


def _run_chat(
    generator,
    system: Optional[str],
    max_tokens: int,
    temperature: float,
    top_p: float,
    stream: bool,
) -> None:
    """Run interactive chat mode."""
    import sys

    click.echo("Entering chat mode. Type 'exit' or 'quit' to end.", err=True)
    click.echo("-" * 40, err=True)

    messages = []
    if system:
        messages.append({"role": "system", "content": system})

    while True:
        try:
            # Get user input
            user_input = click.prompt("You", type=str)

            if user_input.lower() in ("exit", "quit", "/exit", "/quit"):
                click.echo("Goodbye!", err=True)
                break

            if user_input.lower() in ("/clear", "/reset"):
                messages = []
                if system:
                    messages.append({"role": "system", "content": system})
                click.echo("Chat history cleared.", err=True)
                continue

            # Add user message
            messages.append({"role": "user", "content": user_input})

            # Generate response
            click.echo("Assistant: ", nl=False)

            if stream and hasattr(generator.backend, "tokenizer"):
                # Stream the response
                response_text = ""
                # For chat, we need to format and generate
                response_text = generator.chat(
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                )
                click.echo(response_text)
            else:
                response_text = generator.chat(
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                )
                click.echo(response_text)

            # Add assistant message to history
            messages.append({"role": "assistant", "content": response_text})

        except (KeyboardInterrupt, EOFError):
            click.echo("\nGoodbye!", err=True)
            break
        except Exception as e:
            click.echo(f"\nError: {e}", err=True)
