"""
LargeForgeAI Python SDK - Basic Usage Examples
"""

from largeforge import LargeForgeClient

# Initialize client
client = LargeForgeClient()

# =============================================================================
# Text Completions
# =============================================================================

def basic_completion():
    """Basic text completion example."""
    response = client.completions.create(
        model="largeforge-7b",
        prompt="Explain the concept of machine learning in simple terms:",
        max_tokens=256,
        temperature=0.7
    )

    print("=== Basic Completion ===")
    print(response.choices[0].text)
    print(f"\nTokens used: {response.usage.total_tokens}")


def streaming_completion():
    """Streaming text completion example."""
    print("=== Streaming Completion ===")

    stream = client.completions.create(
        model="largeforge-7b",
        prompt="Write a short poem about coding:",
        max_tokens=128,
        temperature=0.9,
        stream=True
    )

    for chunk in stream:
        print(chunk.choices[0].text, end="", flush=True)
    print("\n")


# =============================================================================
# Chat Completions
# =============================================================================

def chat_completion():
    """Basic chat completion example."""
    response = client.chat.completions.create(
        model="largeforge-7b",
        messages=[
            {"role": "system", "content": "You are a helpful coding assistant."},
            {"role": "user", "content": "What is a decorator in Python?"}
        ],
        max_tokens=512,
        temperature=0.5
    )

    print("=== Chat Completion ===")
    print(response.choices[0].message.content)


def multi_turn_chat():
    """Multi-turn conversation example."""
    messages = [
        {"role": "system", "content": "You are a helpful math tutor."}
    ]

    print("=== Multi-turn Chat ===")

    # First turn
    messages.append({"role": "user", "content": "What is a derivative?"})
    response = client.chat.completions.create(
        model="largeforge-7b",
        messages=messages,
        max_tokens=256
    )
    assistant_msg = response.choices[0].message.content
    messages.append({"role": "assistant", "content": assistant_msg})
    print(f"User: What is a derivative?")
    print(f"Assistant: {assistant_msg}\n")

    # Second turn
    messages.append({"role": "user", "content": "Can you give me an example?"})
    response = client.chat.completions.create(
        model="largeforge-7b",
        messages=messages,
        max_tokens=256
    )
    print(f"User: Can you give me an example?")
    print(f"Assistant: {response.choices[0].message.content}")


# =============================================================================
# Expert Routing
# =============================================================================

def expert_routing():
    """Expert routing example."""
    print("=== Expert Routing ===")

    # Route a query
    route_result = client.router.route(
        query="How do I implement a binary search tree in Python?"
    )

    print(f"Query routed to: {route_result.expert}")
    print(f"Confidence: {route_result.confidence:.2%}")
    print("Alternatives:")
    for alt in route_result.alternatives:
        print(f"  - {alt.expert}: {alt.confidence:.2%}")


def route_and_generate():
    """Route and generate in one call."""
    print("\n=== Route and Generate ===")

    response = client.router.generate(
        query="Write a function to check if a number is prime",
        max_tokens=300
    )

    print(f"Routed to: {response.expert}")
    print(f"Response:\n{response.response}")


# =============================================================================
# Model Management
# =============================================================================

def list_models():
    """List available models."""
    print("=== Available Models ===")

    models = client.models.list()
    for model in models.data:
        print(f"  - {model.id} (owned by: {model.owned_by})")


def list_experts():
    """List registered experts."""
    print("\n=== Registered Experts ===")

    experts = client.experts.list()
    for expert in experts.experts:
        print(f"  - {expert.name}")
        print(f"    Description: {expert.description}")
        print(f"    Status: {expert.status}")
        print(f"    Domains: {', '.join(expert.domains)}")
        print()


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    # Run examples
    basic_completion()
    print()
    streaming_completion()
    print()
    chat_completion()
    print()
    multi_turn_chat()
    print()
    expert_routing()
    route_and_generate()
    print()
    list_models()
    list_experts()
