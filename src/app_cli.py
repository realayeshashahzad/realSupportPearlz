import sys
from langchain_core.messages import HumanMessage, AIMessage
from src.config import settings
from src.utils.logging_setup import logger
from src.retrieval.vector_store import load_or_create_vector_store
from src.retrieval.retriever import RelevanceGatedRetriever
from src.chains.rag_chain import SupportPearlzAgent

def main():
    print("==================================================")
    print("      SupportPearlz AI Customer Knowledge Agent   ")
    print("==================================================")
    
    vector_store = load_or_create_vector_store(rebuild=False)
    retriever = RelevanceGatedRetriever(vector_store)
    agent = SupportPearlzAgent(retriever)
    
    chat_history = []
    
    print("\nSession started. Type 'exit' to quit or 'reset' to clear history.\n")
    
    while True:
        try:
            user_input = input("\nYou: ").strip()
            if not user_input:
                continue
            if user_input.lower() == "exit":
                print("Exiting SupportPearlz. Goodbye!")
                break
            if user_input.lower() == "reset":
                chat_history = []
                print("Conversation history reset.")
                continue

            result = agent.ask(user_input, chat_history)
            response = result["response"]
            
            print(f"\n[Rewritten Query]: {result['rewritten_query']}")
            print(f"\nBot > {response.answer}")
            print(f"Confidence: {response.confidence.upper()}")
            
            if response.sources:
                print("Sources:")
                for src in response.sources:
                    print(f"  - {src.source_file} [{src.location}]")
                    
            if response.uncovered_parts:
                print(f"Uncovered Note: {response.uncovered_parts}")

            chat_history.append(HumanMessage(content=user_input))
            chat_history.append(AIMessage(content=response.answer))

        except KeyboardInterrupt:
            print("\nSession interrupted.")
            break

if __name__ == "__main__":
    main()